/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel_operator.h"
#include <stdio.h>
#include "types.h"
#include <string>
#include <stdexcept>

template <typename scalar_t, typename slot_t> class MultiLayerPagedKVCopyV2 {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline MultiLayerPagedKVCopyV2()
    {
    }

    __aicore__ inline void init(GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, GM_ADDR slotmappings,
                                const int64_t hiddenDims, const int32_t numLayers, const int64_t pageBuffSize,
                                const int32_t numTokensChunk, const bool page2L, 
                                AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->numLayers_ = numLayers;
        this->hiddenDims_ = hiddenDims;
        this->pageBuffSize_ = pageBuffSize;
        this->numTokensChunk_ = numTokensChunk;
        this->page2L_ = page2L;
        this->valid_ = true;
        
        // we assume this is taken care of in the kernel launch.
        int64_t pagedTokenQueSize = numTokensChunk_ * this->hiddenDims_ * sizeof(scalar_t);
        
        this->pipe_->InitBuffer(pagedTokenQue_, 2, pagedTokenQueSize);
    }

    __aicore__ inline void _page2LTransfer(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                           __gm__ uint8_t *slotmappings, const int cacheIdx, 
                                           const int layerIdx) {
        // get the slotmappings 
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        
        // 1. alloc per layer per cache buffer
        local_scalar_t perLayerSingleCacheBuffer = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        int64_t slot;
        int64_t pagedOffset = cacheIdx * this->pageBuffSize_ * this->hiddenDims_;
        int64_t tmpPagedOffset;
        int64_t localTensorTokenOffset;
        // 2. copy num tokens
        for (int64_t tokenIdx = 0; tokenIdx < this->numTokensChunk_; tokenIdx++) {
            slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            tmpPagedOffset = pagedOffset + slot * this->hiddenDims_;
            localTensorTokenOffset = tokenIdx * this->hiddenDims_;
            AscendC::DataCopy(perLayerSingleCacheBuffer[localTensorTokenOffset], this->pagedTokenGlobal_[tmpPagedOffset], 
                              this->hiddenDims_);
        }

        // 3. enque & deque
        pagedTokenQue_.EnQue(perLayerSingleCacheBuffer);
        perLayerSingleCacheBuffer = pagedTokenQue_.DeQue<scalar_t>();
        
        // 4. copy singleCache buffer to the right global idx
        int64_t cacheTensorLayerOffset = cacheIdx * this->numLayers_ * this->numTokensChunk_ * this->hiddenDims_ + 
                                         layerIdx * this->numTokensChunk_ * this->hiddenDims_;
        AscendC::DataCopy(this->lmcBufferGlobal_[cacheTensorLayerOffset], perLayerSingleCacheBuffer, 
                          this->numTokensChunk_ * this->hiddenDims_);

        // 5. Free
        pagedTokenQue_.FreeTensor(perLayerSingleCacheBuffer);
    }

    __aicore__ inline void _L2PageTransfer(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                           __gm__ uint8_t *slotmappings, const int cacheIdx, 
                                           const int layerIdx) {
        // get the slotmappings 
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        
        // 1. alloc per layer per cache buffer
        local_scalar_t perLayerSingleCacheBuffer = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        
        // 2. copy the L buffer to local
        int64_t cacheTensorLayerOffset = cacheIdx * this->numLayers_ * this->numTokensChunk_ * this->hiddenDims_ +
                                         layerIdx * this->numTokensChunk_ * this->hiddenDims_;
        
        AscendC::DataCopy(perLayerSingleCacheBuffer, this->lmcBufferGlobal_[cacheTensorLayerOffset], 
                          this->numTokensChunk_ * this->hiddenDims_);
        
        // 3. enque & deque
        pagedTokenQue_.EnQue(perLayerSingleCacheBuffer);
        perLayerSingleCacheBuffer = pagedTokenQue_.DeQue<scalar_t>();

        // 4. now this is in ub
        int64_t slot;
        int64_t pagedOffset = cacheIdx * this->pageBuffSize_ * this->hiddenDims_;
        int64_t tmpPagedOffset;
        int64_t localTensorTokenOffset;
        // copy per token into paged
        for (int64_t tokenIdx = 0; tokenIdx < this->numTokensChunk_; tokenIdx++) {
            slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            tmpPagedOffset = pagedOffset + slot * this->hiddenDims_;
            localTensorTokenOffset = tokenIdx * this->hiddenDims_;
            AscendC::DataCopy(this->pagedTokenGlobal_[tmpPagedOffset], perLayerSingleCacheBuffer[localTensorTokenOffset], 
                              this->hiddenDims_);
        }

        // 5. free
        pagedTokenQue_.FreeTensor(perLayerSingleCacheBuffer);
    }

    __aicore__ inline void processLayerCache(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                             __gm__ uint8_t *slotmappings, const int cacheIdx, 
                                             const int layerIdx, const bool page2L) 
    {
        // set global buffer for pagedKVCaches at layer boundary
        // its a pointer within the GM addr space, that point to another GM addr space
        __gm__ uint8_t * __gm__ *pagedKVCachesPtr = reinterpret_cast<__gm__ uint8_t* __gm__ *>(pagedKVCaches);

        // getting the right ptr to the paged kvcache layer
        __gm__ uint8_t *pagedLayerKVCaches = pagedKVCachesPtr[layerIdx];
        
        // For both page2L and L2Page, we copy per token via and to the pagedcache.
        this->pagedTokenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedLayerKVCaches),
                                                this->hiddenDims_);
        
        // For the cache tensor, since per layer is contiguous, we do contiguous copy.
        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(cacheTensor),
                                               this->numTokensChunk_*this->hiddenDims_);

        if (page2L) {
            this->_page2LTransfer(pagedKVCaches, cacheTensor, slotmappings, cacheIdx, layerIdx);
        } else {
            this->_L2PageTransfer(pagedKVCaches, cacheTensor, slotmappings, cacheIdx, layerIdx);
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> pagedTokenQue_;

    // [layers * [kvs, numPages * pagedSize, heads*headsize]]
    AscendC::GlobalTensor<scalar_t> pagedTokenGlobal_;
    // [kvs, layers, numTokensChunk, heads*headsize]
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;
    int32_t numLayers_; // num layers
    int64_t pageBuffSize_; // pages * pageSize
    int64_t hiddenDims_; // heads * headSize
    int32_t numTokensChunk_; // num tokens in the cache tensor chunk
    bool valid_;
    bool page2L_; // true, from pagedTensor to LMC, false otherwise
};

#define MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_DECLARE(TYPE, SLOTTYPE)                                                      \
        extern "C" __global__ __aicore__ void multi_layer_paged_kv_copy_v2_##TYPE##_##SLOTTYPE(                        \
        __gm__ uint8_t* pagedKVCaches, __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* slotmappings,                   \
        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,                                          \
        const int64_t pageBuffSize, const int32_t numTokensChunk, const int coreNum, const bool page2L)                \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        MultiLayerPagedKVCopyV2<TYPE, SLOTTYPE> op{};                                                                  \
        int32_t bIdx = AscendC::GetBlockIdx();                                                                         \
        int32_t launchedCores = AscendC::GetBlockNum();                                                                \
        int32_t layersPerCore = (numLayers + launchedCores - 1) / launchedCores;                                       \
        int32_t startLayersIdx = bIdx * layersPerCore;                                                                 \
        int32_t endLayersIdx = min(numLayers, startLayersIdx + layersPerCore);                                         \
        op.init(pagedKVCaches, dstCacheTensor, slotmappings, hiddenDims,                                               \
                numLayers, pageBuffSize, numTokensChunk, page2L, &pipe);                                               \
        for (int32_t layerIdx = startLayersIdx; layerIdx < endLayersIdx; layerIdx++) {                                 \
            for (int32_t cacheIdx = 0; cacheIdx < kvs; cacheIdx++) {                                                   \
                op.processLayerCache(pagedKVCaches, dstCacheTensor, slotmappings, cacheIdx, layerIdx, page2L);         \
            }                                                                                                          \
        }                                                                                                              \
    }

#define MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE(TYPE)   \
    MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_DECLARE(TYPE, int32_t);      \
    MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_DECLARE(TYPE, int64_t);

// Declare support kernel entry in the device side
MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE(half);
MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE(int8_t);
#if (__CCE_AICORE__ >= 220)
// At the device side, the macro will be expanded.
MULTI_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE(bfloat16_t);
#endif

namespace kvcache_ops {

#define MULTI_LAYER_PAGED_KV_COPY_V2_KERNEL_CALL(TYPE, SLOTTYPE)                                                          \
    multi_layer_paged_kv_copy_v2_##TYPE##_##SLOTTYPE<<<blockDim, nullptr, stream>>>(pagedKVCaches, dstCacheTensor,        \
                                                                                 slotmappings, hiddenDims, kvs,           \
                                                                                 numLayers, pageBuffSize,                 \
                                                                                 numTokensChunk, blockDim, page2L);

template<typename T, typename SlotT>
void multi_layer_paged_kernel_v2(uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, 
                  uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, 
                  const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L);

#define MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_DECLARE(TYPE, SLOTTYPE)                                                     \
template<>                                                                                                                \
void multi_layer_paged_kernel_v2<TYPE, SLOTTYPE>(uint32_t blockDim, void *stream, uint8_t *pagedKVCaches,                 \
                                              uint8_t *dstCacheTensor, uint8_t *slotmappings,                             \
                                              const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,       \
                                              const int64_t pageBuffSize, const int32_t numTokensChunk,                   \
                                              const bool page2L){                                                         \
    MULTI_LAYER_PAGED_KV_COPY_V2_KERNEL_CALL(TYPE, SLOTTYPE);                                                             \
}

#define MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_SLOTTYPE_DECLARE(TYPE)  \
    MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_DECLARE(TYPE, int32_t);     \
    MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_DECLARE(TYPE, int64_t);

MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_SLOTTYPE_DECLARE(half);
MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_SLOTTYPE_DECLARE(int8_t);
// this compile definition is for the host side.
#if (ASCEND_AICORE_ARCH >= 220)
MULTI_LAYER_PAGED_KERNEL_CALL_V2_TYPE_SLOTTYPE_DECLARE(bfloat16_t);
#endif

template<typename T>
void dispatch_paged_kernel_on_slot_type_v2(kvcache_ops::AscendType slotType, uint32_t blockDim, void *stream,
                                        uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings, 
                                        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,         
                                        const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L) {
    switch(slotType) {
        case kvcache_ops::AscendType::INT32:
            multi_layer_paged_kernel_v2<T, int32_t>(blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings,
                                     hiddenDims, kvs, numLayers, pageBuffSize, numTokensChunk, page2L);
            break;
        case kvcache_ops::AscendType::INT64:
            multi_layer_paged_kernel_v2<T, int64_t>(blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings,
                                     hiddenDims, kvs, numLayers, pageBuffSize, numTokensChunk, page2L);
            break;
        default:
            return;
    }
}

extern void multi_layer_kv_transfer_kernel_v2(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, 
                                           uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, 
                                           uint8_t *dstCacheTensor, uint8_t *slotmappings, 
                                           const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, 
                                           const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L)
{
    switch(type) {
        case kvcache_ops::AscendType::FP16:
            dispatch_paged_kernel_on_slot_type_v2<half>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor, 
                                                     slotmappings, hiddenDims, kvs, numLayers, pageBuffSize, 
                                                     numTokensChunk, page2L);
            break;
#if (ASCEND_AICORE_ARCH >= 220)
        case kvcache_ops::AscendType::BF16:
            dispatch_paged_kernel_on_slot_type_v2<bfloat16_t>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor, 
                                                           slotmappings, hiddenDims, kvs, numLayers, pageBuffSize, 
                                                           numTokensChunk, page2L);
            break;
#endif
        case kvcache_ops::AscendType::INT8:
            dispatch_paged_kernel_on_slot_type_v2<int8_t>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor, 
                                                        slotmappings, hiddenDims, kvs, numLayers, pageBuffSize, 
                                                        numTokensChunk, page2L);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(type)) + " is not supported.")
            throw std::runtime_error("Scalar type: " + std::to_string(static_cast<int>(type)) + " not supported. This should not have happened.");
    }
}

} // namespace kvcache_ops