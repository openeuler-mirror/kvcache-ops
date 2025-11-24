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

template <typename scalar_t, typename slot_t> class MultiLayerPagedKVCopy310P {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline MultiLayerPagedKVCopy310P()
    {
    }

    __aicore__ inline void init(GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, GM_ADDR slotmappings,
                                const int64_t hiddenDims, const int32_t numLayers,
                                const int64_t pageBuffSize, const int32_t numTokensChunk,
                                const bool page2L, const int32_t numKVHead,
                                const int32_t headSize, const int32_t blockSize, AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->numLayers_ = numLayers;
        this->hiddenDims_ = hiddenDims;
        this->pageBuffSize_ = pageBuffSize;
        this->numTokensChunk_ = numTokensChunk;
        this->page2L_ = page2L;
        this->numKVHead_ = numKVHead;
        this->block_size_ = blockSize;
        this->headSize_ = headSize;
        this->chunk_size_ = 16;
        this->chunks_per_head_ = headSize / chunk_size_;
        this->total_chunks_ = numKVHead * chunks_per_head_;
        this->num_blocks_ = pageBuffSize / blockSize;
        this->valid_ = true;

        this->pipe_->InitBuffer(pagedTokenQue_, 4, this->chunk_size_*sizeof(scalar_t));
    }

    __aicore__ inline void reset(){
        this->valid_ = true;
    }

    __aicore__ inline void updateMemOffset(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor,
                                             __gm__ uint8_t *slotmappings, const int tokenIdx,
                                             const int kvIdx, const int layerIdx,
                                             const int headIdx, const int chunkIdx)
    {
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);

        if (slot == -1) {
            this->valid_ = false;
            return;
        }

        int64_t block_id = slot / this->block_size_;
        int64_t token_in_block = slot % this->block_size_;

        int64_t global_chunk_idx = headIdx * this->chunks_per_head_ + chunkIdx;

        // its a pointer within the GM addr space, that point to another GM addr space
        __gm__ uint8_t * __gm__ *pagedKVCachesPtr = reinterpret_cast<__gm__ uint8_t* __gm__ *>(pagedKVCaches);

        // getting the right ptr to the paged kvcache layer
        __gm__ uint8_t *pagedLayerKVCaches = pagedKVCachesPtr[layerIdx];

        int64_t pagedIdxOffset =
            kvIdx * (this->num_blocks_ * this->total_chunks_ * this->block_size_ * this->chunk_size_) +
            block_id * (this->total_chunks_ * this->block_size_ * this->chunk_size_) +
            global_chunk_idx * (this->block_size_ * this->chunk_size_) +
            token_in_block * this->chunk_size_;

        int64_t dstTensorIdxOffset =
            kvIdx * (this->numLayers_ * this->numTokensChunk_ * this->hiddenDims_) +
            layerIdx * (this->numTokensChunk_ * this->hiddenDims_) +
            tokenIdx * this->hiddenDims_ +
            headIdx * this->headSize_ +
            chunkIdx * this->chunk_size_;

        this->pagedTokenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedLayerKVCaches) + pagedIdxOffset,
                                                   this->chunk_size_);
        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(cacheTensor) + dstTensorIdxOffset,
                                               this->chunk_size_);
    }

    __aicore__ inline void processFunc() {
        if (!this->valid_) {
            return;
        }

        // 1. Alloc Tensor for local page
        local_scalar_t chunkTensor = pagedTokenQue_.AllocTensor<scalar_t>();

        // 2. copy from global tensor into local
        if (this->page2L_) {
            AscendC::DataCopy(chunkTensor, this->pagedTokenGlobal_, this->chunk_size_);
        } else {
            AscendC::DataCopy(chunkTensor, this->lmcBufferGlobal_, this->chunk_size_);
        }

        // 3. enque vecin
        pagedTokenQue_.EnQue(chunkTensor);
        // 4. deque vecin, possible to reuse due to QueBind
        chunkTensor = pagedTokenQue_.DeQue<scalar_t>();

        // 5. datacopy into GM
        if (this->page2L_) {
            AscendC::DataCopy(this->lmcBufferGlobal_, chunkTensor, this->chunk_size_);
        } else {
            AscendC::DataCopy(this->pagedTokenGlobal_, chunkTensor, this->chunk_size_);
        }

        // 6. free alloced Tensor
        pagedTokenQue_.FreeTensor(chunkTensor);
    }


private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4> pagedTokenQue_;

    // [layers * [kvs, numPages * pagedSize, heads*headsize]]
    AscendC::GlobalTensor<scalar_t> pagedTokenGlobal_;
    // [kvs, layers, numTokensChunk, heads*headsize]
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;
    int32_t numLayers_; // num layers
    int64_t pageBuffSize_; // pages * pageSize
    int64_t hiddenDims_; // heads * headSize
    int32_t numTokensChunk_; // num tokens in the cache tensor chunk
    int32_t numKVHead_;
    int32_t headSize_;
    int32_t block_size_;
    int32_t chunk_size_;
    int32_t chunks_per_head_;
    int32_t total_chunks_;
    int32_t num_blocks_;
    bool valid_;
    bool page2L_; // true, from pagedTensor to LMC, false otherwise
};

// NOTE: there are potential micro optimizaiton here.
#define MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(TYPE, SLOTTYPE)                                                    \
        extern "C" __global__ __aicore__ void multi_layer_paged_kv_copy_310p_##TYPE##_##SLOTTYPE(                      \
        __gm__ uint8_t* pagedKVCaches, __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* slotmappings,                   \
        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,                                          \
        const int64_t pageBuffSize, const int32_t numTokensChunk, const int coreNum, const bool page2L,                \
        const int32_t numKVHead, const int32_t headSize, const int32_t blockSize)                                      \
    {                                                                                                                  \
        const int chunk_size = 16;                                                                                     \
        const int chunks_per_head = headSize / chunk_size;                                                             \
        const int total_chunks = numKVHead * chunks_per_head;                                                          \
        AscendC::TPipe pipe;                                                                                           \
        MultiLayerPagedKVCopy310P<TYPE, SLOTTYPE> op{};                                                                \
        op.init(pagedKVCaches, dstCacheTensor, slotmappings, hiddenDims,                                               \
                numLayers, pageBuffSize, numTokensChunk, page2L,                                                       \
                numKVHead, headSize, blockSize, &pipe);                                                                \
        int64_t bIdx = AscendC::GetBlockIdx();                                                                         \
        for (int64_t i = bIdx; i < numTokensChunk; i+=coreNum) {                                                       \
            for (int32_t kvIdx = 0; kvIdx < kvs; kvIdx ++) {                                                           \
                for (int32_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {                                         \
                    for (int32_t headIdx = 0; headIdx < numKVHead; headIdx++) {                                        \
                        for(int32_t chunkIdx = 0;chunkIdx < chunks_per_head; chunkIdx++) {                             \
                            op.reset();                                                                                \
                            op.updateMemOffset(pagedKVCaches, dstCacheTensor, slotmappings,                            \
                                                i, kvIdx, layerIdx, headIdx, chunkIdx);                                \
                            op.processFunc();                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

// Declare support kernel entry
MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(half, int32_t);
MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(half, int64_t);
#if (__CCE_AICORE__ >= 220)
MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(bfloat16_t, int32_t);
MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(bfloat16_t, int64_t);
#endif
MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(int8_t, int32_t);
MULTI_LAYER_PAGED_KV_COPY_310P_TYPE_DECLARE(int8_t, int64_t);

namespace kvcache_ops {

#define MULTI_LAYER_PAGED_KV_COPY_310P_KERNEL_CALL(TYPE, SLOTTYPE)                                                     \
    multi_layer_paged_kv_copy_310p_##TYPE##_##SLOTTYPE<<<blockDim, nullptr, stream>>>(pagedKVCaches, dstCacheTensor,   \
                                                                                 slotmappings, hiddenDims, kvs,        \
                                                                                 numLayers, pageBuffSize,              \
                                                                                 numTokensChunk, blockDim, page2L,     \
                                                                                 numKVHead, headSize, blockSize);

template<typename T, typename SlotT>
void multi_layer_paged_kernel_310p(uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor,
                  uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,
                  const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L,
                  const int32_t numKVHead, const int32_t headSize, const int32_t blockSize);

#define MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(TYPE, SLOTTYPE)                                                \
template<>                                                                                                             \
void multi_layer_paged_kernel_310p<TYPE, SLOTTYPE>(uint32_t blockDim, void *stream, uint8_t *pagedKVCaches,            \
                                              uint8_t *dstCacheTensor, uint8_t *slotmappings,                          \
                                              const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,    \
                                              const int64_t pageBuffSize, const int32_t numTokensChunk,                \
                                              const bool page2L, const int32_t numKVHead,                              \
                                              const int32_t headSize, const int32_t blockSize){                        \
    MULTI_LAYER_PAGED_KV_COPY_310P_KERNEL_CALL(TYPE, SLOTTYPE);                                                        \
}

MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(half, int32_t);
MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(half, int64_t);
#if (__CCE_AICORE__ >= 220)
MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int32_t);
MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(bfloat16_t, int64_t);
#endif
MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(int8_t, int32_t);
MULTI_LAYER_PAGED_310P_KERNEL_CALL_TYPE_DECLARE(int8_t, int64_t);

template<typename T>
void dispatch_paged_kernel_310p_on_slot_type(kvcache_ops::AscendType slotType, uint32_t blockDim, void *stream,
                                        uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,
                                        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,
                                        const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L,
                                        const int32_t numKVHead, const int32_t headSize, const int32_t blockSize) {
    switch(slotType) {
        case kvcache_ops::AscendType::INT32:
            multi_layer_paged_kernel_310p<T, int32_t>(blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings,
                                     hiddenDims, kvs, numLayers, pageBuffSize, numTokensChunk, page2L,
                                     numKVHead, headSize, blockSize);
            break;
        case kvcache_ops::AscendType::INT64:
            multi_layer_paged_kernel_310p<T, int64_t>(blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings,
                                     hiddenDims, kvs, numLayers, pageBuffSize, numTokensChunk, page2L,
                                     numKVHead, headSize, blockSize);
            break;
        default:
            return;
    }
}

extern void multi_layer_kv_transfer_kernel_310p(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
                                           uint32_t blockDim, void *stream, uint8_t *pagedKVCaches,
                                           uint8_t *dstCacheTensor, uint8_t *slotmappings,
                                           const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,
                                           const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L,
                                           const int32_t numKVHead, const int32_t headSize, const int32_t blockSize)
{
    switch(type) {
        case kvcache_ops::AscendType::FP16:
            dispatch_paged_kernel_310p_on_slot_type<half>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor,
                                                     slotmappings, hiddenDims, kvs, numLayers, pageBuffSize,
                                                     numTokensChunk, page2L, numKVHead, headSize, blockSize);
            break;
#if (__CCE_AICORE__ >= 220)
        case kvcache_ops::AscendType::BF16:
            dispatch_paged_kernel_310p_on_slot_type<bfloat16_t>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor,
                                                           slotmappings, hiddenDims, kvs, numLayers, pageBuffSize,
                                                           numTokensChunk, page2L, numKVHead, headSize, blockSize);
            break;
#endif
        case kvcache_ops::AscendType::INT8:
            dispatch_paged_kernel_310p_on_slot_type<int8_t>(slotType, blockDim, stream, pagedKVCaches, dstCacheTensor,
                                                        slotmappings, hiddenDims, kvs, numLayers, pageBuffSize,
                                                        numTokensChunk, page2L, numKVHead, headSize, blockSize);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(type)) + " is not supported.")
            throw std::runtime_error("Scalar type: " + std::to_string(static_cast<int>(type)) + " not supported. This should not have happened.");
    }
}

} // namespace kvcache_ops