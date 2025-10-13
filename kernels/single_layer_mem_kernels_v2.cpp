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
#include <iostream>

constexpr int32_t ASCEND_BLOCK_LEN = 32;

template <typename scalar_t, typename slot_t> class SingleLayerPagedKVCopyV2 {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline SingleLayerPagedKVCopyV2()
    {
    }

    __aicore__ inline void init(GM_ADDR lmcKeyValueCachePtr, GM_ADDR vllmKeyValuePtr, GM_ADDR slotMappingPtr, 
                                const int64_t vllmBlockStride, const int64_t vllmValueOffset, const int64_t vllmBufferSize,
                                const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize,
                                const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, 
                                const int32_t numTokens, const int32_t blockSize, const bool page2L, const bool lmcTokensMajor, 
                                AscendC::TPipe *pipe)

    {
        this->pipe_ = pipe;
        this->numHeads_ = numHeads;
        this->numTokens_ = numTokens;
        this->blockSize_ = blockSize;
        this->page2L_ = page2L;
        this->headDims_ = headDims;
        this->vllmBlockStride_ = vllmBlockStride;
        this->lmcTokenStride_ = lmcTokenStride;
        this->lmcTokensMajor_ = lmcTokensMajor;

        // if we are in MLA land, we won't use this for our copy
        this->vllmValueOffset_ = vllmValueOffset;
        this->lmcValueOffset_ = lmcValueOffset;
        this->numKvs_ = 2;

        uint64_t localTokenBufferSize = maxTokensPerLoop * this->numKvs_ * this->numHeads_ * this->headDims_ * sizeof(scalar_t);
        
        this->vllmKVGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(vllmKeyValuePtr), vllmBufferSize);
        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(lmcKeyValueCachePtr), lmcBufferSize);
        this->pipe_->InitBuffer(this->tokenQue_, 2, localTokenBufferSize);
    }

    __aicore__ inline void page2LCopy(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        // alloc local buffer per tokens
        local_scalar_t tokensBufferTensor = this->tokenQue_.template AllocTensor<scalar_t>();
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);

        int64_t blockIdx;
        int64_t blockOffset;
        int64_t vllmKeyIdx;
        int64_t localTokenBuffKIdx;
        int64_t slot;
        int64_t vllmValueIdx;
        int64_t localTokenBuffVIdx;
        int64_t realTokenIdx;
        // per slot mapping we copy the tokens
        // we are moving number of actualTokensPerLoop tokens from paged memory to local UB memory 
        // if we have standard KV tensor, we stride for the value offset,
        // otherwise for MLA, we skip the value offset and the localbuffer would already have it taken into account.
        for (int32_t innerTokenIdx = 0; innerTokenIdx < actualTokensPerLoop; innerTokenIdx ++) {
            realTokenIdx = tokenIdx + innerTokenIdx;
            slot = static_cast<int64_t>(slotmappingPtr[realTokenIdx]);
            // work out where is it in the vllm page buff
            blockIdx = slot / this->blockSize_;
            blockOffset = slot % this->blockSize_;

            vllmKeyIdx = blockIdx * this->vllmBlockStride_ + blockOffset * this->numHeads_ * this->headDims_;

            localTokenBuffKIdx = innerTokenIdx * this->numKvs_ * this->numHeads_ * this->headDims_;

            // copy into the local token buffer
            AscendC::DataCopy(tokensBufferTensor[localTokenBuffKIdx], this->vllmKVGlobal_[vllmKeyIdx], this->numHeads_ * this->headDims_);
            vllmValueIdx = vllmKeyIdx + this->vllmValueOffset_;
            // stride for heads*headDims
            localTokenBuffVIdx = localTokenBuffKIdx + this->numHeads_ * this->headDims_;
            AscendC::DataCopy(tokensBufferTensor[localTokenBuffVIdx], this->vllmKVGlobal_[vllmValueIdx], this->numHeads_ * this->headDims_);
        }

        // now we have the local buffer filled,
        this->tokenQue_.EnQue(tokensBufferTensor);
        tokensBufferTensor = this->tokenQue_.template DeQue<scalar_t>();

        int64_t perCacheBlockLen = (this->numHeads_ * this->headDims_ * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        // copy this into the global memory
        int64_t lmcTokenKOffset = tokenIdx * this->lmcTokenStride_;
        AscendC::DataCopyParams tokenCopyParams;
        tokenCopyParams.blockCount = actualTokensPerLoop;
        // because we are storing the localToken in tokensMajor
        // if its tokensMajor, we can directly copy into the K index
        if (this->lmcTokensMajor_ || this->numKvs_ == 1) {
            // k and v | and this would work for MLA anyway
            tokenCopyParams.blockLen = perCacheBlockLen * this->numKvs_;
            tokenCopyParams.srcStride = 0;
            tokenCopyParams.dstStride = 0;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenKOffset], tokensBufferTensor, tokenCopyParams);
        } else {
            // copying from tokensMajor local buffer to two major
            // [2, tokens, ... ]
            // because we are at tokens major [tokens, 2, ...]
            // so we need to stride per cache block
            tokenCopyParams.blockLen = perCacheBlockLen;
            tokenCopyParams.srcStride = perCacheBlockLen;
            // because all keys/values are next to each other
            tokenCopyParams.dstStride = 0;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenKOffset], tokensBufferTensor, tokenCopyParams);
            // for the v copy
            int64_t lmcTokenVOffset = lmcTokenKOffset + this->lmcValueOffset_;
            int64_t localVOffset = this->numHeads_ * this->headDims_;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenVOffset], tokensBufferTensor[localVOffset], tokenCopyParams);
        }
        
        this->tokenQue_.FreeTensor(tokensBufferTensor);
    }

    __aicore__ inline void l2PageCopy(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        // alloc local buffer per tokens
        local_scalar_t tokensBufferTensor = this->tokenQue_.template AllocTensor<scalar_t>();
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);

        // copy from lmcbuffer to local tokens
        // we always do tokens major in the ub buffer
        AscendC::DataCopyParams tokensCopyParams;
        int64_t perCacheBlockLen = (this->numHeads_ * this->headDims_ * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        tokensCopyParams.blockCount = actualTokensPerLoop;
        int64_t lmcTokenIdx = tokenIdx * this->lmcTokenStride_;
        if (this->lmcTokensMajor_ || this->numKvs_ == 1) {
            tokensCopyParams.blockLen = perCacheBlockLen * this->numKvs_;
            tokensCopyParams.srcStride = 0;
            tokensCopyParams.dstStride = 0;
            AscendC::DataCopy(tokensBufferTensor, this->lmcBufferGlobal_[lmcTokenIdx], tokensCopyParams);
        } else {
            // copying from twoMajor to tokensMajor
            // [2, tokens, ...]
            tokensCopyParams.blockLen = perCacheBlockLen;
            tokensCopyParams.srcStride = 0;
            tokensCopyParams.dstStride = perCacheBlockLen;
            // for the K
            AscendC::DataCopy(tokensBufferTensor, this->lmcBufferGlobal_[lmcTokenIdx], tokensCopyParams);

            // for the V
            int64_t localVOffset = this->numHeads_ * this->headDims_;
            int64_t lmcTokenVIdx = lmcTokenIdx + this->lmcValueOffset_;
            AscendC::DataCopy(tokensBufferTensor[localVOffset], this->lmcBufferGlobal_[lmcTokenVIdx], tokensCopyParams);
        }

        // now we have the local buffer filled,
        this->tokenQue_.EnQue(tokensBufferTensor);
        tokensBufferTensor = this->tokenQue_.template DeQue<scalar_t>();


        int64_t slot;
        int64_t blockIdx;
        int64_t blockOffset;
        int64_t vllmKeyIdx;
        int64_t localTokenBuffKIdx;        
        int64_t vllmValueIdx;
        int64_t localTokenBuffVIdx;
        int64_t realTokenIdx;

        for (int32_t innerTokenIdx = 0; innerTokenIdx < actualTokensPerLoop; innerTokenIdx ++) {
            realTokenIdx = tokenIdx + innerTokenIdx;
            slot = static_cast<int64_t>(slotmappingPtr[realTokenIdx]);

            // work out where is it in the vllm page buff
            blockIdx = slot / this->blockSize_;
            blockOffset = slot % this->blockSize_;

            vllmKeyIdx = blockIdx * this->vllmBlockStride_ + blockOffset * this->numHeads_ * this->headDims_;

            localTokenBuffKIdx = innerTokenIdx * this->numKvs_ * this->numHeads_ * this->headDims_;

            AscendC::DataCopy(this->vllmKVGlobal_[vllmKeyIdx], tokensBufferTensor[localTokenBuffKIdx], this->numHeads_ * this->headDims_);
            vllmValueIdx = vllmKeyIdx + this->vllmValueOffset_;
            // stride for heads*headDims
            localTokenBuffVIdx = localTokenBuffKIdx + this->numHeads_ * this->headDims_;
            AscendC::DataCopy(this->vllmKVGlobal_[vllmValueIdx], tokensBufferTensor[localTokenBuffVIdx], this->numHeads_ * this->headDims_);
        }

        this->tokenQue_.FreeTensor(tokensBufferTensor);
    }

    __aicore__ inline void process(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        if(this->page2L_) {
            this->page2LCopy(slotmappings, tokenIdx, actualTokensPerLoop);
        } else {
            this->l2PageCopy(slotmappings, tokenIdx, actualTokensPerLoop);
        }
    }

private:
    AscendC::TPipe *pipe_;
    // a depth of 2
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> tokenQue_;

    // [kvs, numPages * pagedSize, heads*headsize]
    AscendC::GlobalTensor<scalar_t> vllmKVGlobal_;

    // Depends on LMC setting whether we store in tokensMajor or not.
    // the layout would be the followings:
    // [tokens, kvs, heads*headsize] or [kvs, tokens, heads*headsize]
    // TODO: check whether should combine the two and use a loop
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;

    int64_t vllmBlockStride_;
    int64_t lmcTokenStride_;
    int64_t vllmValueOffset_;
    int64_t lmcValueOffset_;
    int32_t blockSize_; // the size of the paged attention tokens block
    int32_t headDims_;
    int32_t numHeads_;
    int32_t numTokens_; // num tokens in the cache tensor chunk
    int16_t numKvs_; // 1 if MLA else 2
    bool page2L_; // whether the direction of copy is from page to lmc
    bool lmcTokensMajor_; // whether the lmc buffer is in tokens major i.e. [tokens, kvs, ...]
};

// we splits tokens per core
// and loop over tokensPerCore with each loop having maxTokensPerLoop 
#define SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_DECLARE(TYPE, SLOTTYPE)                                              \
    extern "C" __global__ __aicore__ void single_layer_paged_kv_copy_v2_##TYPE##_##SLOTTYPE(                 \
        __gm__ uint8_t* lmcKeyValueCachePtr, __gm__ uint8_t* vllmKeyValuePtr, __gm__ uint8_t* slotMappingPtr,          \
        const int64_t vllmBlockStride, const int64_t vllmValueOffset, const int64_t vllmBufferSize,                    \
        const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize,                       \
        const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims,                                \
        const int32_t numTokens, const int32_t blockSize, const bool page2L, const bool lmcTokensMajor)                \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SingleLayerPagedKVCopyV2<TYPE, SLOTTYPE> op{};                                                          \
        int32_t coreIdx = AscendC::GetBlockIdx();                                                                      \
        int32_t launchedCores = AscendC::GetBlockNum();                                                                \
        int32_t tokensPerCore = (numTokens + launchedCores - 1) / launchedCores;                                       \
        int32_t startTokenIdx = coreIdx * tokensPerCore;                                                               \
        int32_t endTokenIdx = min(numTokens, startTokenIdx + tokensPerCore);                                           \
        op.init(lmcKeyValueCachePtr, vllmKeyValuePtr, slotMappingPtr, vllmBlockStride, vllmValueOffset,                \
                vllmBufferSize, lmcTokenStride, lmcValueOffset, lmcBufferSize, maxTokensPerLoop, numHeads, headDims,   \
                numTokens, blockSize, page2L, lmcTokensMajor, &pipe);                                                  \
        for (int32_t tokenIdx = startTokenIdx; tokenIdx < endTokenIdx; tokenIdx += maxTokensPerLoop) {                 \
            int32_t actualTokensPerLoop = min(maxTokensPerLoop, endTokenIdx - tokenIdx);                               \
            op.process(slotMappingPtr, tokenIdx, actualTokensPerLoop);                                                 \
        }                                                                                                              \
    }


#define SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE_DEVICE(TYPE)    \
    SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_DECLARE(TYPE, int32_t);              \
    SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_DECLARE(TYPE, int64_t);

// Declare support kernel entry at the device side
SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE_DEVICE(half);
SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE_DEVICE(int8_t);
#if (__CCE_AICORE__ >= 220)
SINGLE_LAYER_PAGED_KV_COPY_V2_TYPE_SLOTTYPE_DECLARE_DEVICE(bfloat16_t);
#endif



// HostSide Declaration
namespace kvcache_ops {

#define SINGLE_LAYER_PAGED_KV_COPY_V2_KERNEL_CALL(TYPE, SLOTTYPE)                                              \
    single_layer_paged_kv_copy_v2_##TYPE##_##SLOTTYPE<<<blockDim, nullptr, stream>>>(lmcKeyValueCachePtr,     \
        vllmKeyValuePtr, slotMappingPtr, vllmBlockStride, vllmValueOffset, vllmBufferSize, lmcTokenStride,            \
        lmcValueOffset, lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize,                    \
        page2L, lmcTokensMajor);

template<typename T, typename SlotT>
void single_layer_paged_kernel_v2(uint32_t blockDim, void* stream,
    uint8_t *lmcKeyValueCachePtr, uint8_t *vllmKeyValuePtr, uint8_t *slotMappingPtr, 
    const int64_t vllmBlockStride, const int64_t vllmValueOffset, const int64_t vllmBufferSize, 
    const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize, 
    const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, const int32_t numTokens, 
    const int32_t blockSize, const bool page2L, const bool lmcTokensMajor);

#define SINGLE_LAYER_PAGED_KERNEL_V2_CALL_TYPE_DECLARE(TYPE, SLOTTYPE)                                        \
template<>                                                                                                           \
void single_layer_paged_kernel_v2<TYPE, SLOTTYPE>(uint32_t blockDim, void* stream,                            \
    uint8_t *lmcKeyValueCachePtr, uint8_t *vllmKeyValuePtr, uint8_t *slotMappingPtr,                                 \
    const int64_t vllmBlockStride, const int64_t vllmValueOffset, const int64_t vllmBufferSize,                      \
    const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize,                         \
    const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, const int32_t numTokens,         \
    const int32_t blockSize, const bool page2L, const bool lmcTokensMajor){                                          \
        SINGLE_LAYER_PAGED_KV_COPY_V2_KERNEL_CALL(TYPE, SLOTTYPE);                                            \
}

#define SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(TYPE)      \
    SINGLE_LAYER_PAGED_KERNEL_V2_CALL_TYPE_DECLARE(TYPE, int32_t);               \
    SINGLE_LAYER_PAGED_KERNEL_V2_CALL_TYPE_DECLARE(TYPE, int64_t);

// Declare the kernel entry at the host side
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(half);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(int8_t);
#if (ASCEND_AICORE_ARCH >= 220)
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(bfloat16_t);
#endif

template<typename T>
void dispatch_single_layer_kernel_v2_on_slot_type(kvcache_ops::AscendType slotType, uint32_t blockDim, void *stream, 
        uint8_t *lmcKeyValueCachePtr, uint8_t *vllmKeyValuePtr, uint8_t *slotMappingPtr, 
        const int64_t vllmBlockStride, const int64_t vllmValueOffset, const int64_t vllmBufferSize, 
        const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize, 
        const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, const int32_t numTokens, 
        const int32_t blockSize, const bool page2L, const bool lmcTokensMajor) {
    switch(slotType) {
        case kvcache_ops::AscendType::INT32:
            single_layer_paged_kernel_v2<T, int32_t>(blockDim, stream, lmcKeyValueCachePtr, vllmKeyValuePtr, slotMappingPtr, vllmBlockStride, vllmValueOffset, 
                            vllmBufferSize, lmcTokenStride, lmcValueOffset, lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor);
            break;
        case kvcache_ops::AscendType::INT64:
            single_layer_paged_kernel_v2<T, int64_t>(blockDim, stream, lmcKeyValueCachePtr, vllmKeyValuePtr, slotMappingPtr, vllmBlockStride, vllmValueOffset, 
                            vllmBufferSize, lmcTokenStride, lmcValueOffset, lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(slotType)) + " is not supported.")
            throw std::runtime_error("Slot type: " + std::to_string(static_cast<int>(slotType)) + " not supported. This should not have happened.");
    }                                                 
}

extern void single_layer_kv_transfer_kernel_v2(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, uint32_t blockDim, void *stream,
                                               uint8_t *lmcKeyValueCachePtr, uint8_t *vllmKeyValuePtr, uint8_t *slotMappingPtr, 
                                               const int64_t vllmBlockStride, const int64_t vllmValueOffset, const int64_t vllmBufferSize, 
                                               const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize, 
                                               const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, const int32_t numTokens, 
                                               const int32_t blockSize, const bool page2L, const bool lmcTokensMajor)
{
    
    switch(type) {
        case kvcache_ops::AscendType::FP16:
            dispatch_single_layer_kernel_v2_on_slot_type<half>(slotType, blockDim, stream, lmcKeyValueCachePtr, vllmKeyValuePtr, 
                                                               slotMappingPtr, vllmBlockStride, vllmValueOffset, vllmBufferSize, lmcTokenStride, lmcValueOffset, 
                                                               lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor);
            break;
#if (ASCEND_AICORE_ARCH >= 220)
        case kvcache_ops::AscendType::BF16:
            dispatch_single_layer_kernel_v2_on_slot_type<bfloat16_t>(slotType, blockDim, stream, lmcKeyValueCachePtr, vllmKeyValuePtr, 
                                                                     slotMappingPtr, vllmBlockStride, vllmValueOffset, vllmBufferSize, lmcTokenStride, lmcValueOffset, 
                                                                     lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor);
            break;
#endif
        case kvcache_ops::AscendType::INT8:
            dispatch_single_layer_kernel_v2_on_slot_type<int8_t>(slotType, blockDim, stream, lmcKeyValueCachePtr, vllmKeyValuePtr, 
                                                                 slotMappingPtr, vllmBlockStride, vllmValueOffset, vllmBufferSize, lmcTokenStride, lmcValueOffset, 
                                                                 lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor);
        default:
            return;
    }
}

} // namespace kvcache_ops