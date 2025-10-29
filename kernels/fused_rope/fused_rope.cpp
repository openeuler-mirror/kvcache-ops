#include "kernel_operator.h"
#include "fused_rope_bf16.h"
#include "fused_rope_fp32.h"

using namespace AscendC;
using namespace FusedRope;

extern "C" __global__ __aicore__ void FusedRopeKernel(
    GM_ADDR oldPositions, GM_ADDR newPositions, GM_ADDR keyIn,
    GM_ADDR cosSinCache, GM_ADDR keyOut, uint64_t coreNumUse, uint64_t numTokens,
    uint64_t numHeads, uint64_t headSize, uint64_t rotaryDim,
    uint64_t kLeadingDimension, uint64_t isNeoxStyle,
    uint64_t frontCore, uint64_t tailCore,
    uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
    uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
    uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
    uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop, uint64_t tilingKey)
{
    TPipe pipe;
    // DT_BF16
    if (tilingKey == 20) {
        TPipe* ptr = &pipe;
        if (ptr != nullptr) {
            FusedRopeFP16<bfloat16_t> op;
            op.Init(oldPositions, newPositions, keyIn, cosSinCache, keyOut,
                    coreNumUse, numTokens,
                    numHeads, headSize, rotaryDim,
                    kLeadingDimension, isNeoxStyle,
                    frontCore, tailCore,
                    numTokensFrontCoreEachLoop, numTokensTailCoreEachLoop,
                    numTokensEachFrontCore, numTokensEachTailCore,
                    loopTimeEachFrontCore, loopTimeEachTailCore,
                    numTokensFrontCoreLastLoop, numTokensTailCoreLastLoop, ptr);
            op.Process();
        }
    }
    // DT_FLOAT16
    if (tilingKey == 21) {
        TPipe* ptr = &pipe;
        if (ptr != nullptr) {
            FusedRopeFP16<half> op;
            op.Init(oldPositions, newPositions, keyIn, cosSinCache, keyOut,
                    coreNumUse, numTokens,
                    numHeads, headSize, rotaryDim,
                    kLeadingDimension, isNeoxStyle,
                    frontCore, tailCore,
                    numTokensFrontCoreEachLoop, numTokensTailCoreEachLoop,
                    numTokensEachFrontCore, numTokensEachTailCore,
                    loopTimeEachFrontCore, loopTimeEachTailCore,
                    numTokensFrontCoreLastLoop, numTokensTailCoreLastLoop, ptr);
            op.Process();
        }
    }
    // DT_FLOAT
    if (tilingKey == 22) {
        TPipe* ptr = &pipe;
        if (ptr != nullptr) {
            FusedRopeFP32<float> op;
            op.Init(oldPositions, newPositions, keyIn, cosSinCache, keyOut, 
                    coreNumUse, numTokens,
                    numHeads, headSize, rotaryDim,
                    kLeadingDimension, isNeoxStyle,
                    frontCore, tailCore,
                    numTokensFrontCoreEachLoop, numTokensTailCoreEachLoop,
                    numTokensEachFrontCore, numTokensEachTailCore,
                    loopTimeEachFrontCore, loopTimeEachTailCore,
                    numTokensFrontCoreLastLoop, numTokensTailCoreLastLoop, ptr);
            op.Process();
        }
    }
}


namespace kvcache_ops {
    extern void rotary_embedding_kernel_dispatch(
        uint64_t blockDim, void* stream, uint8_t* oldPositions,
        uint8_t* newPositions, uint8_t* key, uint8_t* cosSinCache,
        uint8_t* keyOut, uint64_t numTokens, uint64_t numHeads,
        uint64_t headSize, uint64_t rotaryDim, uint64_t kLeadingDimension,
        uint64_t isNeoxStyle, uint64_t frontCore, uint64_t tailCore,
        uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
        uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
        uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
        uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop,
        uint64_t tilingKey)
    {
        FusedRopeKernel<<<blockDim, nullptr, stream>>>(
            oldPositions, newPositions, key,
            cosSinCache, keyOut, blockDim, numTokens,
            numHeads, headSize, rotaryDim,
            kLeadingDimension, isNeoxStyle, frontCore,
            tailCore, numTokensFrontCoreEachLoop, numTokensTailCoreEachLoop,
            numTokensEachFrontCore, numTokensEachTailCore, loopTimeEachFrontCore,
            loopTimeEachTailCore, numTokensFrontCoreLastLoop, numTokensTailCoreLastLoop,
            tilingKey
        );
    }
}