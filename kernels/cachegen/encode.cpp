#include "kernel_operator.h"
#include "cachegen_kernels.h"
#include <cstring>
#include <stdexcept>

namespace kvcache_ops {
namespace cachegen {
namespace impl {

__aicore__ inline auto ceil_32(int32_t size) -> uint32_t {
    return size % 32 == 0 ? size : 32 * (1 + (size / 32));
};

__aicore__ inline auto big_to_small(uint32_t value) -> uint32_t {
    return ((value & 0xFF000000U) >> 24) | ((value & 0x00FF0000U) >> 8) | ((value & 0x0000FF00U) << 8) | ((value & 0x000000FFU) << 24);
}

class Encoder {
public:
    __aicore__ inline Encoder(
        GM_ADDR cdf_data_ptr, // Input CDF [n_layers, n_channels, n_bins + 1], uint16
        GM_ADDR input_data_ptr, // Input symbols [n_layers, n_tokens, n_channels], uint8
        GM_ADDR output_data_ptr, // Output bytes [n_layers, n_channels, batch_size], uint8
        GM_ADDR output_lengths_data_ptr, // Output bytes [n_layers, n_channels], uint32
        AscendC::TPipe& pipe,
        int32_t n_tokens,
        int32_t n_layers,
        int32_t n_channels,
        uint32_t n_bins,
        int32_t chunk_size);

    __aicore__ inline void encode(int layer_id, int channel_id);

private:
    // CDF [n_layers, n_channels, n_bins + 1]
    AscendC::GlobalTensor<uint16_t> gm_cdf;
    // Input symbols [n_layers, n_tokens, n_channels]
    AscendC::GlobalTensor<uint8_t> gm_input;
    // Output bytes [n_layers, n_channels, batch_size]
    AscendC::GlobalTensor<uint8_t> gm_output;
    // Output byte lengths [n_layers, n_channels]
    AscendC::GlobalTensor<uint32_t> gm_output_lens;

    AscendC::TPipe& pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> symInQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> CDFInQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::LocalTensor<uint8_t> calc_sym_input; // [batch_size]

    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf;
    // scalar -> GM requires careful DCache handling while scalar -> MTE_3 -> GM is the paved path in AscendC. Have a 
    // tensor for the purpose of carrying the encoded length to output via this paved path.
    AscendC::LocalTensor<uint32_t> ub_encoded_output_len; // [1]

    // Input dimensions
    int32_t n_tokens;
    int32_t n_layers;
    int32_t n_channels;
    uint32_t n_bins;
    int32_t chunk_size;

    // Derived internal chunking values - the input dimensions dictate the how symbols can be copied in in chunks etc.
    // Stored here at class scope to avoid recomputation
    int32_t copy_volume;
    int32_t full_chunk_volume;
    int32_t tokens_per_chunk;
    int32_t max_full_chunk_id;
    int32_t tail_chunk_size;
    int32_t tokens_in_tail;
    bool has_tail_chunk;

    __aicore__ inline void append_bit_and_pending(
        uint32_t bit,
        uint64_t& pending_bits,
        uint32_t& output_reg,
        uint32_t& output_reg_write_head,
        AscendC::LocalTensor<uint8_t> encoded_output,
        uint32_t& encoded_output_write_head);

    __aicore__ inline void spill_reg_to_shared(
        uint32_t& output_reg,
        uint32_t& output_reg_write_head,
        AscendC::LocalTensor<uint8_t> encoded_output,
        uint32_t& encoded_output_write_head);

    __aicore__ inline void spill_partial_reg_to_shared(
        uint32_t& output_reg,
        uint32_t& output_reg_write_head,
        AscendC::LocalTensor<uint8_t> encoded_output,
        uint32_t& encoded_output_write_head);

    __aicore__ inline void copy_in(int layer_id, int channel_id);
    __aicore__ inline void copy_in_enq(int layer_offset, int chunk_id, uint32_t copy_volume);
    __aicore__ inline void copy_in_deq(uint32_t read_offset, uint32_t& write_offset, uint32_t n_to_gather);

    // Class has no known need to support move or copy operations
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) = delete;
    Encoder& operator=(Encoder&&) = delete;
};

__aicore__ inline Encoder::Encoder(
    GM_ADDR cdf_data_ptr,
    GM_ADDR input_data_ptr,
    GM_ADDR output_data_ptr,
    GM_ADDR output_lengths_data_ptr,
    AscendC::TPipe& _pipe,
    int32_t n_tokens, 
    int32_t n_layers, 
    int32_t n_channels,
    uint32_t n_bins,
    int32_t chunk_size):
        pipe(_pipe),
        n_tokens(n_tokens),
        n_layers(n_layers),
        n_channels(n_channels),
        n_bins(n_bins),
        chunk_size(chunk_size) {

    // Internal chunking to handle larger input. Distinct from outer chunking 
    copy_volume = n_channels * n_tokens;
    tokens_per_chunk = (1 << 15) / n_channels; // Implies a maximum supported channel size (of around 3200) 
    full_chunk_volume = tokens_per_chunk * n_channels;
    max_full_chunk_id = copy_volume / full_chunk_volume ;
    tail_chunk_size = copy_volume % full_chunk_volume;
    tokens_in_tail = tail_chunk_size / n_channels;
    has_tail_chunk = tail_chunk_size != 0;

    gm_cdf.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(cdf_data_ptr), n_layers * n_channels * (n_bins + 1));
    gm_input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input_data_ptr), n_layers * n_tokens * n_channels);
    gm_output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output_data_ptr), n_layers * n_channels * chunk_size);
    gm_output_lens.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output_lengths_data_ptr), n_layers * n_channels);

    pipe.InitBuffer(symInQ, 1, full_chunk_volume * sizeof(uint8_t));
    pipe.InitBuffer(CDFInQ, 1, (n_bins + 1) * sizeof(uint16_t));
    pipe.InitBuffer(outQ, 1, n_tokens * sizeof(uint8_t));

    uint32_t calc_buff_size_aligned = sizeof(uint8_t) * ceil_32(n_tokens);
    pipe.InitBuffer(calcBuf, calc_buff_size_aligned);
    int32_t calc_buf_offset = 0;

    // Symbols - [layer, :, channel], where the size is bound by n_tokens (chunked at a higher level so ~256 typically) 
    calc_sym_input = calcBuf.GetWithOffset<uint8_t>(ceil_32(n_tokens), calc_buf_offset);
    calc_buf_offset += ceil_32(sizeof(uint8_t) * ceil_32(n_tokens));

    uint32_t out_buff_size_aligned = ceil_32(sizeof(uint32_t) * 1); // ub_encoded_output_len
    pipe.InitBuffer(outBuf, out_buff_size_aligned);
    int32_t out_buf_offset = 0;
    ub_encoded_output_len = outBuf.GetWithOffset<uint32_t>(1, out_buf_offset);
    out_buf_offset += ceil_32(sizeof(uint32_t) * 1);
}

__aicore__ inline void Encoder::copy_in(int layer_id, int channel_id) {
    auto layer_offset = layer_id * n_channels * n_tokens;
    uint32_t write_offset = 0;
    uint32_t read_offset = channel_id;

    for (auto chunk_id = 0; chunk_id < max_full_chunk_id; ++chunk_id) {
        copy_in_enq(layer_offset, chunk_id, full_chunk_volume);
        copy_in_deq(read_offset, write_offset, tokens_per_chunk);
    }

    if (has_tail_chunk) {
        copy_in_enq(layer_offset, max_full_chunk_id, tail_chunk_size);
        copy_in_deq(read_offset, write_offset, tokens_in_tail);
    }

    AscendC::LocalTensor<uint16_t> ub_cdf_input = CDFInQ.AllocTensor<uint16_t>();
    auto channel_size = (n_bins + 1); // In unit of elements
    auto layer_size = channel_size * n_channels;
    auto offset = layer_size * layer_id + channel_size * channel_id;
    const AscendC::DataCopyExtParams cdf_copyParams = {1, (uint32_t)((n_bins + 1) * sizeof(uint16_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint16_t> cdf_padParams = {false, 0, 0, 0};
    AscendC::DataCopyPad(ub_cdf_input, gm_cdf[offset], cdf_copyParams, cdf_padParams);

    CDFInQ.EnQue(ub_cdf_input);
}

__aicore__ inline void Encoder::copy_in_enq(int layer_offset, int chunk_id, uint32_t copy_volume) {
    AscendC::LocalTensor<uint8_t> ub_sym_input = symInQ.AllocTensor<uint8_t>();
    const AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(copy_volume), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    AscendC::DataCopyPad(ub_sym_input, gm_input[layer_offset + chunk_id * full_chunk_volume], copyParams, padParams);
    symInQ.EnQue(ub_sym_input);
}

__aicore__ inline void Encoder::copy_in_deq(uint32_t read_offset, uint32_t& write_offset, uint32_t n_to_gather) {
    AscendC::LocalTensor<uint8_t> ub_sym_input = symInQ.DeQue<uint8_t>();
    for (auto ii = 0; ii < n_to_gather; ++ii) {
        calc_sym_input.SetValue(write_offset + ii, ub_sym_input(read_offset));
        read_offset += n_channels;
    }
    write_offset += n_to_gather;
    symInQ.FreeTensor(ub_sym_input); 
}

__aicore__ inline void Encoder::encode(int layer_id, int channel_id) {
    copy_in(layer_id, channel_id);

    AscendC::LocalTensor<uint16_t> ub_cdf_input = CDFInQ.DeQue<uint16_t>();

    AscendC::LocalTensor<uint8_t> encoded_output = outQ.AllocTensor<uint8_t>();
    uint32_t encoded_output_write_head = 0;

    // This calculation is inherently autoregressive in that encoding token N needs the state output from encoding
    // token N-1
    uint32_t low = 0U;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;
    const int max_symbol = n_bins -1;
    uint32_t output_reg = 0;
    uint32_t output_reg_write_head = 0;

    // OPTIMIZATION OPPORTUNITY
    //
    // Encode throughput is scalar bound, and this loop with the inner `while(true)` is the bulk of that work.
    // There are possibilities to vecotize over multiple channels at once and/or to replace the inner while loop with
    // with bitwise comparisons of high and low.
    //
    // Currently, the kernel isn't the overall encode bottleneck, but if that changes this is the main opportunity for 
    // optimization.
    for (uint32_t token_idx = 0; token_idx < n_tokens; token_idx += 1) {
        const uint8_t sym = calc_sym_input(token_idx);

        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        const uint32_t c_low = ub_cdf_input(sym);
        const uint32_t c_high = sym == max_symbol ? 0x10000U : ub_cdf_input(sym + 1);

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> 16);
        low = (low) + ((span * static_cast<uint64_t>(c_low)) >> 16);

        while (true) {
            if (high < 0x80000000U) {
                append_bit_and_pending(0, pending_bits, output_reg, output_reg_write_head, encoded_output, encoded_output_write_head);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x80000000U) {
                append_bit_and_pending(1, pending_bits, output_reg, output_reg_write_head, encoded_output, encoded_output_write_head);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            } else {
                break;
            }
        }
    }

    pending_bits += 1;
    if (low < 0x40000000U) {
        append_bit_and_pending(0, pending_bits, output_reg, output_reg_write_head, encoded_output, encoded_output_write_head);
    } else {
        append_bit_and_pending(1, pending_bits, output_reg, output_reg_write_head, encoded_output, encoded_output_write_head);
    }

    spill_partial_reg_to_shared(output_reg, output_reg_write_head, encoded_output, encoded_output_write_head);

    ub_encoded_output_len.SetValue(0, encoded_output_write_head);

    CDFInQ.FreeTensor(ub_cdf_input);
    outQ.EnQue(encoded_output);
    encoded_output = outQ.DeQue<uint8_t>();

    AscendC::DataCopyPad(gm_output_lens[layer_id * n_channels + channel_id], ub_encoded_output_len, {1, 1, 0, 0, 0});

    AscendC::DataCopyExtParams copyParams = {1, ub_encoded_output_len(0), 0, 0, 0};
    auto layer_size = chunk_size * n_channels;
    auto output_offset = layer_size * layer_id + chunk_size * channel_id; 
    AscendC::DataCopyPad(gm_output[output_offset], encoded_output, copyParams);

    outQ.FreeTensor(encoded_output);
}

__aicore__ inline void Encoder::append_bit_and_pending(
    uint32_t bit,
    uint64_t& pending_bits,
    uint32_t& output_reg,
    uint32_t& output_reg_write_head,
    AscendC::LocalTensor<uint8_t> encoded_output,
    uint32_t& encoded_output_write_head) {

    output_reg <<= 1;
    output_reg |= (bit << 1) - bit;
    output_reg_write_head += 1;
    if (output_reg_write_head == 32) {
      spill_reg_to_shared(output_reg, output_reg_write_head, encoded_output, encoded_output_write_head);
    }

    for(;pending_bits > 0;){
        int32_t pending_bit = 1 - bit;
        const unsigned int remaining = min(static_cast<unsigned int>(pending_bits), static_cast<unsigned int>(32 - output_reg_write_head));
        output_reg <<= remaining;
        output_reg |= (pending_bit << remaining) - pending_bit;
        pending_bits -= remaining;
        output_reg_write_head += remaining;
        if (output_reg_write_head == 32) {
            spill_reg_to_shared(output_reg, output_reg_write_head, encoded_output,
                                encoded_output_write_head);
        }
    }
}

__aicore__ inline void Encoder::spill_reg_to_shared(
    uint32_t& output_reg,
    uint32_t& output_reg_write_head,
    AscendC::LocalTensor<uint8_t> encoded_output,
    uint32_t& encoded_output_write_head) {

    output_reg <<= (32 - output_reg_write_head);
    encoded_output.ReinterpretCast<uint32_t>().SetValue(encoded_output_write_head >> 2, big_to_small(output_reg));

    encoded_output_write_head += 4;
    output_reg = 0;
    output_reg_write_head = 0;
}

__aicore__ inline void Encoder::spill_partial_reg_to_shared(
    uint32_t& output_reg,
    uint32_t& output_reg_write_head,
    AscendC::LocalTensor<uint8_t> encoded_output,
    uint32_t& encoded_output_write_head) {

    output_reg <<= (32 - output_reg_write_head);
    encoded_output.ReinterpretCast<uint32_t>().SetValue(
        encoded_output_write_head >> 2,
        ((output_reg & 0xFF000000U) >> 24) | ((output_reg & 0x00FF0000U) >> 8) | ((output_reg & 0x0000FF00U) << 8) | ((output_reg & 0x000000FFU) << 24));

    // Potentially only some of those bytes contain data. Progress the min number of bytes that includes all valid data
    encoded_output_write_head += output_reg_write_head / 8 + (output_reg_write_head % 8 > 0);
    output_reg = 0;
    output_reg_write_head = 0;
}
} // namespace impl
} // namespace cachegen
} // namespace kvcache_ops

extern "C" __global__ __aicore__ void encode__kernel (
    GM_ADDR cdf_data_ptr,
    GM_ADDR input_data_ptr,
    GM_ADDR output_data_ptr,
    GM_ADDR output_lengths_data_ptr,
    const uint32_t n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels,
    const int chunk_size
) {
    AscendC::TPipe pipe{};

    kvcache_ops::cachegen::impl::Encoder encoder {
        cdf_data_ptr,
        input_data_ptr,
        output_data_ptr,
        output_lengths_data_ptr,
        pipe,
        n_tokens,
        n_layers,
        n_channels,
        n_bins,
        chunk_size};

    int max_work_idx = n_layers * n_channels;

    int32_t coreIdx = AscendC::GetBlockIdx();
    int32_t launchedCores = AscendC::GetBlockNum();

    for (int work_idx = coreIdx; work_idx < max_work_idx; work_idx += launchedCores) {
        int layer_id = work_idx % n_layers;
        int channel_id = work_idx / n_layers;
        encoder.encode(layer_id, channel_id);
    }
}

namespace kvcache_ops {
namespace cachegen {

void encode(
    uint8_t* cdf_data_ptr,
    uint8_t* input_data_ptr,
    uint8_t* output_data_ptr,
    uint8_t* output_lengths_data_ptr,
    void* stream,
    const int n_aiv,
    const int n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels,
    const int chunk_size) {

    int blockDim = n_layers * n_channels < n_aiv ? n_layers * n_channels : n_aiv;

    encode__kernel<<<blockDim, nullptr, stream>>>(
        cdf_data_ptr,
        input_data_ptr,
        output_data_ptr,
        output_lengths_data_ptr,
        n_bins,
        n_tokens,
        n_layers,
        n_channels,
        chunk_size);
}
} // namespace cachegen
} // namespace kvcache_ops