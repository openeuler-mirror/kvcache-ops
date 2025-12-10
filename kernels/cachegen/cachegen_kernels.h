namespace kvcache_ops {
namespace cachegen{
    void calculate_cdf(
        uint8_t* input,
        uint8_t* output,
        void* stream,
        const int n_aiv,
        const int n_bins, 
        const int n_tokens, 
        const int n_layers, 
        const int n_channels);

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
        const int chunk_size);

    void decode(
        uint8_t* cdf_data_ptr,
        uint8_t* bytestreams_data_ptr,
        uint8_t* lengths_data_ptr,
        uint8_t* output_data_ptr,
        void* stream,
        const int n_aiv,
        const int n_bins,
        const int n_tokens,
        const int n_layers,
        const int n_channels);
} // cachegen
} // kvcache_ops
