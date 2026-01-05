#pragma once

struct Config {
  const char* model{};
  static constexpr int max_num_bathced_tokens = 16384;
  static constexpr int max_num_seqs = 512;
  static constexpr int max_model_len = 4096;
  static constexpr float gpu_memory_utilization = 0.9;
  static constexpr int tensor_parallel_size = 1;
  static constexpr bool enforce_eager = false;
  static constexpr int eos = -1;
  static constexpr int kvcache_block_size = 256;
  static constexpr int num_kvcache_blocks = -1;
};