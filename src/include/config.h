#pragma once

#include <string>

struct Config {
  std::string model;
  int max_num_bathced_tokens = 16384;
  int max_num_seqs = 512;
  int max_model_len = 4096;
  float gpu_memory_utilization = 0.9f;
  int tensor_parallel_size = 1;
  bool enforce_eager = true;    // no CUDA graph support
  int eos = -1;
  int kvcache_block_size = 256;
  int num_kvcache_blocks = -1;
};