#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace models {

struct Qwen3Hparams {
  static constexpr int32_t n_vocab = 151936;
  static constexpr int32_t n_layer = 28;
  static constexpr int32_t n_ctx = 40960;
  static constexpr int32_t n_embd = 1024;
  static constexpr int32_t n_embd_head = 128;
  static constexpr float rms_eps = 1e-6;
};

struct Qwen3Vocab {
  using id = int32_t;
  using token = std::string;

  std::unordered_map<token, id> token_to_id;
  std::unordered_map<id, token> id_to_token;
  std::vector<token> special_tokens;

  void AddSpecialToken(const token& token);
};

struct Qwen3Layer {
  // normalization
  struct ggml_tensor* attn_norm = nullptr;
  struct ggml_tensor* attn_q_norm = nullptr;
  struct ggml_tensor* attn_k_norm = nullptr;

  // attention
  struct ggml_tensor* wq = nullptr;
  struct ggml_tensor* wk = nullptr;
  struct ggml_tensor* wv = nullptr;
  struct ggml_tensor* wo = nullptr;

  // normalization
  struct ggml_tensor* ffn_norm = nullptr;

  // ff
  struct ggml_tensor* ffn_gate = nullptr;  // w1
  struct ggml_tensor* ffn_down = nullptr;  // w2
  struct ggml_tensor* ffn_up = nullptr;    // w3
};

class Qwen3Model {
public:
  explicit Qwen3Model(const std::string& model_path);

  void BuildGraph();

private:
  Qwen3Hparams hparams;

  struct ggml_tensor* tok_embd = nullptr;
  struct ggml_tensor* output = nullptr;
  struct ggml_tensor* output_norm = nullptr;

  std::vector<Qwen3Layer> layers;
  struct ggml_cgraph* compute_graph = nullptr;

  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer_w;
  ggml_backend_buffer_t buffer_kv;
};

}  // namespace models