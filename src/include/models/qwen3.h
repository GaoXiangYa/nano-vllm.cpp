#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "safetensors.hh"
#include "tokenizers_cpp.h"
namespace models {

struct TensorKeyOffset {
  static constexpr size_t kAttnNorm = 0;
  static constexpr size_t kFfnDown = 1;
  static constexpr size_t kFfnGate = 2;
  static constexpr size_t kFfnUp = 3;
  static constexpr size_t kFfnNorm = 4;
  static constexpr size_t kAttnKNorm = 5;
  static constexpr size_t kWk = 6;
  static constexpr size_t kWo = 7;
  static constexpr size_t kAttnQNorm = 8;
  static constexpr size_t kWq = 9;
  static constexpr size_t kWv = 10;
};

struct Qwen3Hparams {
  int32_t n_vocab = 151936;
  int32_t n_layer = 28;
  int32_t n_head = 16;
  int32_t n_head_kv = 8;    // GQA KV heads
  int32_t n_ctx = 40960;
  int32_t n_embd = 1024;
  int32_t n_embd_gqa = 1024;     // = n_head_kv * n_embd_head
  int32_t n_embd_k_gqa = 1024;
  int32_t n_embd_v_gqa = 1024;
  int32_t n_embd_head = 128;
  int32_t n_embd_head_k = 128;
  int32_t n_embd_head_v = 128;
  int32_t n_ff = 3072;

  float rope_freq_base = 1000000.0f;
  float rope_freq_scale = 1.0f;
  float rope_ext_factor = 1.0f;
  float attn_factor = 1.0f;
  float beta_fast = 32.0f;
  float beta_slow = 1.0f;

  float rms_eps = 1e-6f;

  int32_t eos_token_id = -1;
  int32_t bos_token_id = -1;
};

enum class Qwen3NormType {
  Norm,
  RmsNorm,
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
  struct ggml_tensor* attn_norm = nullptr;    // input_layernorm
  struct ggml_tensor* attn_q_norm = nullptr;  // q_norm
  struct ggml_tensor* attn_k_norm = nullptr;  // k_norm

  // attention
  struct ggml_tensor* wq = nullptr;  // q_proj
  struct ggml_tensor* wk = nullptr;  // k_proj
  struct ggml_tensor* wv = nullptr;  // v_proj
  struct ggml_tensor* wo = nullptr;  // o_proj

  // normalization
  struct ggml_tensor* ffn_norm = nullptr;  // post_attention_layernorm

  // ff
  struct ggml_tensor* ffn_gate = nullptr;  // w1, gate_proj
  struct ggml_tensor* ffn_down = nullptr;  // w2, down_proj
  struct ggml_tensor* ffn_up = nullptr;    // w3, up_proj
};

class Qwen3Model {
public:
  explicit Qwen3Model(const std::string& model_path);
  Qwen3Model(const Qwen3Model& other) = delete;
  Qwen3Model& operator=(const Qwen3Model& other) = delete;
  Qwen3Model(Qwen3Model&& other) = delete;
  Qwen3Model& operator=(Qwen3Model&& other) = delete;
  ~Qwen3Model();

  // Build the compute graph for processing n_tokens
  void BuildPrefillGraph(int n_tokens, int cache_offset, int n_outputs = -1);

  const std::unique_ptr<tokenizers::Tokenizer>& GetTokenizer() const {
    return tokenizer_;
  }

  // Execution helpers
  bool AllocateGraph();
  ggml_status GraphCompute();
  void SetInputTensor(ggml_tensor* tensor, const void* data, size_t size);

  struct ggml_cgraph* GetComputeGraph() const { return compute_graph_; }
  ggml_tensor* GetLogits() const { return logits; }
  const Qwen3Hparams& GetHparams() const { return hparams_; }
  int GetEosTokenId() const { return hparams_.eos_token_id; }
  ggml_gallocr_t GetAllocr() const { return allocr_; }
  ggml_backend_t GetBackend() const { return backend_; }
  ggml_context* GetComputeCtx() const { return ctx_compute; }

  // KV cache read/write (outside graph, for runner)
  void WriteKVToCache(int layer, int cache_pos,
                      ggml_tensor* k_tensor, ggml_tensor* v_tensor);
  void ReadKVFromCache(int layer, int cache_start, int n_tokens,
                       void* k_out, void* v_out);

public:
  void Reset();
  ggml_tensor* BuildInputEmbedding(ggml_tensor* tok_emd, const int n_tokens);
  ggml_tensor* BuildInputPosition(int n_positions);
  ggml_tensor* BuildInputIds(int n_ids);
  ggml_tensor* BuildNorm(ggml_tensor* input, ggml_tensor* norm_weight,
                         ggml_tensor* norm_bias,
                         const Qwen3NormType& norm_type);
  ggml_tensor* BuildFFN(ggml_tensor* input, ggml_tensor* ffn_up,
                        ggml_tensor* ffn_gate, ggml_tensor* ffn_down);
  ggml_tensor* BuildKQMask(int n_kv, int n_tokens, bool is_prefill);
  ggml_tensor* BuildAttention(ggml_tensor* q_cur, ggml_tensor* k_cur,
                              ggml_tensor* v_cur, ggml_tensor* mask);
  ggml_tensor* BuildOutputNorm(ggml_tensor* input);
  ggml_tensor* BuildOutputLayer(ggml_tensor* input);

private:
  static constexpr int kQwen3MaxNodes = 2488;
  Qwen3Hparams hparams_;
  void LoadConfigJson(const std::string& config_path);
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<safetensors::safetensors_t> safetensors_;

  // memory buffers used to evaluate the model
  std::vector<uint8_t> buf_compute_meta;

  struct ggml_tensor* tok_embd_ = nullptr;
  struct ggml_tensor* output_ = nullptr;
  struct ggml_tensor* output_norm_ = nullptr;
  struct ggml_tensor* logits = nullptr;

  struct ggml_tensor* memory_k = nullptr;
  struct ggml_tensor* memory_v = nullptr;
  struct ggml_tensor* memory_kq_mask = nullptr;
  struct ggml_tensor* memory_kq_mask_cnv = nullptr;

  struct ggml_context* ctx_w = nullptr;
  struct ggml_context* ctx_kv = nullptr;
  struct ggml_context* ctx_compute = nullptr;

  std::vector<Qwen3Layer> layers_;
  struct ggml_cgraph* compute_graph_ = nullptr;

  ggml_gallocr_t allocr_ = nullptr;

  ggml_backend_t backend_ = nullptr;
  ggml_backend_buffer_t buffer_w_ = nullptr;
  ggml_backend_buffer_t buffer_kv_ = nullptr;

  std::unordered_map<std::string, struct ggml_tensor*> tensors_;
};

}  // namespace models