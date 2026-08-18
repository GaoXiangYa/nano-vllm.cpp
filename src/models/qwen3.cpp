#include "models/qwen3.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <source_location>
#include <stdexcept>
#include <string>
#include "config.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "safetensors.hh"
#include "tokenizers_cpp.h"

#ifdef USE_CUDA
#include "ggml-cuda.h"
#endif
#include "ggml-cpu.h"

namespace models {

Qwen3Model::Qwen3Model(const std::string& model_path, int n_ctx_total,
                         int num_kvcache_blocks, int kvcache_block_size)
    : n_ctx_total_(n_ctx_total),
      num_kvcache_blocks_(num_kvcache_blocks),
      kvcache_block_size_(kvcache_block_size) {
  std::filesystem::path path(model_path);
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Model path does not exist: " + model_path);
  }

  // Load model hyperparameters from config.json
  LoadConfigJson((path / "config.json").string());

  // If the caller did not specify a context size, use the model's own limit.
  // The KV cache is paged by fixed-size blocks, so round the slot count up to
  // a whole number of blocks.
  if (n_ctx_total_ <= 0) {
    n_ctx_total_ = hparams_.n_ctx;
  }
  if (kvcache_block_size_ <= 0) {
    kvcache_block_size_ = 256;
  }
  if (num_kvcache_blocks_ <= 0) {
    num_kvcache_blocks_ =
        (n_ctx_total_ + kvcache_block_size_ - 1) / kvcache_block_size_;
  }
  n_ctx_total_ = num_kvcache_blocks_ * kvcache_block_size_;

  std::filesystem::path tokenizer_path = path / "tokenizer.json";
  std::filesystem::path safetensors_path = path / "model.safetensors";

  // Read tokenizer JSON file contents
  {
    std::ifstream tfile(tokenizer_path, std::ios::binary);
    if (!tfile.is_open()) {
      throw std::runtime_error("Cannot open tokenizer: " + tokenizer_path.string());
    }
    std::stringstream buffer;
    buffer << tfile.rdbuf();
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(buffer.str());
  }

  std::string warn, err;
  safetensors_ = std::make_unique<safetensors::safetensors_t>();
  auto ret = safetensors::mmap_from_file(safetensors_path.c_str(),
                                          safetensors_.get(), &warn, &err);
  if (!warn.empty()) {
    std::cout << std::format("WARN: {}\n", warn);
  }
  if (!ret) {
    throw std::runtime_error("Failed to load " + safetensors_path.string() +
                             " ERR: " + err);
  }

  const int32_t n_tensors = 1 + 2 + 11 * hparams_.n_layer;
  const int32_t max_tensors = n_tensors + 1 + 2 * hparams_.n_layer;
  const auto ctx_size = ggml_tensor_overhead() * max_tensors;
  constexpr float kMB = 1024.0f * 1024.0f;

  // Create model weight context
  {
    struct ggml_init_params params = {
        ctx_size,
        nullptr,
        true,  // no_alloc
    };
    this->ctx_w = ggml_init(params);
    if (this->ctx_w == nullptr) {
      throw std::runtime_error("ggml_init() failed for ctx_w");
    }
  }

  // Initialize backend
#ifdef USE_CUDA
  this->backend_ = ggml_backend_cuda_init(0);
#endif
  if (this->backend_ == nullptr) {
    this->backend_ = ggml_backend_cpu_init();
  }
  if (this->backend_ == nullptr) {
    throw std::runtime_error("Failed to initialize any ggml backend!");
  }

  // Create tensors from safetensors layout
  {
    const auto n_vocab = this->hparams_.n_vocab;
    const auto n_embd = this->hparams_.n_embd;
    const auto n_ff = this->hparams_.n_ff;
    const auto n_layer = this->hparams_.n_layer;
    const auto n_embd_head = this->hparams_.n_embd_head;
    const auto n_embd_gqa = this->hparams_.n_embd_k_gqa;
    const auto n_head = this->hparams_.n_head;
    const auto n_embd_head_k = this->hparams_.n_embd_head_k;

    this->layers_.resize(n_layer);

    // output weight (lm_head)
    this->output_ =
        ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd, n_vocab);
    this->tensors_["lm_head.weight"] = this->output_;

    // token embedding
    this->tok_embd_ =
        ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd, n_vocab);
    this->tensors_["model.embed_tokens.weight"] = this->tok_embd_;

    // Per-layer tensors (mapped by name, not by file order)
    for (int layer_idx = 0; layer_idx < n_layer; ++layer_idx) {
      auto& layer = this->layers_[layer_idx];
      const std::string prefix =
          "model.layers." + std::to_string(layer_idx) + ".";

      layer.attn_norm =
          ggml_new_tensor_1d(this->ctx_w, GGML_TYPE_BF16, n_embd);
      layer.ffn_down =
          ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_ff, n_embd);
      layer.ffn_gate =
          ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd, n_ff);
      layer.ffn_up =
          ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd, n_ff);
      layer.ffn_norm =
          ggml_new_tensor_1d(this->ctx_w, GGML_TYPE_BF16, n_embd);
      layer.attn_k_norm =
          ggml_new_tensor_1d(this->ctx_w, GGML_TYPE_BF16, n_embd_head_k);
      layer.wk = ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd,
                                     n_embd_gqa);
      layer.wo =
          ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16,
                             n_embd_head_k * n_head, n_embd);
      layer.attn_q_norm =
          ggml_new_tensor_1d(this->ctx_w, GGML_TYPE_BF16, n_embd_head_k);
      layer.wq = ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd,
                                     n_embd_head_k * n_head);
      layer.wv = ggml_new_tensor_2d(this->ctx_w, GGML_TYPE_BF16, n_embd,
                                     n_embd_gqa);

      this->tensors_[prefix + "input_layernorm.weight"] = layer.attn_norm;
      this->tensors_[prefix + "mlp.down_proj.weight"] = layer.ffn_down;
      this->tensors_[prefix + "mlp.gate_proj.weight"] = layer.ffn_gate;
      this->tensors_[prefix + "mlp.up_proj.weight"] = layer.ffn_up;
      this->tensors_[prefix + "post_attention_layernorm.weight"] =
          layer.ffn_norm;
      this->tensors_[prefix + "self_attn.k_norm.weight"] = layer.attn_k_norm;
      this->tensors_[prefix + "self_attn.k_proj.weight"] = layer.wk;
      this->tensors_[prefix + "self_attn.o_proj.weight"] = layer.wo;
      this->tensors_[prefix + "self_attn.q_norm.weight"] = layer.attn_q_norm;
      this->tensors_[prefix + "self_attn.q_proj.weight"] = layer.wq;
      this->tensors_[prefix + "self_attn.v_proj.weight"] = layer.wv;
    }

    // output norm
    this->output_norm_ =
        ggml_new_tensor_1d(this->ctx_w, GGML_TYPE_BF16, n_embd);
    this->tensors_["model.norm.weight"] = this->output_norm_;
  }

  // Allocate weight tensors in backend buffer
  this->buffer_w_ = ggml_backend_alloc_ctx_tensors(this->ctx_w, this->backend_);

  // Allocate KV cache: [n_layer][num_blocks * block_size][n_embd_gqa].
  // This is a paged flat buffer: a physical slot is block_id*block_size+offset.
  {
    const auto n_layer = this->hparams_.n_layer;
    const auto n_embd_gqa = this->hparams_.n_embd_k_gqa;
    const size_t n_slots = static_cast<size_t>(num_kvcache_blocks_) *
                           static_cast<size_t>(kvcache_block_size_);
    const size_t n_elements = static_cast<size_t>(n_layer) * n_slots * n_embd_gqa;

    struct ggml_init_params kv_params = {
        ggml_tensor_overhead() * 2,
        nullptr,
        true,
    };
    this->ctx_kv = ggml_init(kv_params);

    this->memory_k =
        ggml_new_tensor_1d(this->ctx_kv, GGML_TYPE_BF16, static_cast<int64_t>(n_elements));
    this->memory_v =
        ggml_new_tensor_1d(this->ctx_kv, GGML_TYPE_BF16, static_cast<int64_t>(n_elements));

    this->buffer_kv_ =
        ggml_backend_alloc_ctx_tensors(this->ctx_kv, this->backend_);

    const size_t memory_size = ggml_backend_buffer_get_size(this->buffer_kv_);
    std::cout << std::format("KV cache size = {:8.2f} MB\n",
                             static_cast<float>(memory_size) / kMB);
  }

  // Load weights from safetensors
  {
    size_t total_size = 0;
    std::span<std::byte> databuffer{
        safetensors_->mmaped
            ? std::bit_cast<std::byte*>(safetensors_->databuffer_addr)
            : std::bit_cast<std::byte*>(safetensors_->storage.data()),
        safetensors_->databuffer_size};

    for (size_t i = 0; i < safetensors_->tensors.size(); ++i) {
      const auto& tensor_name = safetensors_->tensors.keys()[i];
      if (!this->tensors_.contains(tensor_name)) {
        throw std::runtime_error(std::format(
            "Unknown tensor: {} in {}", tensor_name,
            std::source_location::current().function_name()));
      }
      auto* tensor = this->tensors_[tensor_name];
      safetensors::tensor_t stensor;
      safetensors_->tensors.at(tensor_name, &stensor);
      auto* tensor_data =
          databuffer.data() + stensor.data_offsets[0];

      ggml_backend_tensor_set(tensor, tensor_data, 0, ggml_nbytes(tensor));
      total_size += ggml_nbytes(tensor);
    }
    std::cout << std::format("Loaded model size = {:8.2f} MB\n",
                             static_cast<float>(total_size) / kMB);

  }
}

Qwen3Model::~Qwen3Model() {
  if (this->allocr_) {
    ggml_gallocr_free(this->allocr_);
  }
  if (this->buffer_w_) {
    ggml_backend_buffer_free(this->buffer_w_);
  }
  if (this->buffer_kv_) {
    ggml_backend_buffer_free(this->buffer_kv_);
  }
  if (this->ctx_w) {
    ggml_free(this->ctx_w);
  }
  if (this->ctx_kv) {
    ggml_free(this->ctx_kv);
  }
  if (this->backend_) {
    ggml_backend_free(this->backend_);
  }
}

void Qwen3Model::Reset() {
  static size_t buf_size = ggml_tensor_overhead() * kQwen3MaxNodes +
                           ggml_graph_overhead_custom(kQwen3MaxNodes, false);
  this->buf_compute_meta.resize(buf_size);
  struct ggml_init_params params = {
      buf_size,
      buf_compute_meta.data(),
      true,
  };
  this->ctx_compute = ggml_init(params);
  this->compute_graph_ =
      ggml_new_graph_custom(ctx_compute, kQwen3MaxNodes, false);

  // Lazily create the allocator
  if (this->allocr_ == nullptr) {
    this->allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(this->backend_));
  }
}

bool Qwen3Model::AllocateGraph() {
  if (this->allocr_ == nullptr) {
    this->allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(this->backend_));
  }
  return ggml_gallocr_alloc_graph(this->allocr_, this->compute_graph_);
}

ggml_status Qwen3Model::GraphCompute() {
  // Execute on backend (assumes AllocateGraph was already called)
  return ggml_backend_graph_compute(this->backend_, this->compute_graph_);
}

void Qwen3Model::SetInputTensor(ggml_tensor* tensor, const void* data,
                                 size_t size) {
  ggml_backend_tensor_set(tensor, data, 0, size);
}

// Build input embedding: token_ids -> embedding lookup
ggml_tensor* Qwen3Model::BuildInputEmbedding(ggml_tensor* tok_emd,
                                              const int n_tokens) {
  auto embd = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I32, n_tokens);
  ggml_set_name(embd, "inp_tokens");
  ggml_set_input(embd);
  return ggml_get_rows(this->ctx_compute, tok_emd, embd);
}

ggml_tensor* Qwen3Model::BuildInputPosition(int n_positions) {
  auto pos = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I32, n_positions);
  ggml_set_name(pos, "inp_pos");
  ggml_set_input(pos);
  return pos;
}

ggml_tensor* Qwen3Model::BuildInputIds(int n_ids) {
  auto ids = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I32, n_ids);
  ggml_set_name(ids, "inp_out_ids");
  ggml_set_input(ids);
  return ids;
}

ggml_tensor* Qwen3Model::BuildNorm(ggml_tensor* input,
                                    ggml_tensor* norm_weight,
                                    ggml_tensor* norm_bias,
                                    const Qwen3NormType& norm_type) {
  switch (norm_type) {
    case Qwen3NormType::Norm:
      input = ggml_norm(this->ctx_compute, input, this->hparams_.rms_eps);
      break;
    case Qwen3NormType::RmsNorm:
      input = ggml_rms_norm(this->ctx_compute, input, this->hparams_.rms_eps);
      break;
    default:
      throw std::runtime_error("Unsupported norm type!");
  }
  if (norm_weight) {
    if (norm_weight->type != GGML_TYPE_F32) {
      norm_weight = ggml_cast(this->ctx_compute, norm_weight, GGML_TYPE_F32);
    }
    input = ggml_mul(this->ctx_compute, input, norm_weight);
  }
  if (norm_bias) {
    if (norm_bias->type != GGML_TYPE_F32) {
      norm_bias = ggml_cast(this->ctx_compute, norm_bias, GGML_TYPE_F32);
    }
    input = ggml_add(this->ctx_compute, input, norm_bias);
  }
  return input;
}

ggml_tensor* Qwen3Model::BuildFFN(ggml_tensor* input, ggml_tensor* ffn_up,
                                   ggml_tensor* ffn_gate,
                                   ggml_tensor* ffn_down) {
  auto gate = ggml_mul_mat(this->ctx_compute, ffn_gate, input);
  auto up = ggml_mul_mat(this->ctx_compute, ffn_up, input);
  input = ggml_swiglu_split(this->ctx_compute, gate, up);
  input = ggml_mul_mat(this->ctx_compute, ffn_down, input);
  return input;
}

// Build KQ mask for attention  
// Mask must be F16 4D for ggml_flash_attn_ext: [n_kv, n_batch_pad, 1, 1]
ggml_tensor* Qwen3Model::BuildKQMask(int n_kv, int n_tokens, bool /*is_prefill*/) {
  constexpr int kPad = GGML_KQ_MASK_PAD;
  int n_tokens_pad = ((n_tokens + kPad - 1) / kPad) * kPad;
  auto mask = ggml_new_tensor_4d(this->ctx_compute, GGML_TYPE_F16,
                                  n_kv, n_tokens_pad, 1, 1);
  ggml_set_name(mask, "inp_kq_mask");
  ggml_set_input(mask);
  return mask;
}

// Build attention using flash attention
ggml_tensor* Qwen3Model::BuildAttention(ggml_tensor* q_cur,
                                         ggml_tensor* k_cur,
                                         ggml_tensor* v_cur,
                                         ggml_tensor* mask) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(hparams_.n_embd_head));

  // ggml's CUDA flash-attention kernels only support F16 (or quantized) K/V.
  // The Q/K/V projections here are F32, so cast K/V to F16 for the FA op.
  // The caller still keeps the original F32 K/V tensors for the KV cache
  // outputs, so cache stores are unaffected.
  if (k_cur->type != GGML_TYPE_F16) {
    k_cur = ggml_cast(this->ctx_compute, k_cur, GGML_TYPE_F16);
  }
  if (v_cur->type != GGML_TYPE_F16) {
    v_cur = ggml_cast(this->ctx_compute, v_cur, GGML_TYPE_F16);
  }

  auto kq = ggml_flash_attn_ext(this->ctx_compute, q_cur, k_cur, v_cur, mask,
                                 scale, 0.0f, 0.0f);

  // res: [n_embd_v, n_head, n_batch] = [128, 16, n_tokens]
  // Merge ne0 (head_dim) x ne1 (heads) -> [2048, n_tokens]
  kq = ggml_cont(this->ctx_compute, kq);
  kq = ggml_reshape_2d(this->ctx_compute, kq,
                        hparams_.n_embd_head * hparams_.n_head,
                        ggml_nelements(kq) / (hparams_.n_embd_head * hparams_.n_head));
  return kq;
}

// KV cache helpers (outside graph, used by runner)
// Paged layout: [n_layer][num_blocks][block_size][n_embd_gqa] stored as one
// flat BF16 buffer.  The caller passes the physical slot:
//   slot = block_id * block_size + offset_within_block
void Qwen3Model::WriteKVToCache(int layer, int slot,
                                const void* k_data, const void* v_data) {
  const auto n_embd_gqa = hparams_.n_embd_k_gqa;
  const size_t offset =
      (static_cast<size_t>(layer) * n_ctx_total_ + slot) * n_embd_gqa;
  const size_t byte_offset = offset * ggml_type_size(GGML_TYPE_BF16);
  const size_t n_bytes =
      static_cast<size_t>(n_embd_gqa) * ggml_type_size(GGML_TYPE_BF16);

  ggml_backend_tensor_set(memory_k, k_data, byte_offset, n_bytes);
  ggml_backend_tensor_set(memory_v, v_data, byte_offset, n_bytes);
}

void Qwen3Model::ReadKVFromCache(int layer, int slot,
                                 void* k_out, void* v_out) {
  const auto n_embd_gqa = hparams_.n_embd_k_gqa;
  const size_t offset =
      (static_cast<size_t>(layer) * n_ctx_total_ + slot) * n_embd_gqa;
  const size_t byte_offset = offset * ggml_type_size(GGML_TYPE_BF16);
  const size_t n_bytes =
      static_cast<size_t>(n_embd_gqa) * ggml_type_size(GGML_TYPE_BF16);

  ggml_backend_tensor_get(memory_k, k_out, byte_offset, n_bytes);
  ggml_backend_tensor_get(memory_v, v_out, byte_offset, n_bytes);
}

ggml_tensor* Qwen3Model::BuildOutputNorm(ggml_tensor* input) {
  return BuildNorm(input, this->output_norm_, nullptr, Qwen3NormType::RmsNorm);
}

ggml_tensor* Qwen3Model::BuildOutputLayer(ggml_tensor* input) {
  return ggml_mul_mat(this->ctx_compute, this->output_, input);
}

// Build prefill graph
// All tokens are new (no prefix cache reuse in attention)
void Qwen3Model::BuildPrefillGraph(int n_tokens, int n_outputs) {
  Reset();

  const int n_rot = hparams_.n_embd_head;
  auto cur = BuildInputEmbedding(this->tok_embd_, n_tokens);
  // Cast to F32 for computation (CPU backend doesn't support BF16 arith)
  cur = ggml_cast(this->ctx_compute, cur, GGML_TYPE_F32);
  auto pos = BuildInputPosition(n_tokens);
  if (n_outputs < 0) n_outputs = n_tokens;
  auto inp_out_ids = BuildInputIds(n_outputs);

  // KQ mask for prefill: causal [n_tokens, n_tokens].
  // ggml's CUDA FlashAttention requires the KV sequence length to be a
  // multiple of FATTN_KQ_STRIDE (256), so allocate/pad the KV dimension.
  const int n_kv_pad = GGML_PAD(n_tokens, 256);
  auto mask = BuildKQMask(n_kv_pad, n_tokens, true);

  // Qwen3 pre-norm residual fusion; first layer has no residual (None in Python)
  ggml_tensor* residual = nullptr;

  for (int il = 0; il < hparams_.n_layer; ++il) {
    auto& layer = layers_[il];

    // --- Input layernorm: add residual (except first layer), then norm ---
    if (residual != nullptr) {
      cur = ggml_add(this->ctx_compute, cur, residual);
    }
    auto attn_residual = cur;  // save for post-attention
    cur = BuildNorm(cur, layer.attn_norm, nullptr, Qwen3NormType::RmsNorm);

    // QKV projections
    auto q_cur = ggml_mul_mat(this->ctx_compute, layer.wq, cur);
    auto k_cur = ggml_mul_mat(this->ctx_compute, layer.wk, cur);
    auto v_cur = ggml_mul_mat(this->ctx_compute, layer.wv, cur);
    // Keep the pre-reshape V projection tensor alive until after the graph
    // runs.  The runner reads the permuted view "v_out_i" after compute; the
    // view does not own storage, so the storage tensor itself must be a graph
    // output, otherwise ggml's allocator may reuse its buffer.
    ggml_set_name(v_cur, std::format("v_proj_raw_{}", il).c_str());
    ggml_set_output(v_cur);

    // Reshape to [head_dim, n_head, n_tokens]: matches raw memory layout,
    // where each token's 2048-dim (16 heads x 128) is contiguous
    q_cur = ggml_reshape_3d(this->ctx_compute, q_cur, hparams_.n_embd_head,
                             hparams_.n_head, n_tokens);
    k_cur = ggml_reshape_3d(this->ctx_compute, k_cur, hparams_.n_embd_head,
                             hparams_.n_head_kv, n_tokens);
    v_cur = ggml_reshape_3d(this->ctx_compute, v_cur, hparams_.n_embd_head,
                             hparams_.n_head_kv, n_tokens);

    // Q/K norm (acts on ne0=head_dim, layout-independent)
    q_cur = BuildNorm(q_cur, layer.attn_q_norm, nullptr,
                       Qwen3NormType::RmsNorm);
    k_cur = BuildNorm(k_cur, layer.attn_k_norm, nullptr,
                       Qwen3NormType::RmsNorm);

    // RoPE: positions indexed by ne2=seq-len (see ggml-cpu/ops.cpp rope loop)
    // Qwen3 uses rotate_half RoPE (pairs (i, i+n_dims/2)) = 0
    q_cur = ggml_rope_ext(this->ctx_compute, q_cur, pos, nullptr, n_rot,
                           GGML_ROPE_TYPE_NEOX, hparams_.n_ctx,
                           hparams_.rope_freq_base, hparams_.rope_freq_scale,
                           hparams_.rope_ext_factor, hparams_.attn_factor,
                           hparams_.beta_fast, hparams_.beta_slow);
    k_cur = ggml_rope_ext(this->ctx_compute, k_cur, pos, nullptr, n_rot,
                           GGML_ROPE_TYPE_NEOX, hparams_.n_ctx,
                           hparams_.rope_freq_base, hparams_.rope_freq_scale,
                           hparams_.rope_ext_factor, hparams_.attn_factor,
                           hparams_.beta_fast, hparams_.beta_slow);
    // This is the storage tensor backing the permuted "k_out_i" view read by
    // the runner after graph execution; it must itself be a graph output so
    // ggml's allocator keeps its buffer alive.
    ggml_set_name(k_cur, std::format("k_out_{}", il).c_str());
    ggml_set_output(k_cur);
    ggml_set_name(v_cur, std::format("v_out_{}", il).c_str());
    ggml_set_output(v_cur);


    auto k_attn = ggml_pad(this->ctx_compute, k_cur, 0, 0,
                           n_kv_pad - n_tokens, 0);
    auto v_attn = ggml_pad(this->ctx_compute, v_cur, 0, 0,
                           n_kv_pad - n_tokens, 0);
    q_cur = ggml_permute(this->ctx_compute, q_cur, 0, 2, 1, 3);
    k_attn = ggml_permute(this->ctx_compute, k_attn, 0, 2, 1, 3);
    v_attn = ggml_permute(this->ctx_compute, v_attn, 0, 2, 1, 3);

    // Attention with self K,V (causal mask)
    cur = BuildAttention(q_cur, k_attn, v_attn, mask);

    // (KV-cache outputs are the unpadded raw K/V tensors named above.)

    // Output projection
    cur = ggml_mul_mat(this->ctx_compute, layer.wo, cur);

    // --- Post-attention layernorm: add residual, then norm ---
    cur = ggml_add(this->ctx_compute, cur, attn_residual);
    residual = cur;  // save for next layer
    cur = BuildNorm(cur, layer.ffn_norm, nullptr, Qwen3NormType::RmsNorm);

    // FFN (no internal residual - handled by the norm above)
    cur = BuildFFN(cur, layer.ffn_up, layer.ffn_gate, layer.ffn_down);

  }

  // Extract output tokens from last layer
  if (inp_out_ids != nullptr) {
    cur = ggml_get_rows(this->ctx_compute, cur, inp_out_ids);
    residual = ggml_get_rows(this->ctx_compute, residual, inp_out_ids);
  }

  // Final output norm: norm(cur + residual)
  cur = ggml_add(this->ctx_compute, cur, residual);
  cur = BuildOutputNorm(cur);
  this->logits = BuildOutputLayer(cur);
  // Ensure logits are F32
  if (this->logits->type != GGML_TYPE_F32) {
    this->logits = ggml_cast(this->ctx_compute, this->logits, GGML_TYPE_F32);
  }

  ggml_build_forward_expand(this->compute_graph_, this->logits);
}

// Build decode graph: one new token per sequence.
// History K/V is pre-filled by the runner into k_hist_in/v_hist_in
// ([head_dim, total_hist, n_head_kv]); the newly computed K/V is concatenated
// in-graph so attention sees the full context. New K/V are graph outputs so
// the runner can append them to the KV cache.
void Qwen3Model::BuildDecodeGraph(int n_seqs, int total_hist,
                                  const std::vector<int>& /*ctx_lens*/) {
  Reset();

  const int n_rot = hparams_.n_embd_head;
  const int total_kv = total_hist + n_seqs;

  auto cur = BuildInputEmbedding(this->tok_embd_, n_seqs);
  cur = ggml_cast(this->ctx_compute, cur, GGML_TYPE_F32);
  auto pos = BuildInputPosition(n_seqs);

  // Varlen mask [total_kv_pad, PAD(n_seqs, 64)]: filled by runner per
  // sequence.  total_kv is padded to ggml's CUDA FlashAttention KV stride.
  const int total_kv_pad = GGML_PAD(total_kv, 256);
  auto mask = BuildKQMask(total_kv_pad, n_seqs, false);

  ggml_tensor* residual = nullptr;
  for (int il = 0; il < hparams_.n_layer; ++il) {
    auto& layer = layers_[il];

    // Per-layer history K/V inputs (runner pre-fills from KV cache)
    auto k_hist = ggml_new_tensor_3d(
        this->ctx_compute, GGML_TYPE_F32, hparams_.n_embd_head,
        hparams_.n_head_kv, total_hist);
    auto v_hist = ggml_new_tensor_3d(
        this->ctx_compute, GGML_TYPE_F32, hparams_.n_embd_head,
        hparams_.n_head_kv, total_hist);
    ggml_set_name(k_hist, std::format("k_hist_in_{}", il).c_str());
    ggml_set_name(v_hist, std::format("v_hist_in_{}", il).c_str());
    ggml_set_input(k_hist);
    ggml_set_input(v_hist);

    if (residual != nullptr) {
      cur = ggml_add(this->ctx_compute, cur, residual);
    }
    auto attn_residual = cur;
    cur = BuildNorm(cur, layer.attn_norm, nullptr, Qwen3NormType::RmsNorm);

    auto q_cur = ggml_mul_mat(this->ctx_compute, layer.wq, cur);
    auto k_cur = ggml_mul_mat(this->ctx_compute, layer.wk, cur);
    auto v_cur = ggml_mul_mat(this->ctx_compute, layer.wv, cur);
    // Same lifetime requirement as prefill: the append step reads permuted
    // views of the projection outputs after graph execution.
    ggml_set_name(v_cur, std::format("v_proj_raw_{}", il).c_str());
    ggml_set_output(v_cur);

    // Rope layout [head_dim, n_head, n_seqs] (positions indexed by ne2)
    q_cur = ggml_reshape_3d(this->ctx_compute, q_cur, hparams_.n_embd_head,
                             hparams_.n_head, n_seqs);
    k_cur = ggml_reshape_3d(this->ctx_compute, k_cur, hparams_.n_embd_head,
                             hparams_.n_head_kv, n_seqs);
    v_cur = ggml_reshape_3d(this->ctx_compute, v_cur, hparams_.n_embd_head,
                             hparams_.n_head_kv, n_seqs);

    q_cur = BuildNorm(q_cur, layer.attn_q_norm, nullptr,
                       Qwen3NormType::RmsNorm);
    k_cur = BuildNorm(k_cur, layer.attn_k_norm, nullptr,
                       Qwen3NormType::RmsNorm);

    q_cur = ggml_rope_ext(this->ctx_compute, q_cur, pos, nullptr, n_rot,
                           GGML_ROPE_TYPE_NEOX, hparams_.n_ctx,
                           hparams_.rope_freq_base, hparams_.rope_freq_scale,
                           hparams_.rope_ext_factor, hparams_.attn_factor,
                           hparams_.beta_fast, hparams_.beta_slow);
    k_cur = ggml_rope_ext(this->ctx_compute, k_cur, pos, nullptr, n_rot,
                           GGML_ROPE_TYPE_NEOX, hparams_.n_ctx,
                           hparams_.rope_freq_base, hparams_.rope_freq_scale,
                           hparams_.rope_ext_factor, hparams_.attn_factor,
                           hparams_.beta_fast, hparams_.beta_slow);
    // The runner appends this RoPE'd K tensor (via the later permuted view
    // "k_out_i") to the KV cache after the graph has finished.  Make the
    // storage tensor a graph output so the allocator cannot reuse it.
    ggml_set_name(k_cur, std::format("d_k_rope_{}", il).c_str());
    ggml_set_output(k_cur);

    // Q permutes to flash-attn layout [head_dim, n_seqs, n_head]
    q_cur = ggml_permute(this->ctx_compute, q_cur, 0, 2, 1, 3);

    // K/V stay in [head_dim, n_head_kv, positions]; concat history + current
    // along the position dim, then permute to flash-attn layout
    auto k_full = ggml_concat(this->ctx_compute, k_hist, k_cur, 2);
    auto v_full = ggml_concat(this->ctx_compute, v_hist, v_cur, 2);
    k_full = ggml_pad(this->ctx_compute, k_full, 0, 0,
                      total_kv_pad - total_kv, 0);
    v_full = ggml_pad(this->ctx_compute, v_full, 0, 0,
                      total_kv_pad - total_kv, 0);
    k_full = ggml_permute(this->ctx_compute, k_full, 0, 2, 1, 3);
    v_full = ggml_permute(this->ctx_compute, v_full, 0, 2, 1, 3);

    cur = BuildAttention(q_cur, k_full, v_full, mask);

    // New K/V as graph outputs for cache append
    ggml_set_name(k_cur, std::format("k_out_{}", il).c_str());
    ggml_set_name(v_cur, std::format("v_out_{}", il).c_str());
    ggml_set_output(k_cur);
    ggml_set_output(v_cur);

    cur = ggml_mul_mat(this->ctx_compute, layer.wo, cur);

    cur = ggml_add(this->ctx_compute, cur, attn_residual);
    residual = cur;
    cur = BuildNorm(cur, layer.ffn_norm, nullptr, Qwen3NormType::RmsNorm);
    cur = BuildFFN(cur, layer.ffn_up, layer.ffn_gate, layer.ffn_down);
  }

  cur = ggml_add(this->ctx_compute, cur, residual);
  cur = BuildOutputNorm(cur);
  this->logits = BuildOutputLayer(cur);
  if (this->logits->type != GGML_TYPE_F32) {
    this->logits = ggml_cast(this->ctx_compute, this->logits, GGML_TYPE_F32);
  }

  ggml_build_forward_expand(this->compute_graph_, this->logits);
}

void Qwen3Model::LoadConfigJson(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    std::cerr << "Warning: cannot open config.json at " << config_path
              << ", using default hparams\n";
    return;
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

  // Simple JSON parser for flat config objects
  auto get_int = [&](const std::string& key, int default_val = 0) -> int {
    std::regex pattern("\"" + key + R"("\s*:\s*(\d+))");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
      return std::stoi(match[1].str());
    }
    return default_val;
  };

  auto get_float = [&](const std::string& key, float default_val = 0.0f) -> float {
    std::regex pattern("\"" + key + R"("\s*:\s*([\d.eE+-]+))");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
      return std::stof(match[1].str());
    }
    return default_val;
  };

  auto get_str = [&](const std::string& key, const std::string& default_val = "") -> std::string {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(content, match, pattern)) {
      return match[1].str();
    }
    return default_val;
  };

  auto get_nullable = [&](const std::string& key) -> bool {
    std::regex pattern("\"" + key + R"("\s*:\s*null)");
    return std::regex_search(content, pattern);
  };

  // Read hyperparameters from config.json
  hparams_.n_vocab   = get_int("vocab_size", 151936);
  hparams_.n_layer   = get_int("num_hidden_layers", 28);
  hparams_.n_head    = get_int("num_attention_heads", 16);
  hparams_.n_head_kv = get_int("num_key_value_heads", 8);
  hparams_.n_ctx     = get_int("max_position_embeddings", 40960);
  hparams_.n_embd    = get_int("hidden_size", 1024);
  hparams_.n_ff      = get_int("intermediate_size", 3072);

  int head_dim = get_int("head_dim", 0);
  if (head_dim == 0) {
    head_dim = hparams_.n_embd / hparams_.n_head;
  }
  hparams_.n_embd_head   = head_dim;
  hparams_.n_embd_head_k = head_dim;
  hparams_.n_embd_head_v = head_dim;

  // Derived GQA dimensions
  hparams_.n_embd_gqa   = hparams_.n_head_kv * head_dim;
  hparams_.n_embd_k_gqa = hparams_.n_embd_gqa;
  hparams_.n_embd_v_gqa = hparams_.n_embd_gqa;

  // RoPE parameters
  hparams_.rope_freq_base  = get_float("rope_theta", 1000000.0f);
  hparams_.rms_eps         = get_float("rms_norm_eps", 1e-6f);

  // Handle rope_scaling if present
  if (get_nullable("rope_scaling")) {
    // No rope scaling - use defaults
  }

  // Token IDs
  hparams_.eos_token_id = get_int("eos_token_id", -1);
  hparams_.bos_token_id = get_int("bos_token_id", -1);

  std::cout << std::format("Loaded config: n_layer={}, n_embd={}, n_head={}, " 
                           "n_head_kv={}, n_embd_head={}, n_ctx={}, n_ff={}, "
                           "rope_theta={}\n",
                           hparams_.n_layer, hparams_.n_embd, hparams_.n_head,
                           hparams_.n_head_kv, hparams_.n_embd_head,
                           hparams_.n_ctx, hparams_.n_ff,
                           hparams_.rope_freq_base);
}

}  // namespace models
