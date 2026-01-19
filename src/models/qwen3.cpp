#include "models/qwen3.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <source_location>
#include <stdexcept>
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

Qwen3Model::Qwen3Model(const std::string& model_path) {
  // Load model from the specified path
  // (Implementation omitted for brevity)
  std::filesystem::path path(model_path);
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Model path does not exist: " + model_path);
  }

  std::filesystem::path tokenizer_path = path / "tokenizer.json";
  std::filesystem::path safetensors_path = path / "model.safetensors";

  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_path.c_str());
  std::string warn, err;
  auto ret = safetensors::mmap_from_file(safetensors_path.c_str(),
                                         safetensors_.get(), &warn, &err);

  if (warn.size()) {
    std::cout << std::format("WARN: {}\n", warn);
  }

  if (!ret) {
    throw std::runtime_error(
        "Faile to load: " + safetensors_path.generic_string() + " ERR: " + err);
  }

  const int32_t n_tensors = 1 + 2 + 11 * hparams_.n_layer;
  const int32_t max_tensors = n_tensors + 1 + 2 * hparams_.n_layer;
  const auto ctx_size = ggml_tensor_overhead() * max_tensors;
  constexpr float kMB = 1024.0 * 1024.0;

  // create model context
  {
    struct ggml_init_params params = {
        ctx_size,
        nullptr,
        true,
    };

    this->ctx_w = ggml_init(params);
    if (this->ctx_w == nullptr) {
      std::cerr << std::format("{} ggml_init() failed!\n",
                               std::source_location::current().function_name());
    }
  }

#ifdef USE_CUDA
  this->backend_ = ggml_backend_cuda_init(0);
#endif
  if (this->backend_ == nullptr) {
    std::cerr << std::format("{}: using CPU backend!\n",
                             std::source_location::current().function_name());
    this->backend_ = ggml_backend_cpu_init();
  }
  if (this->backend_ == nullptr) {
    throw std::runtime_error("Failed to initialize any ggml backend!");
  }
  // create tensor for the model
  {
    auto size = safetensors_->tensors.size();
    auto n_vocab = this->hparams_.n_vocab;
    auto n_embd = this->hparams_.n_embd;
    auto n_ff = this->hparams_.n_ff;
    auto n_layer = this->hparams_.n_layer;
    auto n_embd_head_k = this->hparams_.n_embd_head_k;
    auto n_embd_gqa = this->hparams_.n_embd_k_gqa;
    auto n_head = this->hparams_.n_head;
    const int kTensorCount = 11;
    auto& ctx = this->ctx_w;
    this->layers_.resize(hparams_.n_layer);

    this->output_ = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_vocab);
    this->tensors_[safetensors_->tensors.keys()[0]] = this->output_;
    this->tok_embd_ = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_vocab);
    this->tensors_[safetensors_->tensors.keys()[1]] = this->tok_embd_;

    for (size_t i = 2; i < size - 1; i += kTensorCount) {
      size_t layer_idx = i / kTensorCount;
      this->layers_[layer_idx].attn_norm =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_vocab);

      this->layers_[layer_idx].ffn_down =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_ff, n_embd);
      this->layers_[layer_idx].ffn_gate =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_ff);
      this->layers_[layer_idx].ffn_up =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_ff);
      this->layers_[layer_idx].ffn_norm =
          ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_embd);

      this->layers_[layer_idx].attn_k_norm =
          ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_embd_head_k);
      this->layers_[layer_idx].wk =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_embd_gqa);

      this->layers_[layer_idx].wo =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd_head_k * n_head,
                             n_embd);

      this->layers_[layer_idx].attn_q_norm =
          ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_embd_head_k);

      this->layers_[layer_idx].wq =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd,
                             n_embd_head_k * n_head);
      this->layers_[layer_idx].wv =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, n_embd_gqa);

      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kAttnNorm]] =
          this->layers_[layer_idx].attn_norm;

      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kFfnDown]] =
          this->layers_[layer_idx].ffn_down;
      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kFfnGate]] =
          this->layers_[layer_idx].ffn_gate;
      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kFfnUp]] =
          this->layers_[layer_idx].ffn_up;
      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kFfnNorm]] =
          this->layers_[layer_idx].ffn_norm;

      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kAttnKNorm]] =
          this->layers_[layer_idx].attn_k_norm;
      this->tensors_[safetensors_->tensors.keys()[i + TensorKeyOffset::kWk]] =
          this->layers_[layer_idx].wk;

      this->tensors_[safetensors_->tensors.keys()[i + TensorKeyOffset::kWo]] =
          this->layers_[layer_idx].wo;

      this->tensors_[safetensors_->tensors
                         .keys()[i + TensorKeyOffset::kAttnQNorm]] =
          this->layers_[layer_idx].attn_q_norm;

      this->tensors_[safetensors_->tensors.keys()[i + TensorKeyOffset::kWq]] =
          this->layers_[layer_idx].wq;
      this->tensors_[safetensors_->tensors.keys()[i + TensorKeyOffset::kWv]] =
          this->layers_[layer_idx].wv;
    }
    this->output_norm_ = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_embd);
    this->tensors_[safetensors_->tensors.keys()[size - 1]] = this->output_norm_;
  }

  // allocate the model tensors in a backend buffer
  this->buffer_w_ = ggml_backend_alloc_ctx_tensors(this->ctx_w, this->backend_);

  // allocate key + value memory
  auto* ctx = this->ctx_kv;
  {
    size_t n_tensors = 2;
    ctx = ggml_init({ggml_tensor_overhead() * n_tensors, nullptr, true});

    const auto n_embd = this->hparams_.n_embd;
    const auto n_layer = this->hparams_.n_layer;
    const auto n_ctx = this->hparams_.n_ctx;
    const auto n_mem = n_layer * n_ctx;
    const auto n_elements = n_embd * n_mem;

    this->memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_elements);
    this->memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_elements);

    this->buffer_kv_ = ggml_backend_alloc_ctx_tensors(ctx, this->backend_);
    const size_t memory_size = ggml_backend_buffer_get_size(this->buffer_kv_);
    std::cout << std::format("{}: memory size = {:8.2f} MB, n_mem = {}\n",
                             std::source_location::current().function_name(),
                             static_cast<float>(memory_size) / kMB, n_mem);
  }

  // load weights data
  {
    size_t total_size = 0;
    std::span<std::byte> databuffer{
        safetensors_->mmaped
            ? std::bit_cast<std::byte*>(safetensors_->databuffer_addr)
            : std::bit_cast<std::byte*>(safetensors_->storage.data()),
        safetensors_->databuffer_size};

    auto safetensor_size = safetensors_->tensors.size();
    for (size_t i = 0; i < safetensor_size; ++i) {
      const auto& tensor_name = safetensors_->tensors.keys()[i];
      if (!this->tensors_.contains(tensor_name)) {
        throw std::runtime_error(
            "Unknown tensor name: " + tensor_name +
            std::source_location::current().function_name() + "\n");
      }

      auto tensor = this->tensors_[tensor_name];
      safetensors::tensor_t stensor;
      safetensors_->tensors.at(tensor_name, &stensor);

      auto nitems = safetensors::get_shape_size(stensor);
      auto item_bytes = safetensors::get_dtype_bytes(stensor.dtype);
      auto tensor_data = databuffer.data() + stensor.data_offsets[0];

      if (ggml_backend_buffer_is_host(this->buffer_w_)) {
        std::memcpy(tensor->data, tensor_data, ggml_nbytes(tensor));
      } else {
        ggml_backend_tensor_set(tensor, tensor_data, 0, ggml_nbytes(tensor));
      }

      total_size += ggml_nbytes(tensor);
    }

    std::cout << std::format("{}: loaded model size = {:8.2f} MB\n",
                             std::source_location::current().function_name(),
                             static_cast<float>(total_size) / kMB);
  }
}

Qwen3Model::~Qwen3Model() {
  ggml_free(this->ctx_w);
  ggml_free(this->ctx_kv);

  ggml_gallocr_free(this->allocr_);
  ggml_backend_buffer_free(this->buffer_w_);
  ggml_backend_buffer_free(this->buffer_kv_);
  ggml_backend_free(this->backend_);
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
  ctx_compute = ggml_init(params);
  this->compute_graph_ =
      ggml_new_graph_custom(ctx_compute, kQwen3MaxNodes, false);
}

ggml_tensor* Qwen3Model::BuildInputEmbedding(ggml_tensor* tok_emd,
                                             const int n_tokens) {
  auto embd = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I32, n_tokens);
  ggml_set_input(embd);
  auto cur = ggml_get_rows(this->ctx_compute, tok_emd, embd);
  return cur;
}

ggml_tensor* Qwen3Model::BuildInputPosition() {
  const int n_ctx = this->hparams_.n_ctx;
  auto pos = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I32, n_ctx);
  ggml_set_input(pos);
  return pos;
}

ggml_tensor* Qwen3Model::BuildInputIds(const int n_output) {
  auto ids = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I32, n_output);
  ggml_set_input(ids);
  return ids;
}

ggml_tensor* Qwen3Model::BuildNorm(ggml_tensor* input, ggml_tensor* norm_weight,
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
    input = ggml_mul(this->ctx_compute, input, norm_weight);
  }
  if (norm_bias) {
    input = ggml_add(this->ctx_compute, input, norm_bias);
  }

  return input;
}

ggml_tensor* Qwen3Model::BuildFFN(ggml_tensor* input, ggml_tensor* ffn_up,
                                  ggml_tensor* ffn_gate,
                                  ggml_tensor* ffn_down) {
  auto tmp = ggml_mul_mat(this->ctx_compute, ffn_up, input);
  input = ggml_mul_mat(this->ctx_compute, ffn_gate, input);
  input = ggml_swiglu_split(this->ctx_compute, input, tmp);
  input = ggml_mul(this->ctx_compute, input, tmp);
  input = ggml_mul_mat(this->ctx_compute, ffn_down, input);
  return input;
}

ggml_tensor* Qwen3Model::BuildAttentionKV(const int n_tokens) {
  this->memory_k =
      ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I64, n_tokens);
  ggml_set_input(this->memory_k);
  this->memory_v =
      ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_I64, n_tokens);
  ggml_set_input(this->memory_v);

  // TODO: get kv cache size from config
  const auto n_kv = 256;
  this->memory_kq_mask = ggml_new_tensor_4d(this->ctx_compute, GGML_TYPE_F32,
                                            n_kv, n_tokens, 1, 1);
  ggml_set_input(this->memory_kq_mask);
  this->memory_kq_mask_cnv =
      ggml_cast(this->ctx_compute, this->memory_kq_mask, GGML_TYPE_F16);
  return this->memory_kq_mask_cnv;
}

ggml_tensor* Qwen3Model::BuildAttention(ggml_tensor* input, ggml_tensor* q_cur,
                                        ggml_tensor* k_cur,
                                        ggml_tensor* v_cur) {
  ggml_build_forward_expand(this->compute_graph_, q_cur);
  ggml_build_forward_expand(this->compute_graph_, k_cur);
  ggml_build_forward_expand(this->compute_graph_, v_cur);

  // store to kv cache memory
  {
    ggml_build_forward_expand(this->compute_graph_, ggml_cpy(this->ctx_compute, k_cur, this->memory_k));
    ggml_build_forward_expand(this->compute_graph_, ggml_cpy(this->ctx_compute, v_cur, this->memory_v));
  }

  return nullptr;
}

void Qwen3Model::BuildGraph(const int n_past, const int n_tokens) {
  Reset();
}

}  // namespace models