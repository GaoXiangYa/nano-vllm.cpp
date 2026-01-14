#include "models/qwen3.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include "ggml-alloc.h"
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
    for (size_t i = 2; i < size - 1; i += kTensorCount) {
      size_t layer_idx = i / kTensorCount;
      this->layers_[layer_idx].attn_norm =
          ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, hparams_.n_vocab);

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

      this->tensors_[safetensors_->tensors.keys()[i]] =
          this->layers_[layer_idx].attn_norm;

      this->tensors_[safetensors_->tensors.keys()[i + 1]] =
          this->layers_[layer_idx].ffn_down;
      this->tensors_[safetensors_->tensors.keys()[i + 2]] =
          this->layers_[layer_idx].ffn_gate;
      this->tensors_[safetensors_->tensors.keys()[i + 3]] =
          this->layers_[layer_idx].ffn_up;
      this->tensors_[safetensors_->tensors.keys()[i + 4]] =
          this->layers_[layer_idx].ffn_norm;

      this->tensors_[safetensors_->tensors.keys()[i + 5]] =
          this->layers_[layer_idx].attn_k_norm;
      this->tensors_[safetensors_->tensors.keys()[i + 6]] =
          this->layers_[layer_idx].wk;

      this->tensors_[safetensors_->tensors.keys()[i + 7]] =
          this->layers_[layer_idx].wo;

      this->tensors_[safetensors_->tensors.keys()[i + 8]] =
          this->layers_[layer_idx].attn_q_norm;

      this->tensors_[safetensors_->tensors.keys()[i + 9]] =
          this->layers_[layer_idx].wq;
      this->tensors_[safetensors_->tensors.keys()[i + 10]] =
          this->layers_[layer_idx].wv;
    }
  }

  // allocate the model tensors in a backend buffer
  this->buffer_w_ = ggml_backend_alloc_ctx_tensors(this->ctx_w, this->backend_);

  // allocate key + value memory
  auto* ctx = this->ctx_kv;
  {
    size_t n_tensors = 2;
    ctx = ggml_init({ggml_tensor_overhead() * n_tensors, nullptr, true});
  }

  const auto n_embd = this->hparams_.n_embd;
  const auto n_layer = this->hparams_.n_layer;
  const auto n_ctx = this->hparams_.n_ctx;
  const auto n_mem = n_layer * n_ctx;
  const auto n_elements = n_embd * n_mem;

  this->memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_elements);
  this->memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n_elements);

  this->buffer_kv_ = ggml_backend_alloc_ctx_tensors(ctx, this->backend_);
}
}  // namespace models