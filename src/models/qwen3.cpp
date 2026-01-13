#include "models/qwen3.h"
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include "ggml.h"
#include "safetensors.hh"
#include "tokenizers_cpp.h"
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
  auto ret = safetensors::load_from_file(safetensors_path.c_str(),
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

  
}
}  // namespace models