#include "models/qwen3.h"
#include <filesystem>
#include "safetensors.h"

namespace models {

Qwen3Model::Qwen3Model(const std::string& model_path) {
  // Load model from the specified path
  // (Implementation omitted for brevity)
  constexpr std::string_view tokenizer_file = "tokenizer.json";
  constexpr std::string_view safetensors_file = "model.safetensors";
  std::filesystem::path path(model_path);
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Model path does not exist: " + model_path);
  }

  

}
}  // namespace models