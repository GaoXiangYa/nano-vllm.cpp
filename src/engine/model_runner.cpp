#include "engine/model_runner.h"

namespace engine {

ModelRunner::ModelRunner(const std::string& model_path,
                         const Config& config) noexcept
    : model_(std::make_unique<models::Qwen3Model>(model_path)),
      config_(config) {
}

}  // namespace engine