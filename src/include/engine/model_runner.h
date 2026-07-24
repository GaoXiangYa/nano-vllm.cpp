#pragma once

#include <memory>
#include "config.h"
#include "models/qwen3.h"
namespace engine {
class ModelRunner {
public:
  explicit ModelRunner(const std::string& model_path, const Config& config) noexcept;

private:
  std::unique_ptr<models::Qwen3Model> model_;
  Config config_;
};

}  // namespace engine