#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "config.h"
#include "engine/sequence.h"
#include "engine/model_runner.h"
#include "engine/scheduler.h"
#include "sampling_params.h"

namespace engine {

class LLMEngine {
public:
  LLMEngine(const std::string& model_path, Config config = Config());
  ~LLMEngine();

  // Add a request (prompt text)
  void AddRequest(const std::string& prompt, SamplingParams params = SamplingParams());

  // Add a request (token IDs)
  void AddRequest(const std::vector<int>& token_ids, SamplingParams params = SamplingParams());

  // Run one scheduler step
  // Returns: pairs of (seq_id, completion_token_ids) for finished sequences
  std::vector<std::pair<int, std::vector<int>>> Step();

  // Check if all work is done
  bool IsFinished() const;

  // Generate text for a list of prompts
  std::vector<std::string> Generate(const std::vector<std::string>& prompts,
                                     SamplingParams params = SamplingParams());

private:
  Config config_;
  std::unique_ptr<ModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  std::vector<std::unique_ptr<Sequence>> sequences_;
};

}  // namespace engine
