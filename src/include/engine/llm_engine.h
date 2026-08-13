#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "chat_template.h"
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

  // Chat: render messages with the model's chat template, then generate
  std::string Chat(const std::vector<chat::ChatMessage>& messages,
                   SamplingParams params = SamplingParams());

  // Chat template family detected from tokenizer_config.json
  chat::ChatTemplateType GetChatTemplateType() const {
    return chat_template_type_;
  }

private:
  Config config_;
  std::unique_ptr<ModelRunner> model_runner_;
  std::unique_ptr<Scheduler> scheduler_;
  std::vector<std::unique_ptr<Sequence>> sequences_;
  chat::ChatTemplateType chat_template_type_ = chat::ChatTemplateType::UNKNOWN;
};

}  // namespace engine
