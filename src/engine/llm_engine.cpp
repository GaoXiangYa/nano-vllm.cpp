#include "engine/llm_engine.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>
#include "chat_template.h"
#include "config.h"
#include "engine/model_runner.h"
#include "engine/scheduler.h"
#include "engine/sequence.h"
#include "sampling_params.h"
#include "tokenizers_cpp.h"

namespace engine {

LLMEngine::LLMEngine(const std::string& model_path, Config config)
    : config_(std::move(config)) {
  model_runner_ = std::make_unique<ModelRunner>(model_path, config_);

  // Read EOS token from model config
  int eos_id = model_runner_->GetEosToken();
  if (eos_id >= 0) {
    config_.eos = eos_id;
  }

  // Detect chat template from tokenizer_config.json
  {
    std::ifstream f(std::filesystem::path(model_path) / "tokenizer_config.json");
    if (f.is_open()) {
      std::string content((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
      std::smatch m;
      if (std::regex_search(content, m,
                            std::regex("\"chat_template\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\""))) {
        std::string tmpl = m[1].str();
        chat_template_type_ = chat::DetectChatTemplate(tmpl);
      }
    }
  }

  // Compute KV cache block count from model's context length
  int n_ctx = model_runner_->GetMaxContextLen();
  config_.num_kvcache_blocks = n_ctx / config_.kvcache_block_size;

  scheduler_ = std::make_unique<Scheduler>(config_);
}

LLMEngine::~LLMEngine() = default;

void LLMEngine::AddRequest(const std::string& prompt, SamplingParams params) {
  auto token_ids = model_runner_->GetTokenizer()->Encode(prompt);
  std::vector<size_t> ids(token_ids.begin(), token_ids.end());
  auto seq = std::make_unique<Sequence>(ids, params);
  scheduler_->Add(seq.get());
  sequences_.push_back(std::move(seq));
}

void LLMEngine::AddRequest(const std::vector<int>& token_ids,
                            SamplingParams params) {
  std::vector<size_t> ids(token_ids.begin(), token_ids.end());
  auto seq = std::make_unique<Sequence>(ids, params);
  scheduler_->Add(seq.get());
  sequences_.push_back(std::move(seq));
}

std::vector<std::pair<int, std::vector<int>>> LLMEngine::Step() {
  auto [seqs, is_prefill] = scheduler_->Schedule();
  if (seqs.empty()) {
    return {};
  }

  auto token_ids = model_runner_->Run(seqs, is_prefill);

  std::vector<size_t> size_t_ids(token_ids.begin(), token_ids.end());
  scheduler_->PostProcess(seqs, size_t_ids);

  // Collect finished sequences
  std::vector<std::pair<int, std::vector<int>>> outputs;
  for (auto* seq : seqs) {
    if (seq->IsFinished()) {
      auto completion = seq->GetCompletionTokenIds();
      std::vector<int> comp_ids(completion.begin(), completion.end());
      outputs.emplace_back(seq->GetSeqId(), std::move(comp_ids));
    }
  }

  return outputs;
}

bool LLMEngine::IsFinished() const {
  return scheduler_->IsFinished();
}

std::vector<std::string> LLMEngine::Generate(
    const std::vector<std::string>& prompts,
    SamplingParams params) {
  for (const auto& prompt : prompts) {
    AddRequest(prompt, params);
  }

  std::unordered_map<int, std::vector<int>> results;

  while (!IsFinished()) {
    auto step_outputs = Step();
    for (auto& [seq_id, token_ids] : step_outputs) {
      results[seq_id] = std::move(token_ids);
    }
    if (!step_outputs.empty()) {
      std::cout << "." << std::flush;
    }
  }
  std::cout << "\n";

  // Decode results
  std::vector<std::string> decoded;
  for (const auto& prompt : prompts) {
    // Find the result for each prompt by order
    // (Sequences are processed in order of addition)
    (void)prompt;  // for now
  }

  // Collect results in order
  for (auto& seq : sequences_) {
    if (seq->IsFinished()) {
      auto it = results.find(seq->GetSeqId());
      if (it != results.end()) {
        decoded.push_back(
            model_runner_->GetTokenizer()->Decode(it->second));
      }
    }
  }

  return decoded;
}

std::string LLMEngine::Chat(const std::vector<chat::ChatMessage>& messages,
                            SamplingParams params) {
  if (chat_template_type_ == chat::ChatTemplateType::UNKNOWN) {
    throw std::runtime_error(
        "No supported chat template found in tokenizer_config.json");
  }
  std::string prompt =
      chat::ApplyChatTemplate(chat_template_type_, messages, true);
  auto outputs = Generate({prompt}, params);
  return outputs.empty() ? std::string() : outputs[0];
}

}  // namespace engine
