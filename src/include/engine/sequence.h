#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>
#include "../sampling_params.h"
namespace engine {

enum class SequenceStatus { WAITING, RUNNING, FINISHED };

class Sequence {
public:
  explicit Sequence(const std::vector<size_t>& token_ids,
                    SamplingParams sampling_params = SamplingParams());

  size_t GetNumBlocks() const {
    return (this->num_tokens_ + this->block_size_ - 1) / this->block_size_;
  }

  bool IsEmptyBlockTable() const { return block_table_.empty(); }

  const std::vector<size_t>& GetBlockTable() { return block_table_; }

  void ClearCachedBlocks();

  std::span<size_t> GetTokenIds(size_t block_id);

  void UpdateCachedTokensNum(size_t num_tokens);

  void AddBlock(size_t block_id) { this->block_table_.push_back(block_id); }

  size_t SequenceLen() const { return num_tokens_; }

  void AppendToken(size_t token_id);

  bool IsFinished() const { return this->status_ == SequenceStatus::FINISHED; }

  int GetSeqId() const { return seq_id_; }

  float GetTemperature() const { return temperature_; }

  float GetRepetitionPenalty() const { return repetition_penalty_; }

  size_t GetNumCompletionTokens() const {
    return this->num_tokens_ - this->num_prompt_tokens;
  }

  std::span<size_t> GetPromptTokenIds() {
    return std::span(this->token_ids_.begin(),
                     this->token_ids_.begin() +
                         static_cast<int64_t>(this->num_prompt_tokens));
  }

  std::span<size_t> GetCompletionTokenIds() {
    return std::span(this->token_ids_.begin() +
                         static_cast<int64_t>(this->num_prompt_tokens),
                     this->token_ids_.end());
  }

  size_t GetNumCachedBlocks() const {
    return (this->num_cached_tokens_ + this->block_size_ - 1) /
           this->block_size_;
  }

  size_t GetLastBlockNumTokens() const {
    return this->num_tokens_ - (this->GetNumBlocks() - 1) * this->block_size_;
  }

  size_t GetNumCachedTokens() const { return num_cached_tokens_; }

  void SetStatus(const SequenceStatus& status) { this->status_ = status; }

  bool IsIgnoreEos() const { return ignore_eos_; }

  size_t GetMaxTokens() const { return max_tokens_; }

private:
  int seq_id_;
  SequenceStatus status_;
  size_t last_token_;
  size_t num_tokens_;
  std::vector<size_t> block_table_;
  std::vector<size_t> token_ids_;
  static constexpr size_t block_size_ = 256;
  size_t num_cached_tokens_{0};
  size_t num_prompt_tokens;
  static inline std::atomic<int> counter_{0};
  float temperature_;
  float repetition_penalty_;
  size_t max_tokens_;
  bool ignore_eos_;
};
}  // namespace engine