#include "engine/sequence.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace engine {

Sequence::Sequence(const std::vector<size_t>& token_ids,
                   SamplingParams sampling_params)
    : seq_id_(counter_++),
      status_(SequenceStatus::WATTING),
      last_token_(token_ids.back()),
      num_tokens_(token_ids.size()),
      token_ids_(token_ids),
      num_prompt_tokens(token_ids.size()),
      temperature_(sampling_params.temperature),
      max_tokens_(sampling_params.max_tokens),
      ignore_eos_(sampling_params.ignore_eos) {
}

std::span<size_t> Sequence::GetTokenIds(size_t block_id) {
  assert(block_id >= 0 && block_id < this->num_blocks_);
  return std::span(
      this->token_ids_.begin() + static_cast<int64_t>(block_id * block_size_),
      this->token_ids_.begin() +
          static_cast<int64_t>((block_id + 1) * this->block_size_));
}

void Sequence::UpdateCachedTokensNum(size_t num_tokens) {
  this->num_cached_tokens_ += num_tokens;
}

void Sequence::ClearCachedBlocks() {
  this->num_cached_tokens_ = 0;
  this->block_table_.clear();
}

void Sequence::AppendToken(size_t token_id) {
  this->token_ids_.push_back(token_id);
  this->last_token_ = token_id;
  this->num_tokens_++;
}

}  // namespace engine