#include "engine/sequence.h"
#include <cassert>
#include <cstddef>
#include <span>

namespace engine {

std::span<size_t> Sequence::GetTokenIds(int block_id) {
  assert(block_id >= 0 && block_id < this->num_blocks_);
  return std::span(this->token_ids_.begin() + block_id * this->block_size_,
                   this->token_ids_.begin() +
                       (block_id + 1) * this->block_size_);
}

void Sequence::UpdateCachedTokensNum(size_t num_tokens) {
  this->num_cached_tokens_ += num_tokens;
}

void Sequence::ClearCachedBlocks() {
  this->num_cached_tokens_ = 0;
  this->block_table_.clear();
}

}  // namespace engine