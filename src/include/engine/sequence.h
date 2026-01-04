#pragma once

#include <cstddef>
#include <span>
#include <unordered_set>
#include <vector>
namespace engine {
class Sequence {
public:
  int GetNumBlocks() const { return num_blocks_; }

  bool IsEmptyBlockTable() const { return block_table_.empty(); }

  const std::vector<size_t>& GetBlockTable() { return block_table_; }

  void ClearCachedBlocks();

  std::span<size_t> GetTokenIds(int block_id);

  void UpdateCachedTokensNum(size_t num_tokens);

  void AddBlock(size_t block_id) { this->block_table_.push_back(block_id); }

  int SequenceLen() const { return num_tokens_; }

private:
  int num_tokens_;
  int num_blocks_;
  std::vector<size_t> block_table_;
  std::vector<size_t> token_ids_;
  static constexpr int block_size_ = 256;
  size_t num_cached_tokens_{};
};
}  // namespace engine