#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sequence.h"

namespace engine {

class Block {
public:
  explicit Block(size_t block_id)
      : block_id_(block_id), ref_count_(0), hash_(-1), token_ids_(0) {}

  void Update(size_t hash, const std::span<size_t>& token_ids);

  void Reset();

  size_t GetRefCount() const { return ref_count_; }

  void AddRefCount() { ++ref_count_; }

  void SubRefCount() { --ref_count_; }

  std::span<size_t> GetTokenIds() {
    return std::span(token_ids_.begin(), token_ids_.end());
  }

  bool IsSameTokenIds(const std::span<size_t>& token_ids);

  size_t GetHash() const { return hash_; }

  size_t GetBlockId() const { return block_id_; }

private:
  size_t block_id_;   // Unique identifier for the block
  size_t ref_count_;  // Reference count for this block
  size_t hash_;          // Hash of the token content (for prefix caching)
  std::vector<size_t> token_ids_;  // Token IDs stored in this block
};

class BlockManager {
public:
  BlockManager(int num_blocks, size_t block_size);

  bool CanAllocate(const Sequence* seq) const;

  void Allocate(Sequence* seq);

  void Deallocate(Sequence* seq);

  bool CanAppend(const Sequence* seq) const;

  void Append(Sequence* seq);

private:
  size_t ComputeHash(const std::span<size_t>& token_ids, size_t prefix = 0);

  Block* AllocateBlock(size_t block_id);

  Block* GetBlock(size_t block_id);

  void DeallocateBlock(size_t block_id);

private:
  size_t block_size_;  // Size of each block in bytes
  std::vector<std::unique_ptr<Block>> blocks_;
  std::unordered_map<size_t, size_t>
      hash_to_block_id_{};                    // Map from hash to block ID
  std::unordered_set<size_t> free_block_ids_{};  // Set of free block IDs
  std::unordered_set<size_t> used_block_ids_{};  // Set of used block IDs
};

}  // namespace engine