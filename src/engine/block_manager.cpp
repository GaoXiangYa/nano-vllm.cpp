#include "engine/block_manager.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include "engine/sequence.h"
#include "xxhash.h"

namespace engine {

void Block::Update(size_t hash, const std::span<size_t>& token_ids) {
  this->hash_ = hash;
  this->token_ids_.resize(token_ids.size());
  size_t size = this->token_ids_.size();
  for (int i = 0; i < size; ++ i) {
    this->token_ids_[i] = token_ids[i];
  }
}

void Block::Reset() {
  ref_count_ = 1;
  hash_ = -1;
  token_ids_.clear();
}

bool Block::IsSameTokenIds(const std::span<size_t>& token_ids) {
  if (token_ids.size() != this->token_ids_.size()) {
    return false;
  }
  size_t size = token_ids.size();
  for (int i = 0; i < size; ++ i) {
    if (token_ids[i] != this->token_ids_[i]) {
      return false;
    }
  }
  return true;
}

BlockManager::BlockManager(int num_blocks, size_t block_size)
    : block_size_(block_size) {
  blocks_.resize(num_blocks);
  for (int i = 0; i < num_blocks; ++i) {
    blocks_[i] = std::make_unique<Block>(i);
    free_block_ids_.insert(i);
  }
}

size_t BlockManager::ComputeHash(const std::span<size_t>& token_ids,
                                 size_t prefix) {
  auto state = XXH64_createState();
  XXH64_reset(state, /*seed=*/0);

  if (prefix != -1) {
    auto prefix_le = prefix;
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    prefix_le = __builtin_bswap64(prefix_le);
#endif
    XXH64_update(state, &prefix_le, sizeof(prefix_le));
  }

  if (!token_ids.empty()) {
    XXH64_update(state, token_ids.data(), token_ids.size() * sizeof(size_t));
  }

  auto result = XXH64_digest(state);
  XXH64_freeState(state);
  return result;
}

Block* BlockManager::AllocateBlock(size_t block_id) {
  auto block = this->blocks_[block_id].get();
  assert(block->GetRefCount() == 0);
  block->Reset();
  assert(this->free_block_ids_.contains(block_id));
  this->free_block_ids_.erase(block_id);
  this->used_block_ids_.insert(block_id);
  return block;
}

Block* BlockManager::GetBlock(size_t block_id) {
  return this->blocks_[block_id].get();
}

void BlockManager::DeallocateBlock(size_t block_id) {
  assert(this->blocks_[block_id]->GetRefCount() == 0);
  assert(this->used_block_ids_.contains(block_id));
  this->used_block_ids_.erase(block_id);
  this->free_block_ids_.insert(block_id);
}

bool BlockManager::CanAllocate(const Sequence* seq) const {
  return this->free_block_ids_.size() >= seq->GetNumBlocks();
}


void BlockManager::Allocate(Sequence* seq) {
  assert(seq->IsEmptyBlockTable());

  size_t prefix_hash = -1;
  bool cache_miss = false;
  const size_t num_blocks = seq->GetNumBlocks();
  Block* block = nullptr;

  for (size_t i = 0; i < num_blocks; ++i) {
    auto token_ids = seq->GetTokenIds(i);
    // Only compute hash for full blocks (matching Python behavior)
    if (token_ids.size() == static_cast<size_t>(this->block_size_)) {
      prefix_hash = this->ComputeHash(token_ids, prefix_hash);
    } else {
      prefix_hash = static_cast<size_t>(-1);
    }
    
    size_t block_id = -1;
    if (this->hash_to_block_id_.contains(prefix_hash)) {
      block_id = this->hash_to_block_id_.at(prefix_hash);
    }
    if (block_id == -1 || !this->blocks_[block_id]->IsSameTokenIds(token_ids)) {
      cache_miss = true;
    }

    if (cache_miss) {
      block_id = *this->free_block_ids_.begin();
      block = this->AllocateBlock(block_id);
    } else {
      seq->UpdateCachedTokensNum(this->block_size_);
      if (this->used_block_ids_.contains(block_id)) {
        block = this->GetBlock(block_id);
        block->AddRefCount();
      } else {
        block = this->AllocateBlock(block_id);
      }
    }

    if (prefix_hash != -1 &&
        token_ids.size() == static_cast<size_t>(this->block_size_)) {
      block->Update(prefix_hash, token_ids);
      this->hash_to_block_id_[prefix_hash] = block_id;
    }

    seq->AddBlock(block_id);
  }
}

void BlockManager::Deallocate(Sequence* seq) {
  for (auto block_id : seq->GetBlockTable()) {
    auto block = this->GetBlock(block_id);
    block->SubRefCount();
    if (block->GetRefCount() == 0) {
      this->DeallocateBlock(block_id);
    }
  }
  seq->ClearCachedBlocks();
}

bool BlockManager::CanAppend(const Sequence* seq) const {
  if (seq->SequenceLen() % this->block_size_ == 1) {
    return !this->free_block_ids_.empty();
  }
  return true;
}

void BlockManager::Append(Sequence* seq) {
  auto last_block_idx = seq->GetBlockTable().size() - 1;
  auto last_block_id = seq->GetBlockTable().back();
  auto last_block = this->GetBlock(last_block_id);
  auto& block_table = seq->GetBlockTable();
  if (seq->SequenceLen() % this->block_size_ == 1) {
    auto block_id = *this->free_block_ids_.begin();
    this->AllocateBlock(block_id);
    seq->AddBlock(block_id);
  } else if (seq->SequenceLen() % this->block_size_ == 0) {
    auto token_ids = seq->GetTokenIds(seq->GetNumBlocks() - 1);
    size_t prefix_hash = -1;
    if (seq->GetBlockTable().size() > 1) {
      prefix_hash = this->blocks_[block_table[last_block_idx - 1]]->GetHash();
    }
    prefix_hash = this->ComputeHash(token_ids, prefix_hash);
    last_block->Update(prefix_hash, token_ids);
    this->hash_to_block_id_[prefix_hash] = last_block->GetBlockId();
  } else {
    // In the middle of a block, hash should be unset
    // assert(last_block->GetHash() == -1);
  }
}

}  // namespace engine