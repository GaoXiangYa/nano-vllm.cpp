#include "engine/scheduler.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "config.h"
#include "engine/block_manager.h"
#include "engine/sequence.h"

namespace engine {
Scheduler::Scheduler(const Config& config)
    : max_num_seqs_(config.max_num_seqs),
      max_num_bathced_tokens_(config.max_num_bathced_tokens),
      eos_(config.eos) {
  block_manager_ = std::make_unique<BlockManager>(config.num_kvcache_blocks,
                                                  config.kvcache_block_size);
}

auto Scheduler::Schedule() {
  // prefill
  std::vector<std::shared_ptr<Sequence>> scheduled_seqs;
  size_t num_seqs = 0;
  size_t num_batched_tokens = 0;

  while (!this->waiting_.empty() && num_seqs < this->max_num_seqs_) {
    auto& seq = this->waiting_.front();
    if (num_batched_tokens + seq->SequenceLen() >
            this->max_num_bathced_tokens_ ||
        !this->block_manager_->CanAllocate(seq.get())) {
      break;
    }
    ++num_seqs;
    this->block_manager_->Allocate(seq.get());
    num_batched_tokens += seq->SequenceLen() - seq->GetNumCachedTokens();
    seq->SetStatus(SequenceStatus::RUNNING);
    seq = std::move(this->waiting_.front());
    this->waiting_.pop_front();
    this->running_.push_back(seq);
    scheduled_seqs.push_back(seq);
  }
  if (!scheduled_seqs.empty()) {
    return std::make_pair(scheduled_seqs, true);
  }

  // decode
  while (!this->running_.empty() && num_seqs < this->max_num_seqs_) {
    auto seq = std::move(this->running_.front());
    this->running_.pop_front();
    while (!this->block_manager_->CanAppend(seq.get())) {
      if (!this->running_.empty()) {
        this->Preempt(this->running_.back().get());
        this->running_.pop_back();
      } else {
        this->Preempt(seq.get());
        seq = nullptr;
        break;
      }
    }
    if (seq) {
      ++num_seqs;
      this->block_manager_->Append(seq.get());
      scheduled_seqs.push_back(seq);
    }
  }

  assert(!scheduled_seqs.empty());
  std::for_each(scheduled_seqs.rbegin(), scheduled_seqs.rend(), [&](const auto &val) {
    this->running_.push_front(val);
  });
  return std::make_pair(scheduled_seqs, false);
}
}  // namespace engine