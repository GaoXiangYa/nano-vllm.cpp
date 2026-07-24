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

Scheduler::ScheduleResult Scheduler::Schedule() {
  // prefill
  std::vector<Sequence*> scheduled_seqs;
  size_t num_seqs = 0;
  size_t num_batched_tokens = 0;

  while (!this->waiting_.empty() && num_seqs < this->max_num_seqs_) {
    auto seq = this->waiting_.front();
    if (num_batched_tokens + seq->SequenceLen() >
            this->max_num_bathced_tokens_ ||
        !this->block_manager_->CanAllocate(seq)) {
      break;
    }
    ++num_seqs;
    this->block_manager_->Allocate(seq);
    num_batched_tokens += seq->SequenceLen() - seq->GetNumCachedTokens();
    seq->SetStatus(SequenceStatus::RUNNING);
    this->waiting_.pop_front();
    this->running_.push_back(seq);
    scheduled_seqs.push_back(seq);
  }
  if (!scheduled_seqs.empty()) {
    return std::make_pair(scheduled_seqs, true);
  }

  // decode
  while (!this->running_.empty() && num_seqs < this->max_num_seqs_) {
    auto seq = this->running_.front();
    this->running_.pop_front();
    while (!this->block_manager_->CanAppend(seq)) {
      if (!this->running_.empty()) {
        this->Preempt(this->running_.back());
        this->running_.pop_back();
      } else {
        this->Preempt(seq);
        seq = nullptr;
        break;
      }
    }
    if (seq) {
      ++num_seqs;
      this->block_manager_->Append(seq);
      scheduled_seqs.push_back(seq);
    }
  }

  assert(!scheduled_seqs.empty());
  std::for_each(scheduled_seqs.rbegin(), scheduled_seqs.rend(),
                [&](const auto& val) { this->running_.push_front(val); });
  return std::make_pair(scheduled_seqs, false);
}

void Scheduler::Preempt(Sequence* seq) {
  seq->SetStatus(SequenceStatus::WAITING);
  this->block_manager_->Deallocate(seq);
  this->waiting_.push_front(seq);
}

void Scheduler::PostProcess(const std::vector<Sequence*>& seqs,
                            const std::vector<size_t>& token_ids) {
  size_t size = std::min(token_ids.size(), seqs.size());
  for (size_t i = 0; i < size; ++i) {
    seqs[i]->AppendToken(token_ids[i]);
    if ((!seqs[i]->IsIgnoreEos() && token_ids[i] == this->eos_) ||
        seqs[i]->GetNumCompletionTokens() == seqs[i]->GetMaxTokens()) {
      seqs[i]->SetStatus(SequenceStatus::FINISHED);
      this->block_manager_->Deallocate(seqs[i]);
      if (auto ite =
              std::find(this->running_.begin(), this->running_.end(), seqs[i]);
          ite != this->running_.end()) {
        this->running_.erase(ite);
      }
    }
  }
}
}  // namespace engine