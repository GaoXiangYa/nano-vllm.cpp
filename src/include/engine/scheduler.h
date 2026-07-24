#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include "../config.h"
#include "engine/block_manager.h"
#include "engine/sequence.h"

namespace engine {

class Scheduler {
public:
  explicit Scheduler(const Config& config);

  bool IsFinished() const {
    return this->waiting_.empty() && this->running_.empty();
  }

  void Add(Sequence* seq) { this->waiting_.push_back(seq); }

  using ScheduleResult = std::pair<std::vector<Sequence*>, bool>;
  ScheduleResult Schedule();

  void Preempt(Sequence* seq);

  void PostProcess(const std::vector<Sequence*>& seqs,
                   const std::vector<size_t>& token_ids);

private:
  size_t max_num_seqs_;
  size_t max_num_bathced_tokens_;
  size_t eos_;
  std::unique_ptr<BlockManager> block_manager_;
  std::deque<Sequence*> waiting_{};
  std::deque<Sequence*> running_{};
};

}  // namespace engine