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

  void Add(const std::shared_ptr<Sequence>& seq) { this->waiting_.push_back(seq); }

  auto Schedule();

  void Preempt(const Sequence* seq);

  void PostProcess(const std::vector<std::unique_ptr<Sequence>>& seqs,
                   const std::vector<size_t>& token_ids);

private:
  size_t max_num_seqs_;
  size_t max_num_bathced_tokens_;
  size_t eos_;
  std::unique_ptr<BlockManager> block_manager_;
  std::deque<std::shared_ptr<Sequence>> waiting_{};
  std::deque<std::shared_ptr<Sequence>> running_{};
};

}  // namespace engine