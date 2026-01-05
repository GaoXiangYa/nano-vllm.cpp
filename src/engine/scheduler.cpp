#include "engine/scheduler.h"
#include "config.h"
#include "engine/block_manager.h"

namespace engine {
Scheduler::Scheduler(const Config& config)
    : max_num_seqs_(config.max_num_seqs),
      max_num_bathced_tokens_(config.max_num_bathced_tokens),
      eos_(config.eos) {
  block_manager_ = std::make_unique<BlockManager>(config.num_kvcache_blocks,
                                                  config.kvcache_block_size);
}
}  // namespace engine