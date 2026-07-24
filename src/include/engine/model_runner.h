#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "config.h"
#include "engine/sequence.h"
#include "models/qwen3.h"

namespace engine {

class ModelRunner {
public:
  explicit ModelRunner(const std::string& model_path, Config& config);

  // Run one step for a batch of sequences
  // For prefill: processes uncached tokens of each sequence
  // For decode: processes one token per sequence (with full context)
  std::vector<int> Run(const std::vector<Sequence*>& seqs, bool is_prefill);

  const std::unique_ptr<tokenizers::Tokenizer>& GetTokenizer() const {
    return model_->GetTokenizer();
  }

  int GetEosToken() const { return model_->GetEosTokenId(); }

  // Get model context length (for computing KV cache block count)
  int GetMaxContextLen() const;

private:
  // Prepare input IDs and positions for a batch
  void PrepareInputs(const std::vector<Sequence*>& seqs, bool is_prefill,
                     std::vector<int>& input_ids,
                     std::vector<int>& positions,
                     std::vector<int>& output_indices,
                     std::vector<int>& seq_boundaries);

  // Build and fill the KQ mask tensor (block-diagonal causal for batched)
  void FillKQMask(ggml_tensor* mask, int n_kv, int n_tokens,
                  const std::vector<int>& seq_boundaries);

  // Sampling: temperature + softmax + multinomial
  std::vector<int> Sample(const float* logits_data, int n_vocab, int n_seqs,
                          const std::vector<float>& temperatures);

private:
  Config& config_;
  std::unique_ptr<models::Qwen3Model> model_;
};

}  // namespace engine
