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
  // Prefill: processes uncached tokens, stores K/V into the KV cache.
  // Decode:  processes one new token per sequence, reads history K/V from the
  //          cache, appends new K/V.
  std::vector<int> Run(const std::vector<Sequence*>& seqs, bool is_prefill);

  const std::unique_ptr<tokenizers::Tokenizer>& GetTokenizer() const {
    return model_->GetTokenizer();
  }

  int GetEosToken() const { return model_->GetEosTokenId(); }

  // Get model context length (for computing KV cache block count)
  int GetMaxContextLen() const;

private:
  void PrepareInputs(const std::vector<Sequence*>& seqs, bool is_prefill,
                     std::vector<int>& input_ids,
                     std::vector<int>& positions,
                     std::vector<int>& output_indices,
                     std::vector<int>& seq_boundaries,
                     std::vector<int>& seq_starts,
                     std::vector<int>& token_offsets);

  // Block-diagonal causal mask for batched prefill:
  // within a sequence key_pos <= query_pos, and cross-sequence keys are masked.
  void FillCausalMask(ggml_tensor* mask, int n_tokens,
                      const std::vector<int>& seq_boundaries);

  // Varlen decode mask: seq i can attend to its own history + its new token
  void FillVarlenMask(ggml_tensor* mask, int n_kv, int n_seqs,
                      const std::vector<int>& ctx_lens);

  // After prefill graph compute: extract per-layer K/V and write to cache
  void StorePrefillKV(const std::vector<Sequence*>& seqs,
                      const std::vector<int>& token_offsets,
                      const std::vector<int>& seq_starts,
                      int n_tokens);

  // Before decode graph compute: load history K/V into k_hist_in/v_hist_in
  void LoadHistoryKV(const std::vector<Sequence*>& seqs,
                     const std::vector<int>& ctx_lens, int total_hist);

  // After decode graph compute: append new K/V to cache
  void AppendDecodeKV(const std::vector<Sequence*>& seqs);

  std::vector<int> Sample(const float* logits_data, int n_vocab, int n_seqs,
                          const std::vector<float>& temperatures,
                          const std::vector<Sequence*>& seqs);

private:
  Config config_;
  std::unique_ptr<models::Qwen3Model> model_;

};

}  // namespace engine
