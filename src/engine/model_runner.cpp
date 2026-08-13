#include "engine/model_runner.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include "engine/sequence.h"
#include "ggml.h"
#include "models/qwen3.h"

namespace engine {

ModelRunner::ModelRunner(const std::string& model_path, Config& config)
    : config_(config),
      model_(std::make_unique<models::Qwen3Model>(model_path)) {
}

int ModelRunner::GetMaxContextLen() const {
  return model_->GetHparams().n_ctx;
}

std::vector<int> ModelRunner::Run(const std::vector<Sequence*>& seqs,
                                  bool is_prefill) {
  if (seqs.empty())
    return {};

  std::vector<int> input_ids;
  std::vector<int> positions;
  std::vector<int> output_indices;
  std::vector<int> seq_boundaries;
  PrepareInputs(seqs, is_prefill, input_ids, positions, output_indices,
                seq_boundaries);

  const int n_tokens = static_cast<int>(input_ids.size());
  const int n_outputs = static_cast<int>(output_indices.size());
  model_->BuildPrefillGraph(n_tokens, 0, n_outputs);

  // Allocate graph buffers BEFORE setting input tensor data
  if (!model_->AllocateGraph()) {
    std::cerr << "Failed to allocate compute graph\n";
    return {};
  }

  // Set input tensors
  auto* graph = model_->GetComputeGraph();
  auto* inp_tok = ggml_graph_get_tensor(graph, "inp_tokens");
  auto* inp_pos = ggml_graph_get_tensor(graph, "inp_pos");
  auto* inp_mask = ggml_graph_get_tensor(graph, "inp_kq_mask");
  auto* inp_out = ggml_graph_get_tensor(graph, "inp_out_ids");

  model_->SetInputTensor(inp_tok, input_ids.data(),
                         n_tokens * static_cast<int>(sizeof(int)));
  model_->SetInputTensor(inp_pos, positions.data(),
                         n_tokens * static_cast<int>(sizeof(int)));
  model_->SetInputTensor(inp_out, output_indices.data(),
                         static_cast<int>(output_indices.size() * sizeof(int)));

  // Fill KQ mask (block-diagonal causal for batched sequences)
  FillKQMask(inp_mask, n_tokens, n_tokens, seq_boundaries);

  auto status = model_->GraphCompute();
  if (status != GGML_STATUS_SUCCESS) {
    return {};
  }

  // Extract logits
  auto* logits_tensor = model_->GetLogits();
  auto& hparams = model_->GetHparams();
  const int n_vocab = hparams.n_vocab;
  const int n_out = static_cast<int>(output_indices.size());

  std::vector<float> logits_data(static_cast<size_t>(n_vocab) *
                                 static_cast<size_t>(n_out));
  ggml_backend_tensor_get(logits_tensor, logits_data.data(), 0,
                          logits_data.size() * sizeof(float));

  // Sample
  std::vector<float> temperatures;
  temperatures.reserve(seqs.size());
  for (auto* seq : seqs) {
    temperatures.push_back(seq->GetTemperature());
  }

  return Sample(logits_data.data(), n_vocab, n_out, temperatures, seqs);
}

void ModelRunner::PrepareInputs(const std::vector<Sequence*>& seqs,
                                bool is_prefill, std::vector<int>& input_ids,
                                std::vector<int>& positions,
                                std::vector<int>& output_indices,
                                std::vector<int>& seq_boundaries) {
  input_ids.clear();
  positions.clear();
  output_indices.clear();
  seq_boundaries.clear();

  int token_offset = 0;
  for (auto* seq : seqs) {
    size_t seq_len = seq->SequenceLen();
    size_t start_pos = 0;
    int seq_tokens = 0;

    if (is_prefill) {
      // Process all uncached tokens
      start_pos = seq->GetNumCachedTokens();
      seq_tokens = static_cast<int>(seq_len - start_pos);
    } else {
      // Process all tokens (full context including generated)
      start_pos = 0;
      seq_tokens = static_cast<int>(seq_len);
    }

      for (size_t i = start_pos; i < seq_len; ++i) {
        auto prompt = seq->GetPromptTokenIds();
        auto completion = seq->GetCompletionTokenIds();
        if (i < prompt.size()) {
          input_ids.push_back(static_cast<int>(prompt[i]));
        } else {
          size_t comp_idx = i - prompt.size();
          input_ids.push_back(static_cast<int>(completion[comp_idx]));
        }
        positions.push_back(static_cast<int>(i));
      }

    output_indices.push_back(token_offset + seq_tokens - 1);
    seq_boundaries.push_back(token_offset);
    token_offset += seq_tokens;
  }
}

void ModelRunner::FillKQMask(ggml_tensor* mask, int n_kv, int n_tokens,
                              const std::vector<int>& seq_boundaries) {
  constexpr int kPad = 64;
  int n_tokens_pad = ((n_tokens + kPad - 1) / kPad) * kPad;

  // Use F16 for mask
  size_t total = static_cast<size_t>(n_kv) * static_cast<size_t>(n_tokens_pad);
  std::vector<uint16_t> mask_data(total, 0x0000);  // F16 zero

  const uint16_t f16_neg_inf = 0xFC00;  // F16 -inf

  int n_seqs = static_cast<int>(seq_boundaries.size());

  // Build block-diagonal causal mask
  for (int s = 0; s < n_seqs; ++s) {
    int seq_start = seq_boundaries[s];
    int seq_end = (s + 1 < n_seqs) ? seq_boundaries[s + 1] : n_tokens;
    int seq_len = seq_end - seq_start;

    // Causal within the sequence
    // mask[i,j] = 0 if i <= j (can attend), -inf if i > j
    for (int j = 0; j < seq_len; ++j) {
      for (int i = 0; i < seq_len; ++i) {
        if (i > j) {  // future tokens - mask out
          size_t idx = static_cast<size_t>(seq_start + i) +
                       static_cast<size_t>(seq_start + j) * static_cast<size_t>(n_kv);
          mask_data[idx] = f16_neg_inf;
        }
      }
    }
  }

  // Fill padded columns (beyond n_tokens) with -inf
  for (int j = n_tokens; j < n_tokens_pad; ++j) {
    for (int i = 0; i < n_kv; ++i) {
      size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(n_kv);
      mask_data[idx] = f16_neg_inf;
    }
  }

  model_->SetInputTensor(mask, mask_data.data(),
                          mask_data.size() * sizeof(uint16_t));
}

std::vector<int> ModelRunner::Sample(const float* logits_data, int n_vocab,
                                     int n_seqs,
                                     const std::vector<float>& temperatures,
                                     const std::vector<Sequence*>& seqs) {
  std::vector<int> tokens(static_cast<size_t>(n_seqs));
  static std::mt19937 rng(std::random_device{}());

  for (int s = 0; s < n_seqs; ++s) {
    float temp = temperatures[static_cast<size_t>(s)];
    const float* row =
        logits_data + static_cast<size_t>(s) * static_cast<size_t>(n_vocab);

    // Apply repetition penalty (vLLM convention: divide positive logits,
    // multiply negative ones) for tokens seen in prompt + generated context
    std::vector<float> penalized;
    float penalty = seqs[static_cast<size_t>(s)]->GetRepetitionPenalty();
    if (penalty > 1.0f) {
      penalized.assign(row, row + n_vocab);
      auto seen = seqs[static_cast<size_t>(s)]->GetPromptTokenIds();
      auto gen = seqs[static_cast<size_t>(s)]->GetCompletionTokenIds();
      for (auto tok : seen) {
        if (tok < static_cast<size_t>(n_vocab)) {
          float& v = penalized[tok];
          v = v < 0.0f ? v * penalty : v / penalty;
        }
      }
      for (auto tok : gen) {
        if (tok < static_cast<size_t>(n_vocab)) {
          float& v = penalized[tok];
          v = v < 0.0f ? v * penalty : v / penalty;
        }
      }
      row = penalized.data();
    }

    if (temp < 1e-6f) {
      // Greedy
      int best = 0;
      float best_val = row[0];
      for (int i = 1; i < n_vocab; ++i) {
        if (row[i] > best_val) {
          best_val = row[i];
          best = i;
        }
      }
      tokens[static_cast<size_t>(s)] = best;
    } else {
      // Temperature sampling with softmax
      float max_logit = row[0];
      for (int i = 1; i < n_vocab; ++i) {
        if (row[i] > max_logit)
          max_logit = row[i];
      }

      std::vector<float> probs(static_cast<size_t>(n_vocab));
      float sum = 0.0f;
      for (int i = 0; i < n_vocab; ++i) {
        probs[static_cast<size_t>(i)] = std::exp((row[i] - max_logit) / temp);
        sum += probs[static_cast<size_t>(i)];
      }

      std::uniform_real_distribution<float> dist(0.0f, sum);
      float r = dist(rng);
      float cumulative = 0.0f;
      int sample_idx = n_vocab - 1;
      for (int i = 0; i < n_vocab; ++i) {
        cumulative += probs[static_cast<size_t>(i)];
        if (cumulative >= r) {
          sample_idx = i;
          break;
        }
      }
      tokens[static_cast<size_t>(s)] = sample_idx;
    }
  }

  return tokens;
}

}  // namespace engine
