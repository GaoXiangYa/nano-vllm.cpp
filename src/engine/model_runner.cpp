#include "engine/model_runner.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <random>
#include "engine/sequence.h"
#include "ggml.h"
#include "models/qwen3.h"

namespace engine {

ModelRunner::ModelRunner(const std::string& model_path, Config& config)
    : config_(config),
      model_(std::make_unique<models::Qwen3Model>(
          model_path, 0, 0, config_.kvcache_block_size)) {
}

int ModelRunner::GetMaxContextLen() const {
  return model_->GetHparams().n_ctx;
}

std::vector<int> ModelRunner::Run(const std::vector<Sequence*>& seqs,
                                  bool is_prefill) {
  if (seqs.empty()) return {};

  std::vector<int> input_ids;
  std::vector<int> positions;
  std::vector<int> output_indices;
  std::vector<int> seq_boundaries;
  std::vector<int> seq_starts;   // per-seq start logical token position
  std::vector<int> token_offsets;  // per-seq token offset in concatenated input
  PrepareInputs(seqs, is_prefill, input_ids, positions, output_indices,
                seq_boundaries, seq_starts, token_offsets);

  auto& hparams = model_->GetHparams();
  const int n_vocab = hparams.n_vocab;

  if (is_prefill) {
    const int n_tokens = static_cast<int>(input_ids.size());
    const int n_outputs = static_cast<int>(output_indices.size());
    model_->BuildPrefillGraph(n_tokens, n_outputs);

    if (!model_->AllocateGraph()) {
      std::cerr << "Failed to allocate compute graph\n";
      return {};
    }

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

    FillCausalMask(inp_mask, n_tokens, seq_boundaries);

    auto status = model_->GraphCompute();
    if (status != GGML_STATUS_SUCCESS) return {};

    // Store per-layer K/V into the KV cache.  This must happen while the
    // graph outputs (and their backing tensors) are still allocated.
    StorePrefillKV(seqs, token_offsets, seq_starts);

    auto* logits_tensor = model_->GetLogits();
    std::vector<float> logits_data(static_cast<size_t>(n_vocab) * n_outputs);
    ggml_backend_tensor_get(logits_tensor, logits_data.data(), 0,
                            logits_data.size() * sizeof(float));

    std::vector<float> temperatures;
    temperatures.reserve(seqs.size());
    for (auto* seq : seqs) temperatures.push_back(seq->GetTemperature());

    return Sample(logits_data.data(), n_vocab, n_outputs, temperatures, seqs);
  }

  // Decode: one new token per sequence
  const int n_seqs = static_cast<int>(seqs.size());
  std::vector<int> ctx_lens(n_seqs);
  int total_hist = 0;
  for (int i = 0; i < n_seqs; ++i) {
    ctx_lens[i] = static_cast<int>(seqs[i]->SequenceLen()) - 1;
    total_hist += ctx_lens[i];
  }

  model_->BuildDecodeGraph(n_seqs, total_hist, ctx_lens);

  if (!model_->AllocateGraph()) {
    std::cerr << "Failed to allocate compute graph\n";
    return {};
  }

  auto* graph = model_->GetComputeGraph();
  auto* inp_tok = ggml_graph_get_tensor(graph, "inp_tokens");
  auto* inp_pos = ggml_graph_get_tensor(graph, "inp_pos");
  auto* inp_mask = ggml_graph_get_tensor(graph, "inp_kq_mask");

  model_->SetInputTensor(inp_tok, input_ids.data(),
                         n_seqs * static_cast<int>(sizeof(int)));
  model_->SetInputTensor(inp_pos, positions.data(),
                         n_seqs * static_cast<int>(sizeof(int)));

  // Load history K/V from cache into k_hist_in/v_hist_in
  LoadHistoryKV(seqs, ctx_lens, total_hist);

  FillVarlenMask(inp_mask, total_hist + n_seqs, n_seqs, ctx_lens);

  auto status = model_->GraphCompute();
  if (status != GGML_STATUS_SUCCESS) return {};

  // Append new K/V to cache
  AppendDecodeKV(seqs);

  auto* logits_tensor = model_->GetLogits();
  std::vector<float> logits_data(static_cast<size_t>(n_vocab) * n_seqs);
  ggml_backend_tensor_get(logits_tensor, logits_data.data(), 0,
                          logits_data.size() * sizeof(float));

  std::vector<float> temperatures;
  temperatures.reserve(seqs.size());
  for (auto* seq : seqs) temperatures.push_back(seq->GetTemperature());

  return Sample(logits_data.data(), n_vocab, n_seqs, temperatures, seqs);
}

void ModelRunner::PrepareInputs(const std::vector<Sequence*>& seqs,
                                bool is_prefill, std::vector<int>& input_ids,
                                std::vector<int>& positions,
                                std::vector<int>& output_indices,
                                std::vector<int>& seq_boundaries,
                                std::vector<int>& seq_starts,
                                std::vector<int>& token_offsets) {
  input_ids.clear();
  positions.clear();
  output_indices.clear();
  seq_boundaries.clear();
  seq_starts.clear();
  token_offsets.clear();

  int token_offset = 0;
  for (auto* seq : seqs) {
    size_t seq_len = seq->SequenceLen();
    int seq_tokens;
    size_t start_pos;

    if (is_prefill) {
      start_pos = seq->GetNumCachedTokens();
      seq_tokens = static_cast<int>(seq_len - start_pos);
    } else {
      // Decode: only the last (new) token
      start_pos = seq_len - 1;
      seq_tokens = 1;
    }

    seq_starts.push_back(static_cast<int>(start_pos));
    token_offsets.push_back(token_offset);

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

void ModelRunner::FillCausalMask(ggml_tensor* mask, int n_tokens,
                                 const std::vector<int>& seq_boundaries) {
  constexpr int kPad = 64;
  int n_pad = ((n_tokens + kPad - 1) / kPad) * kPad;
  const uint16_t neg_inf = 0xFC00;
  const uint16_t zero = 0x0000;

  // Start with everything masked: sequences must not attend across batch
  // boundaries.  Within each sequence only key_pos <= query_pos is unmasked.
  std::vector<uint16_t> mask_data(static_cast<size_t>(n_tokens) * n_pad,
                                  neg_inf);
  const int n_seqs = static_cast<int>(seq_boundaries.size());
  for (int s = 0; s < n_seqs; ++s) {
    const int seq_start = seq_boundaries[s];
    const int seq_end =
        (s + 1 < n_seqs) ? seq_boundaries[s + 1] : n_tokens;
    for (int j = seq_start; j < seq_end; ++j) {
      for (int i = seq_start; i <= j; ++i) {
        mask_data[static_cast<size_t>(i) +
                  static_cast<size_t>(j) * n_tokens] = zero;
      }
    }
  }

  model_->SetInputTensor(mask, mask_data.data(),
                         mask_data.size() * sizeof(uint16_t));
}

void ModelRunner::FillVarlenMask(ggml_tensor* mask, int n_kv, int n_seqs,
                                 const std::vector<int>& ctx_lens) {
  constexpr int kPad = 64;
  int n_pad = ((n_seqs + kPad - 1) / kPad) * kPad;
  std::vector<uint16_t> mask_data(static_cast<size_t>(n_kv) * n_pad, 0xFC00);
  const uint16_t zero = 0x0000;

  // History is concatenated per sequence; each seq's new token follows at
  // global position total_hist + s. Seq s attends only its own region.
  int total_hist = 0;
  for (int c : ctx_lens) total_hist += c;
  int hist_offset = 0;
  for (int s = 0; s < n_seqs; ++s) {
    for (int t = 0; t < ctx_lens[s]; ++t) {
      mask_data[static_cast<size_t>(hist_offset + t) +
                static_cast<size_t>(s) * n_kv] = zero;
    }
    mask_data[static_cast<size_t>(total_hist + s) +
              static_cast<size_t>(s) * n_kv] = zero;
    hist_offset += ctx_lens[s];
  }

  model_->SetInputTensor(mask, mask_data.data(),
                         mask_data.size() * sizeof(uint16_t));
}

// Extract per-layer K/V (graph outputs, F32) and write into the KV cache
// (BF16). Input tokens are concatenated per sequence; token_offsets/seq_starts
// map them back to (slot, position).
void ModelRunner::StorePrefillKV(const std::vector<Sequence*>& seqs,
                                 const std::vector<int>& token_offsets,
                                 const std::vector<int>& seq_starts) {
  auto& hparams = model_->GetHparams();
  const int n_layer = hparams.n_layer;
  const int kv_per_token = hparams.n_embd_k_gqa;  // 1024 elements

  auto* graph = model_->GetComputeGraph();
  const int n_seqs = static_cast<int>(seqs.size());

  for (int il = 0; il < n_layer; ++il) {
    auto* k_out = ggml_graph_get_tensor(graph, std::format("k_out_{}", il).c_str());
    auto* v_out = ggml_graph_get_tensor(graph, std::format("v_out_{}", il).c_str());
    if (!k_out || !v_out) continue;

    const int n_tokens = static_cast<int>(ggml_nelements(k_out)) / kv_per_token;
    std::vector<float> k_buf(static_cast<size_t>(n_tokens) * kv_per_token);
    std::vector<float> v_buf(static_cast<size_t>(n_tokens) * kv_per_token);
    ggml_backend_tensor_get(k_out, k_buf.data(), 0,
                            k_buf.size() * sizeof(float));
    ggml_backend_tensor_get(v_out, v_buf.data(), 0,
                            v_buf.size() * sizeof(float));

    // Convert per sequence's tokens to BF16 and write at (slot, start+i)
    for (int s = 0; s < n_seqs; ++s) {
      int off = token_offsets[s];
      int start = seq_starts[s];
      int cnt = (s + 1 < n_seqs ? token_offsets[s + 1] : n_tokens) - off;
      if (cnt <= 0) continue;

      const auto& block_table = seqs[s]->GetBlockTable();
      const int block_size = config_.kvcache_block_size;
      std::vector<ggml_bf16_t> k_bf16(static_cast<size_t>(cnt) * kv_per_token);
      std::vector<ggml_bf16_t> v_bf16(static_cast<size_t>(cnt) * kv_per_token);
      ggml_fp32_to_bf16_row(k_buf.data() + static_cast<size_t>(off) * kv_per_token,
                            k_bf16.data(), static_cast<int64_t>(cnt) * kv_per_token);
      ggml_fp32_to_bf16_row(v_buf.data() + static_cast<size_t>(off) * kv_per_token,
                            v_bf16.data(), static_cast<int64_t>(cnt) * kv_per_token);
      for (int i = 0; i < cnt; ++i) {
        const int pos = start + i;
        const int block_id = static_cast<int>(block_table[pos / block_size]);
        const int slot = block_id * block_size + pos % block_size;
        model_->WriteKVToCache(
            il, slot,
            k_bf16.data() + static_cast<size_t>(i) * kv_per_token,
            v_bf16.data() + static_cast<size_t>(i) * kv_per_token);
      }
    }
  }
}

// Load each sequence's history K/V from the BF16 cache into the F32 graph
// inputs k_hist_in/v_hist_in, shape [head_dim, n_head_kv, total_hist].
void ModelRunner::LoadHistoryKV(const std::vector<Sequence*>& seqs,
                                const std::vector<int>& ctx_lens,
                                int total_hist) {
  auto& hparams = model_->GetHparams();
  const int n_layer = hparams.n_layer;
  const int kv_per_token = hparams.n_embd_k_gqa;
  const int n_seqs = static_cast<int>(seqs.size());

  auto* graph = model_->GetComputeGraph();
  const size_t hist_bytes =
      static_cast<size_t>(total_hist) * kv_per_token * sizeof(uint16_t);

  const size_t hist_f32_bytes =
      static_cast<size_t>(total_hist) * kv_per_token * sizeof(float);
  std::vector<uint8_t> k_bf16(hist_bytes, 0);
  std::vector<uint8_t> v_bf16(hist_bytes, 0);
  for (int il = 0; il < n_layer; ++il) {
    auto* k_hist = ggml_graph_get_tensor(
        graph, std::format("k_hist_in_{}", il).c_str());
    auto* v_hist = ggml_graph_get_tensor(
        graph, std::format("v_hist_in_{}", il).c_str());
    if (!k_hist || !v_hist) continue;

    int hist_offset = 0;
    std::vector<ggml_bf16_t> k_tmp(kv_per_token);
    std::vector<ggml_bf16_t> v_tmp(kv_per_token);
    const int block_size = config_.kvcache_block_size;
    for (int s = 0; s < n_seqs; ++s) {
      int ctx = ctx_lens[s];
      if (ctx <= 0) continue;
      const auto& block_table = seqs[s]->GetBlockTable();
      for (int i = 0; i < ctx; ++i) {
        const int block_id = static_cast<int>(block_table[i / block_size]);
        const int slot = block_id * block_size + i % block_size;
        model_->ReadKVFromCache(il, slot, k_tmp.data(), v_tmp.data());
        std::memcpy(
            k_bf16.data() +
                static_cast<size_t>(hist_offset + i) * kv_per_token * sizeof(uint16_t),
            k_tmp.data(), static_cast<size_t>(kv_per_token) * sizeof(uint16_t));
        std::memcpy(
            v_bf16.data() +
                static_cast<size_t>(hist_offset + i) * kv_per_token * sizeof(uint16_t),
            v_tmp.data(), static_cast<size_t>(kv_per_token) * sizeof(uint16_t));
      }
      hist_offset += ctx;
    }

    std::vector<float> k_f32(static_cast<size_t>(total_hist) * kv_per_token);
    std::vector<float> v_f32(static_cast<size_t>(total_hist) * kv_per_token);
    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(k_bf16.data()),
                          k_f32.data(),
                          static_cast<int64_t>(total_hist) * kv_per_token);
    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(v_bf16.data()),
                          v_f32.data(),
                          static_cast<int64_t>(total_hist) * kv_per_token);

    // Reorder cache head-major blocks into k_hist standard layout
    // [head_dim, n_head_kv, total_hist]: idx = d + h*head_dim + t*1024
    std::vector<float> k_reordered(static_cast<size_t>(total_hist) * kv_per_token);
    std::vector<float> v_reordered(static_cast<size_t>(total_hist) * kv_per_token);
    for (int t = 0; t < total_hist; ++t) {
      for (int h = 0; h < hparams.n_head_kv; ++h) {
        for (int d = 0; d < hparams.n_embd_head; ++d) {
          size_t src = static_cast<size_t>(t) * kv_per_token +
                       static_cast<size_t>(h) * hparams.n_embd_head + d;
          size_t dst = static_cast<size_t>(d) +
                       static_cast<size_t>(h) * hparams.n_embd_head +
                       static_cast<size_t>(t) * kv_per_token;
          k_reordered[dst] = k_f32[src];
          v_reordered[dst] = v_f32[src];
        }
      }
    }
    model_->SetInputTensor(k_hist, k_reordered.data(), hist_f32_bytes);
    model_->SetInputTensor(v_hist, v_reordered.data(), hist_f32_bytes);
  }
}

// After decode graph compute, append each sequence's new K/V (F32 outputs)
// converted to BF16 into the paged cache at its physical slot.
void ModelRunner::AppendDecodeKV(const std::vector<Sequence*>& seqs) {
  auto& hparams = model_->GetHparams();
  const int n_layer = hparams.n_layer;
  const int kv_per_token = hparams.n_embd_k_gqa;
  const int n_seqs = static_cast<int>(seqs.size());

  auto* graph = model_->GetComputeGraph();

  for (int il = 0; il < n_layer; ++il) {
    auto* k_out = ggml_graph_get_tensor(graph, std::format("k_out_{}", il).c_str());
    auto* v_out = ggml_graph_get_tensor(graph, std::format("v_out_{}", il).c_str());
    if (!k_out || !v_out) continue;

    std::vector<float> k_buf(static_cast<size_t>(n_seqs) * kv_per_token);
    std::vector<float> v_buf(static_cast<size_t>(n_seqs) * kv_per_token);
    ggml_backend_tensor_get(k_out, k_buf.data(), 0,
                            k_buf.size() * sizeof(float));
    ggml_backend_tensor_get(v_out, v_buf.data(), 0,
                            v_buf.size() * sizeof(float));

    const int block_size = config_.kvcache_block_size;
    for (int s = 0; s < n_seqs; ++s) {
      const int pos = static_cast<int>(seqs[s]->SequenceLen()) - 1;
      const auto& block_table = seqs[s]->GetBlockTable();
      const int block_id = static_cast<int>(block_table[pos / block_size]);
      const int slot = block_id * block_size + pos % block_size;
      ggml_bf16_t k_bf16[1024];
      ggml_bf16_t v_bf16[1024];
      ggml_fp32_to_bf16_row(
          k_buf.data() + static_cast<size_t>(s) * kv_per_token, k_bf16, kv_per_token);
      ggml_fp32_to_bf16_row(
          v_buf.data() + static_cast<size_t>(s) * kv_per_token, v_bf16, kv_per_token);
      model_->WriteKVToCache(il, slot, k_bf16, v_bf16);
    }
  }
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
      float max_logit = row[0];
      for (int i = 1; i < n_vocab; ++i) {
        if (row[i] > max_logit) max_logit = row[i];
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
