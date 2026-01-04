#pragma once

struct SamplingParams {
  float temperature = 1.0f;
  static constexpr int max_tokens = 64;
  bool ignore_eos = false;
};