#pragma once

struct SamplingParams {
  float temperature = 1.0f;
  int max_tokens = 64;
  bool ignore_eos = false;
  // Repetition penalty > 1.0 penalizes already-seen tokens
  float repetition_penalty = 1.0f;
};