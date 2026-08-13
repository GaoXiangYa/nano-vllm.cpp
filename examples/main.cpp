#include <iostream>
#include <string>
#include <vector>
#include "config.h"
#include "engine/llm_engine.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [prompt]\n";
    return 1;
  }

  std::string model_path = argv[1];
  std::string prompt = (argc >= 3) ? argv[2] : "Hello, how are you?";

  std::cout << "Loading model from: " << model_path << "\n";
  std::cout << "Prompt: " << prompt << "\n\n";

  try {
    Config config;
    config.model = model_path;
    config.enforce_eager = true;
    config.tensor_parallel_size = 1;

    engine::LLMEngine engine(model_path, config);

    SamplingParams params;
    params.temperature = 0.8f;
    params.max_tokens = 64;
    params.repetition_penalty = 1.2f;

    auto outputs = engine.Generate({prompt}, params);

    for (size_t i = 0; i < outputs.size(); ++i) {
      std::cout << "Output " << i << ": " << outputs[i] << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
