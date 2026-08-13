#include <iostream>
#include <string>
#include <vector>
#include "chat_template.h"
#include "config.h"
#include "engine/llm_engine.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [--chat] [prompt]\n";
    return 1;
  }

  std::string model_path = argv[1];
  bool chat_mode = false;
  std::string prompt = "Hello, how are you?";
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--chat") {
      chat_mode = true;
    } else {
      prompt = arg;
    }
  }

  std::cout << "Loading model from: " << model_path << "\n";

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

    if (chat_mode) {
      if (engine.GetChatTemplateType() == chat::ChatTemplateType::UNKNOWN) {
        std::cerr << "No supported chat template found\n";
        return 1;
      }
      std::vector<chat::ChatMessage> messages = {
          {"user", prompt},
      };
      std::cout << "Prompt: " << prompt << "\n\n";
      auto reply = engine.Chat(messages, params);
      std::cout << "Reply: " << reply << "\n";
    } else {
      std::cout << "Prompt: " << prompt << "\n\n";
      auto outputs = engine.Generate({prompt}, params);
      for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "Output " << i << ": " << outputs[i] << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
