#pragma once

#include <string>
#include <vector>

namespace chat {

struct ChatMessage {
  std::string role;     // "system" | "user" | "assistant"
  std::string content;
};

// Supported template families (built-in renderers, llama.cpp style)
enum class ChatTemplateType {
  CHATML,   // <|im_start|>role\ncontent<|im_end|>\n (Qwen2/3, ChatGLM4, etc.)
  UNKNOWN,
};

// Detect template family from a Jinja2 chat_template string
ChatTemplateType DetectChatTemplate(const std::string& tmpl);

// Render messages into a prompt string
std::string ApplyChatTemplate(ChatTemplateType type,
                              const std::vector<ChatMessage>& messages,
                              bool add_generation_prompt);

}  // namespace chat
