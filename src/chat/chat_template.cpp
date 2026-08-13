#include "chat_template.h"
#include <sstream>

namespace chat {

ChatTemplateType DetectChatTemplate(const std::string& tmpl) {
  if (tmpl.find("<|im_start|>") != std::string::npos &&
      tmpl.find("<|im_end|>") != std::string::npos) {
    return ChatTemplateType::CHATML;
  }
  return ChatTemplateType::UNKNOWN;
}

std::string ApplyChatTemplate(ChatTemplateType type,
                              const std::vector<ChatMessage>& messages,
                              bool add_generation_prompt) {
  if (type == ChatTemplateType::CHATML) {
    std::stringstream ss;
    for (const auto& msg : messages) {
      ss << "<|im_start|>" << msg.role << "\n" << msg.content
         << "<|im_end|>\n";
    }
    if (add_generation_prompt) {
      ss << "<|im_start|>assistant\n";
    }
    return ss.str();
  }
  // Unknown template: fall back to plain concatenation
  std::stringstream ss;
  for (const auto& msg : messages) {
    ss << msg.content;
  }
  return ss.str();
}

}  // namespace chat
