# nano-vllm.cpp

A lightweight LLM inference engine written in C++20 + [ggml](https://github.com/ggml-org/ggml), inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

Currently focuses on **Qwen3-family models** and includes:

- Prefill / Decode two-stage scheduling
- Block-table-driven Paged KV Cache
- Prefix Caching (shared prefix blocks)
- BF16 KV Cache
- Multi-sequence batch decode
- Chat template detection (ChatML, etc.)
- Temperature sampling and repetition penalty

---

## Table of Contents

- [nano-vllm.cpp](#nano-vllmcpp)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Getting the Code](#getting-the-code)
  - [Building](#building)
    - [1. Build GGML](#1-build-ggml)
    - [2. Build This Project](#2-build-this-project)
    - [3. Optional: Enable CUDA](#3-optional-enable-cuda)
  - [Running](#running)
    - [Plain Text Generation](#plain-text-generation)
    - [Chat Mode](#chat-mode)
  - [C++ API Example](#c-api-example)
  - [Project Structure](#project-structure)
  - [Current Status and Limitations](#current-status-and-limitations)

---

## Requirements

- Linux / macOS / WSL
- CMake >= 3.20
- A C++20 compiler:
  - GCC >= 11
  - Clang >= 14
  - MSVC (may need adaptation)
- Enough RAM or VRAM to load the model
- Optional: NVIDIA GPU + CUDA Toolkit (for `USE_CUDA=ON`)

---

## Getting the Code

This project uses git submodules for third-party dependencies:

```bash
git clone https://github.com/GaoXiangYa/nano-vllm.cpp
cd nano-vllm.cpp
git submodule update --init --recursive
```

Dependencies include:

- `third_party/ggml`
- `third_party/tokenizers-cpp`
- `third_party/safetensors-cpp`
- `third_party/xxHash`

---

## Building

### 1. Build GGML

This project's CMake looks for:

```text
third_party/ggml/build/src
```

and expects `libggml.so` / `libggml-cpu.so` there. Build GGML first:

```bash
cmake -S third_party/ggml -B third_party/ggml/build \
  -DGGML_BUILD_EXAMPLES=OFF \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_BENCHMARKS=OFF

cmake --build third_party/ggml/build -j$(nproc)
```

### 2. Build This Project

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

After building, the example binary is located at:

```text
build/examples/nano-vllm-example
```

### 3. Optional: Enable CUDA

The default backend is CPU. To try CUDA:

```bash
cmake -S . -B build-cuda \
  -DUSE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda -j$(nproc)
```

Note: GGML itself also needs to be built with CUDA support. This project is primarily validated with the CPU backend, so the CUDA path may need extra adaptation.

---

## Running

### Plain Text Generation

```bash
./build/examples/nano-vllm-example /path/to/Qwen3-0.6B "Hello, how are you?"
```

### Chat Mode

```bash
./build/examples/nano-vllm-example /path/to/Qwen3-0.6B --chat "你好"
```

Arguments:

| Argument | Description |
|---|---|
| `<model_path>` | HuggingFace-format model directory, must contain `config.json`, `tokenizer.json`, and `model.safetensors` |
| `--chat` | Use chat template mode |
| `[prompt]` | User prompt, default is `Hello, how are you?` |

Example output:

```text
Loading model from: /path/to/Qwen3-0.6B
Loaded config: n_layer=28, n_embd=1024, n_head=16, n_head_kv=8, n_embd_head=128, n_ctx=40960, n_ff=3072, rope_theta=1e+06
KV cache size =  4480.00 MB
Loaded model size =  1433.62 MB
Prompt: 你好

.
Reply: <think>

</think>

您好！有什么可以帮助您的吗？需要尽快解决什么问题呢？ 😊<|im_end|>
```

---

## C++ API Example

`examples/main.cpp` is a complete reference. The core usage is:

```cpp
#include "config.h"
#include "engine/llm_engine.h"
#include "sampling_params.h"

Config config;
config.model = "/path/to/Qwen3-0.6B";
config.enforce_eager = true;
config.tensor_parallel_size = 1;

engine::LLMEngine engine(config.model, config);

SamplingParams params;
params.temperature = 0.8f;
params.max_tokens = 64;
params.repetition_penalty = 1.2f;

// Plain generation
auto outputs = engine.Generate({"Hello, how are you?"}, params);

// Chat generation
std::vector<chat::ChatMessage> messages = {
    {"user", "你好"},
};
auto reply = engine.Chat(messages, params);
```

---

## Project Structure

```text
src/
├── chat/
│   └── chat_template.cpp          # Chat template handling
├── engine/
│   ├── block_manager.cpp          # Block allocation, refcount, prefix hash
│   ├── llm_engine.cpp             # High-level engine API
│   ├── model_runner.cpp           # Prefill/decode and KV cache I/O
│   ├── scheduler.cpp              # Scheduler
│   └── sequence.cpp               # Sequence state
├── include/
│   ├── config.h
│   ├── sampling_params.h
│   ├── chat_template.h
│   ├── engine/
│   └── models/
└── models/
    └── qwen3.cpp                  # Qwen3 model implementation
```

---

## Current Status and Limitations

What is implemented:

- Paged KV Cache: physical slots computed via `block_table`
- Prefix Caching: shared prefix blocks, skipped duplicate computation
- Multi-sequence batch decode
- Chat template support

Current limitations:

- Uses GGML compute graphs instead of PyTorch + flash-attn
- Attention currently gathers K/V through `block_table` and then concatenates; it is not as fast as a native paged attention kernel
- Tensor Parallel is not supported yet
- CUDA Graph is not supported yet
- The model side currently targets the Qwen3 family

Feel free to continue improving it.
