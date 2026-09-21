# Third-party notices

Swiftlet builds on the work of several open-source projects.

## Design references

- **TurboFieldfare** (Apache 2.0), Andrey Mikhaylov,
  https://github.com/drumih/turbo-fieldfare: the `.qpack` container layout,
  single-pread expert blobs, streaming installer design, expert cache, and
  runtime-compiled Metal library pattern are modeled on TurboFieldfare's
  `.gturbo` format and runtime.
- **colibrì** (Apache 2.0), https://github.com/JustVugg/colibri: cache and
  placement policies (learned pinning, router-lookahead prefetch, batch-union
  prefill) and the correctness-first measurement discipline.

## Reference implementations

- **mlx-lm** (MIT), Apple Inc., https://github.com/ml-explore/mlx-lm: the
  `qwen3_next` model implementation is the correctness oracle for this
  project; the gated-delta Metal kernel is a port of mlx-lm's
  `gated_delta.py` kernel. Vendored reference copies live in `references/`.
- **llama.cpp** (MIT), https://github.com/ggml-org/llama.cpp: secondary
  reference for the Qwen3-Next graph.

## Dependencies

The experimental GLM CPU feed-forward, linear-attention, and hyper-connection
implementations and fixture generators use **mlx-vlm** as their reference,
specifically `glm5_next/language.py`, `gated_delta.py`, and
`deepseek_v4/hyper_connection.py` at
commit `a74c7de90a344a2c2c7334acb4e48b57a40480e2`:
https://github.com/Blaizzy/mlx-vlm. Its MIT notice follows:

Copyright © 2025 Prince Canuma
Copyright (c) 2026 Apple Inc. (hyper-connection source)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

- **swift-transformers** (Apache 2.0), Hugging Face: tokenization and chat
  templates.
- **swift-nio** (Apache 2.0), Apple Inc.: the loopback HTTP server.

## Model weights

Model weights are not distributed with this project. The Qwen3-Next and
Qwen3.5/3.6 checkpoints are released by Alibaba's Qwen team under Apache 2.0;
quantized community conversions are downloaded from their respective Hugging
Face repositories.
