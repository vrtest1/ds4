# Experimental oldest-image eviction

This is a vrtest1 fork experiment based on antirez/ds4 main
`0aaea5a238fb41a35106a551e73c8409dfb751ac` and PR #1022 commit
`fa0af869d554f96a1d5d747d1aaf0962301054e9`. The original commit is preserved
with cherry-pick provenance. No upstream submission is part of this work.

## Build and enable

Build on the machine/backend on which you normally run ds4 (`make` for the
normal Metal/CUDA target; see the upstream build instructions). Enable the
policy in the server process environment:

```sh
DS4_VISION_KEEP_IMAGES=16 ./ds4-server -m /path/to/model.gguf \
  --vision /path/to/vision.gguf --ctx 262144
```

Use the same other options as your baseline server. Without this variable,
requests over 16 images are rejected as before. Restart the server to change
the setting: it is read once on the first image request.

Unlike the original PR, this branch accepts only 1..16 retained images. The
upstream visible-history and live-suffix bookkeeping still has 16-entry
arrays. Values such as 32 or 64 warn and fall back to rejection, rather than
silently disabling some continuation paths. Configurable larger windows are
deferred. This branch does not add a `--max-images` option.

## Semantics and limitations

- Count images in the entire submitted history, in message/block order.
- Replace the oldest image markers with a fixed omission note; retain the
  surrounding rendered user, assistant and tool text.
- Encode only the retained images. A repeated text-only follow-up with the
  full history uses the same reduced image set.
- Image eviction invalidates incompatible live KV. Expect a cold prefill;
  subsequent unchanged-image follow-ups can reuse the rebuilt live state.
- Embedding cache hits are separate from KV reuse. Its shared upstream limits
  remain 32 entries and 128 MiB; hits still copy embedding memory.
- Full history replay is required to recover when live state is unavailable.
  Tool-output-only continuation is not made durable by this patch.
- Upstream Vision disk KV persistence remains disabled. Restarting requires
  replay and recomputation. Hidden reasoning absent from replay can be lost.
- The 64 MiB HTTP body limit and context limit remain. Reduction happens after
  JSON/base64 parsing, so it does not bound network traffic or parsing memory.
- With reduction enabled, submitted histories over 1024 images are rejected.
  This is not an unlimited-history solution.
- Historical image pixels that were omitted cannot be revisited, even though
  previous assistant descriptions of those images remain in the transcript.

## Validation

The additional model-free server tests cover setting boundaries, exact text
preservation for 1/15/16/17/18/24 images, deterministic history replay, marker
failure atomicity, rejection of shifted image identities by the slot probe,
and reuse after a synthetic checkpoint is rebuilt with the retained set.
These are not substitutes for a real-model end-to-end test.

Validation performed on 2026-10-06: Linux x86_64 in the official `gcc:14`
Docker image, CPU backend, no model loaded. `make -j4 cpu` built all five
executables; `./ds4_test --server` and `./tests/test_session_state` passed.
The final build/test logs contained no compiler warnings. The Python benchmark
passed syntax compilation and its `--help` smoke check. GPU compilation,
real-model image encoding/generation, performance, and restart/concurrent
Vision scenarios have not been executed in this environment.

The CPU test runner was built with:

```sh
make -j4 cpu
make -j4 ds4_test \
  CORE_OBJS='ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o' \
  CFLAGS='-O2 -g -D_GNU_SOURCE -DDS4_NO_GPU -Wall -Wextra -std=c99' \
  DS4_LINK=cc DS4_LINK_LIBS='-lm -pthread'
./ds4_test --server
make -j4 tests/test_session_state
./tests/test_session_state
```

Use a clean build directory when switching the test runner between GPU and CPU
flags; make does not detect changes in compile flags on existing object files.

Run the existing server and session tests, and on your real vision backend:

```sh
./ds4_test --server
python3 tests/bench_vision_window.py --model YOUR_MODEL_ID \
  --url http://127.0.0.1:8080 --output vision-window.json
python3 tests/test_server_vision_cache.py --model YOUR_MODEL_ID \
  --url http://127.0.0.1:8080 --output vision-cache-results
```

The window benchmark exercises 16 images, a text-only continuation, image 17,
another text-only continuation, image 18, and another text-only continuation.
It records TTFT, total time, usage/cached tokens, and estimated decode tok/s.
The SSE-based decode rate is approximate; use server logs for exact engine
prefill/decode measurements. Run on an idle server, compare identical settings,
and repeat with `--archive-lines N` to examine growing text-history cost.
The fixture set repeats five small images, so this measures a warm-embedding
case, not the encoder cost of a fresh large screenshot every turn.

Run the same benchmark with the variable unset to check baseline behavior:
the first 17-image request should return HTTP 400, which the script records
before exiting nonzero. Test actual screenshots and all APIs separately before
using this experimental branch for unattended agent work.
