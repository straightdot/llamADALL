# llamaADALL

Custom llama.cpp build for RTX 40 series (Ada Lovelace). This has been optimized, tested, benchmarked & built using Opus 5.5.

This is a modified version of [llama.cpp](https://github.com/ggml-org/llama.cpp), based on release b11243.

It is tuned for NVIDIA RTX 40 series cards (Ada Lovelace) and for one model family: hybrid Qwen models with an MTP draft layer, such as Qwen3.8-27B.

The output is the same as normal llama.cpp. All changes are on by default, and each one can be turned off with an environment variable.

## What it changes

- Sampling runs fully on the GPU, including tool calls (grammar) and repeat penalty.
- The MTP draft model (speculative decoding) also runs fully on the GPU, so generation is faster.
- Faster GPU code for attention and for the model's Gated DeltaNet layers, especially at long context.
- Smarter context checkpoints for hybrid models: they are kept in VRAM and copied to RAM in the background, so going back to an earlier point in a chat is quick.
- VRAM use is planned around what Windows actually allows, so the card does not slow down from running out of memory.
- Better for agents and tools that switch between chats. The main chat stays in VRAM while a short side request runs, and saving to RAM happens in the background.

## Speed compared to normal llama.cpp

Tested on one long chat that grows from 8K to 124K tokens. Each turn adds new code or docs and asks a real question. Both builds used the same server settings: MTP with 3 draft tokens, the chat template and thinking on. The numbers are the average of 2 runs on a 16 GB RTX 4060 Ti, power limited to 133 W (To have better hotspot temps. If power is not limited, results are even better by 5-8%.).

| context size | generation speed (tokens/s), base llama -> this build | prompt speed (tokens/s), base llama -> this build |
|---|---|---|
| 8K | 26.2 -> **44.3** | 479 -> **870** |
| 25K | 23.9 -> **42.8** | 458 -> **837** |
| 41K | 22.6 -> **37.0** | 373 -> **750** |
| 58K | 20.9 -> **40.9** | 315 -> **676** |
| 75K | 21.2 -> **40.5** | 273 -> **624** |
| 91K | 16.7 -> **33.6** | 240 -> **576** |
| 108K | 15.5 -> **31.0** | 214 -> **540** |
| 124K | 16.6 -> **33.0** | 193 -> **496** |

Both builds get slower as the chat gets longer, but this build slows down about half as much. From around 90K tokens it generates about twice as fast as normal llama.cpp.

## How to run

Build it the same way as normal llama.cpp with CUDA. Then start the server as usual, for example:

This is the command used for benchmarks.

```
llama-server -m Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf --chat-template-file chat_template.jinja --reasoning-format deepseek -ngl 99 -fa on -c 132000 -np 1 -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3 --jinja --no-context-shift --reasoning on --reasoning-preserve --reasoning-budget 2048 -b 1536 -ub 1536 -sm none -mg 0 --fit off --cache-ram 4096 --metrics
```

To turn a feature off, set its variable to 0 before starting the server. For example: `LLAMA_PARK=0`, `LLAMA_CKPT_NOWAIT=0`, `LLAMA_PCACHE_PREFIX=0`, `LLAMA_PCACHE_SAVE_OVERLAP=0`, `LLAMA_CKPT_IN_SPAN=0`, `LLAMA_VRAM_BUDGET=0`, `LLAMA_MTP_DRAFT_VOCAB=0`.

| variable | what setting it to 0 does |
|---|---|
| `LLAMA_PARK` | The main chat is no longer kept in VRAM while a side request runs; it is saved to RAM instead. |
| `LLAMA_CKPT_NOWAIT` | A new request may wait until earlier checkpoint copies to RAM have finished. |
| `LLAMA_PCACHE_PREFIX` | An edited chat that comes back from the RAM cache is read again from the start, instead of reusing its unchanged beginning. |
| `LLAMA_PCACHE_SAVE_OVERLAP` | Saving a chat to RAM makes the next request wait, instead of running in the background. |
| `LLAMA_CKPT_IN_SPAN` | No extra checkpoints at the end of thinking or at the start of a tool call. |
| `LLAMA_VRAM_BUDGET` | Checkpoint slots and image memory are sized from free VRAM only, ignoring the Windows VRAM limit. |
| `LLAMA_MTP_DRAFT_VOCAB` | The draft model searches the whole vocabulary instead of a short list of common tokens (slower drafts). |


## Limits

- **GPU.** Made for RTX 40 series (Ada Lovelace). It was tested on one 16 GB card only. Other 40 series cards use the same code, but cards with a different VRAM size or core count may not get the same speedup.
- **Windows only.** It was built and tested on Windows 10 with CUDA 13.3. Linux and macOS were not tested.
- **One model family.** The speedups only work for models shaped like this one (hybrid Qwen with an MTP layer).
- **KV cache types.** Only q4_0/q4_0 and q8_0/q4_0 were tested.
- **One chat slot and one GPU.** Use `-np 1` and `-sm none`. With more slots, some features turn themselves off.
- **Fixed base.** It is based on release b11243 and does not include newer llama.cpp updates.
