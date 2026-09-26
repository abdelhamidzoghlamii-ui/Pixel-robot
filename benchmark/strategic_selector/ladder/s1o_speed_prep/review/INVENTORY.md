# Inventory (2026-09-26, Pixel 6 Tensor G1: 4x A55 0xd05, 2x A78 0xd41, 2x X1 0xd44)
/proc/cpuinfo Features: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
-> dotprod YES, i8mm NO, SVE NO.

## llama.cpp builds
| build | path | commit | runs | ggml-cpu flags | runtime system_info |
|---|---|---|---|---|---|
| 1609 | /termux-home/llama.cpp/build/bin | e1a1abb7 (tag b10194, 2026-07-30) | yes | GGML_NATIVE=OFF, GGML_CPU_ARM_ARCH empty -> no -mcpu/-march (baseline armv8-a); OpenMP ON; REPACK ON; LLAMAFILE ON; KleidiAI OFF | NEON, ARM_FMA, LLAMAFILE, OPENMP, REPACK (no DOTPROD, no FP16_VA); 608 tensors "cannot be used with CPU_REPACK" |
| 2351 | /termux-home/llama.cpp-upstream/build/bin | 790cf51a (b10935-1, 2026-09-12) | yes | GGML_NATIVE=ON -> -mcpu=native+dotprod+noi8mm+nosve+nosme; OpenMP OFF; REPACK ON; LLAMAFILE ON; KleidiAI OFF | NEON, ARM_FMA, FP16_VA, DOTPROD, LLAMAFILE, REPACK; Q4_K_M Gemma: CPU_REPACK 1422 MiB |
Also: Termux python llama_cpp_python 0.3.19 (site-packages/llama_cpp/lib, not used by s1o); whisper.cpp ggml (not llama).
Both run flash attention by default (-fa auto -> "Flash Attention enabled").
b2351 FA BUG: llama-server segfaults (exit 139) on Gemma 4 E2B prompts >= 64 tokens with --swa-full (>= 70 without);
independent of --no-repack/--cache-ram/n_probs; with -fa off all lengths pass. b2351 variants therefore use --flash-attn off.
Build "numbers" 1609/2351 are the local builds' --version counters, not upstream bNNNN tags.

## GGUFs (SHA-256)
cded614c9b24be92e5a868d2ba38fb24e15dfea34fc650193c475a6debc233a7 3462677760 models/gemma-4-e2b-it-q4_k_m.gguf (s1o current)
8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52 2841481184 models/gemma-4-E2B-it-Q4_0.gguf  NEW: ggml-org/gemma-4-E2B-it-GGUF @ b4243c156154b6dca9324415f8c7ccc098b4aed1, LFS sha256 matched
12d878964d21f1779dea15abeee048855151b27089fe98b32c628f85740933f3 4967490208 models/gemma-4-e2b-it-q8_0.gguf
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855          0 models/gemma-4-e4b-it-q3_k_m.gguf (EMPTY file)
dff4e4ca848e33e678a63b5b7d1f8bfa4a17e764415d0c0aaaad07c84f4d8fad 5335285440 models/gemma-4-e4b-it-q4_k_m.gguf
f36cbf8236a5c2ed0ad1c3606c2f16f30cd5b67914dfd4000eff1f9138663bf3   26168672 models/gemma4-nav-lora.gguf (= /sdcard copy)
850eaaee46735793a5433eea3de97931cf56e39d0579a823635af36856966b1d   26168768 models/gemma4-voice-lora.gguf (= /sdcard copy)
3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9 4368439584 models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292  557368064 models/mmproj-gemma-4-E2B-it-Q8_0.gguf
e9b34d45e01e81c5b92744a482ed02b197f6aa9dbb66fa94f7ce3b4b435d7154  335790368 models/mmproj-gemma-4-e2b-it-q4_0.gguf (vision projector, not a Q4_0 LM)
d460bb51ce5115232a723a7366694c8ee70d8aa8f60159d182a1da0777d10db3  986833408 models/mmproj-google_gemma-4-E2B-it-bf16.gguf
6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e 1117320736 models/qwen2.5-1.5b-instruct-q4_k_m.gguf
626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d 2104932768 models/qwen2.5-3b-instruct-q4_k_m.gguf
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  532517120 models/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf
aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223 1280835840 models/qwen35/Qwen3.5-2B-Q4_K_M.gguf
00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4 2740937888 models/qwen35/Qwen3.5-4B-Q4_K_M.gguf
8814232b85594dcd46c50e5b8b29324a7efe9e746edbe8a3d1df3d3fce7aad39 3143656608 models/qwen35/Qwen3.5-4B-Q5_K_M.gguf
fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66 3525956768 models/qwen35/Qwen3.5-4B-Q6_K.gguf
7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7  668227264 models/qwen35/mmproj-F16.gguf
56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453  204987232 models/qwen35/mmproj-Qwen3.5-0.8B-F16.gguf
cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864  672423616 models/qwen35/mmproj-Qwen3.5-4B-F16.gguf
(+ ggml-vocab-*.gguf tokenizer test fixtures under each llama.cpp/models/, not models)
