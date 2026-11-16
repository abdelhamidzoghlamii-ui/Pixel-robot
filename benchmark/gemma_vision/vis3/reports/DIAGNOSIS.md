# Coder diagnosis claims to verify (read-only evidence; builds at /data/data/com.termux/files/home/llama.cpp-{b1609-dotprod,vis})
1. Build config equal: CMakeCache (Release, -O3, GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16, OPENMP ON, REPACK ON, LLAMAFILE ON,
   KLEIDIAI OFF, shared libs); differences only LLAMA_BUILD_IS_DEV (version string) and LLAMA_BUILD_UI. Clang 21.1.8 both.
   libggml-cpu.so: sdot 1045 (old) vs 1117 (new); fmla .8h 456 both; no i8mm (Tensor G2 has none).
2. Runtime defaults equal (help_*.txt here + VIS-1 V2 -lv 4 log): load mode mmap, flash_attn auto->enabled, n_batch 2048,
   n_ubatch 512, --fit on; load mode default mmap (OLD) vs auto=mmap on CPU (NEW, logged load_mode = mmap). The MTP draft
   thread-inheritance branch of common_base_params_to_speculative matches; NEW adds other behaviour in that function.
   b11716 removed --no-mmap/--mlock (use --load-mode); per-request speculative overrides are compiled out (#if 0 in
   tools/server/server-schema.cpp). b11716 logs "llama threadpool init" (explicit common_threadpools); OpenMP in both.
3. Prior OLD vs NEW numbers are thermally confounded: gate = idle+1.5 C with idle 27..36.7 C; VIS-1 T1 (old+MTP) started skin 39 C,
   X1 scaling_max 1.58 GHz, 7.07 tok/s; VIS-1 T2 (new+MTP) 32.5 C, 9.22; VIS-2 A (new+MTP) 34.2 C, 7.74; #137 (old+MTP) ~24 C, 19.18.
4. Shallow clone of llama.cpp-vis (1 commit, b11716 tag); /props build_info shows build 1.
5. Scope C (b11716 source): mmproj is read into allocated buffers (clip.cpp fin.read), not mmap -> resident until process exit;
   server_context_impl::destroy() frees target+draft+mmproj together; only whole-server unload exists (--sleep-idle-seconds, or
   router /models/load|unload which spawns child processes). Router mode is unavailable on this build: LLAMA_SUBPROCESS defaults
   OFF on Android ("subprocess is not enabled on this build"). Upstream PR #25963 (reload_context/mmproj/speculative) closed
   unmerged 2026-08-11; discussion #25674 open; draft PR #28346 (--mmproj-evict-draft, GPU VRAM swap) open; #30776 open.
