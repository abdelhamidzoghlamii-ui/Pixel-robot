# Prototype evidence — historical STATUS sections

The four sections below were copied verbatim from `docs/STATUS.md` at
`905cb3f`, before shortening the live status. Their observations retain
the original dates and uncertainty. Check STATUS and later decisions for current
state.

## Thermal governance (reported 2026-09-11, battery power, Pixel 7 "panther", MP1.0, Android 17) — see DECISIONS #90-93

Device thermal governance was read from /vendor/etc/thermal_info_config.json
(three files present: base, _charge, _proto) and cross-checked against
`dumpsys thermalservice`, as reported this session — raw dump not yet attached.
AOSP panther source differs from this device; the device file takes precedence
pending independent spot-check.

Sensor thresholds as reported [LIGHT/MODERATE/SEVERE/CRITICAL/EMERGENCY/SHUTDOWN]:
  VIRTUAL-SKIN          [NAN, 39.0, 43.0, 45.0, 46.5, 52.0, 55.0]  Poll=300000 Passive=7000
  VIRTUAL-SKIN-CPU-GPU  [NAN, 37.0, 43.0, 45.0, 46.5, 52.0, 55.0]  Poll=300000 Passive=7000
  VIRTUAL-SKIN-CHARGE   [NAN, 35.0, 41.0, 45.0, 47.0, 51.0, 55.0]  Poll=300000 Passive=7000
  BIG                   [NAN x 7]  — no severity at any level on this device
  battery               [NAN x6, 60.0]

Key consequences (DECISIONS #90):
- BIG (thermal_zone9) has no HAL throttling severity on this device. Its only
  limits are kernel trips: active 20/55/80, PASSIVE 100, active 104/106/110,
  hot 120 °C (DECISIONS #74).
- VIRTUAL-SKIN-CPU-GPU (first trip 37.0 °C) is the sensor bound to the cpufreq
  PID. It is Type=UNKNOWN, so it does not appear in `Thermal Status:` — the
  phone can be actively throttling while dumpsys reports status 0.
- VIRTUAL-SKIN is not a kernel thermal zone; it is HAL-computed.

VIRTUAL-SKIN formula, device coefficients as reported this session (DECISIONS #91):
  a = 0.7*quiet_therm   + 0.3*qi_therm    - 500
  b = 0.58*usb_pwr_therm+ 0.42*quiet_therm- 500
  c = 1.1*quiet_therm   - 0.1*disp_therm  - 1500
  d = 0.3*neutral_therm + 0.7*quiet_therm - 500
  VIRTUAL-SKIN = max(a,b,c,d) / 1000   (inputs in millidegrees)

Thermal zone map, as reported this session — battery zone number unconfirmed,
pending direct read (see Pending):
  BIG=9  MID=10  LITTLE=11  G3D=12  TPU=14  neutral_therm=16  quiet_therm=17
  qi_therm=18  usb_pwr_therm=19  usb_pwr_therm2=20  disp_therm=21

Throttle detection (ground truth, sysfs — HAL cooling-device dump is empty):
  cooling devices: gxp-cooling, thermal-cpufreq-0/1/2, thermal-gpufreq-0, tpu_cooling
  cpufreq hw max:  policy0=1803000  policy4=2348000  policy6=2850000
  Throttling = any cur_state > 0, or scaling_max_freq < cpuinfo_max_freq.

Reported idle baseline (servers stopped, battery, 36 samples over 3 min):
  skin 26.73-27.94 °C (stable);  BIG 31.0-40.0 °C (swings ~9 °C, meaningless)
  cooling devices 0/36;  cpufreq capped 0/36
  Headroom to first throttle (37.0) approx 9-10 °C from cold idle.

`main.py: get_temp()` field-checked: one live call returned 41000 -> 41 °C. The
silent-zero branch (`except: return 0`, lines 26-27) remains in the code and was
not exercised by this check. The code path is directionally trustworthy; the
SENSOR CHOICE is wrong (reads BIG, not skin) per DECISIONS #90.

## Memory (reported 2026-09-11)

  MemTotal 7.29 GiB. Gemma-4-E2B q4_k_m resident: RSS 3.14-3.21 GB, PSS approx
  RSS (little sharing — an honest cost, not an mmap illusion).
  MemAvailable: approx 3.4 GB typical idle; approx 3.72 GB after app cleanup;
  approx 2.9-3.0 GB with Gemma loaded. Practical headroom with model loaded
  approx 0.5 GB. See DECISIONS #94 for the process-management ceiling on
  freeing more.

## Known defect: vision path is non-functional — confirmed by live call, 2026-09-11

No mmproj / multimodal projector file exists anywhere on device.
server_manager.py start_server() passes: -m --port --ctx-size --threads
--threads-batch --parallel 1 --swa-full --host. It does not pass --mmproj.

Confirmed live: a /completion request built exactly as gemma_decide() builds
it — image_data array plus an [img-1] prompt marker — returned HTTP 200 with
no error. stderr contains no mention of image_data, a vision/mmproj encoder,
or any image-related processing. prompt_n / tokens_evaluated = 310,
tokens_cached = 337 — consistent with the text prompt alone; the literal
string "[img-1]" was tokenized as ordinary text, not replaced by image
embeddings. The unsupported image_data field is silently dropped, not
rejected, and no error surfaces to the caller.

Consequence: every gemma_decide() call made with an image today is a
text-only call. The vision-triggered subset of consultations (every_5 /
person / goal / new_room, per the per-cycle flow above) carries no visual
information; those cycles run identically to text-only ones. See DECISIONS
#96.

Update (DECISIONS #105): projectors were sourced and vision tested live on
b2351. Not a fix — Gemma segfaults during image decode at usable budgets;
Qwen3.5-2B needs >=1024 image tokens to ground and takes ~125s/image with
hallucinated output. On-device LLM vision is a documented dead-end on this
hardware. Perception stays with YOLO.

## Last hardware test

**Power-up bring-up (bench, not yet driving).**
Battery pack 2× 18650 in series, cells balanced at 4.1 V / 4.1 V, pack 8.2 V.
BMS (HW-391 2S 20A) output steady 8.2 V. Buck converter set to 5.03 V no-load,
holding 5 V under ESP32 load. ESP32 powers up, 3V3 rail reads exactly 3.3 V on USB.
On buck power the 3V3 pin misreads ~3.76 V and the GPIO2 blue LED lights — both
traced to a marginal ground return on the buck path; fix is a solid star ground at
final assembly.

BMS output leg confirmed — P+/P− reads stable 8.17 V (⊕ = P+). Rail-to-star short
check clean (~1.35 kΩ, caps charging). Power distribution verified end to end.

**First motor spin achieved.** All four motors run continuously under PWM (1kHz,
8-bit, speed 130) with the ESP32 on USB/phone power and motors on the battery rail.
Pin map confirmed working (P16/17/18/19 → MX1508 #1 right side, P21/22/23/25 → MX1508 #2
left side — see DECISIONS #43 for corner mapping). Motors NOT yet verified with the ESP32 on buck power — no decoupling caps
fitted, and PWM rail collapse brownout-resets a buck-powered ESP32 (DECISIONS #38).
Remaining: fit caps, re-verify on buck power, then confirm direction sense per corner.

HC-SR04 verified on P27/P26 with 1kΩ/2kΩ divider on ECHO (DECISIONS #42).

Buck #2 (3.3 V sensor rail) remains unwired, per the hardware thread's as-built
diagram. Nothing currently depends on it — HC-SR04 draws from the ESP32 5V pin
(DECISIONS #42), not buck #2.
