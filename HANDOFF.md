# Windows RTX handoff

## Current state

The test machine uses an RTX 5070 on Windows. The game is already available locally and is supplied at launch with `VICTORY_GAME_DIR`. Keep game data, caches, saves, and captures outside the runtime source trees.

| Runtime | Verified | Open |
| --- | --- | --- |
| CGBJ, Pareto Residency, Series S | Windows release build, zero missing imports, game window presenting at 30 FPS during startup | Gameplay frame rate and lighting on this GPU |
| Local VIctory RTX, Series S, DLSS Preset M | Title loads, DLSS recorded 1280×720 to 1920×1080 upscaling, 194 active GPU tests passed | Intro typically around 1–2 FPS, intermittent white frames and nearly black indoor gameplay |

## What changed

The VIctory snapshot includes the Streamline integration, Series S present-plane compositing, descriptor lifetime and table refresh fixes, NVIDIA format and Wave64 translation work, shadow-copy optimizations, and diagnostics. The current RTX working tree changes to `descriptor/host.rs`, `draw.rs`, and `memory.rs` are included in the snapshot.

`archive/victory-rtx-patches` contains 19 local commits. These are reference patches for the older work and show the source of individual changes. `archive/launchers` contains two older launcher versions that expect their original build filenames.

## Reproduce before changing the renderer

1. Build CGBJ with `Build-Windows.ps1 -Runtime CGBJ` and confirm the intro and the first movement prompt. Record a repeatable scene, actual gameplay FPS, draw count, and whether the interior renders.
2. Use the default logging settings for timing. Collect a separate diagnostic run for dropped shader hashes or device errors.
3. In the RTX build, compare `victory_gpu::host` frame preparation, copy volume, and GPU time at the same scene. One high-draw sample averaged about 176 ms in CPU copies with 1.2 GB copied per frame and 261 ms on the GPU.

The 262,144-slot descriptor reserve experiment removed slot-exhaustion warnings during a test, but the room stayed dark and the sampled FPS was lower. That reserve change was reverted. A separate fog-pass skip still ran around 0.7–1.5 FPS in the intro. Its visual effect was unverified. Missing Wave64 shader support is a confirmed cause of dropped work on this GPU. The full cause of the black interior remains unresolved.

## Work in parallel

- Establish the CGBJ gameplay baseline before porting changes between the runtimes.
- For lighting, trace which passes write the scene color at the movement prompt and compare the dropped shader hashes. [Upstream rendering investigation](https://github.com/Marvisak/VIctory/issues/841)
- For performance, measure preparation and GPU time independently. Keep each proposed change in a separate commit so both collaborators can reproduce its effect on the same scene.

Share a short, redacted symptom and timings in GitHub. Keep raw logs and game-derived captures local.

## RTX 5070 performance follow-up

The controlled CGBJ comparison and remaining GPU-memory bottleneck are recorded in [CGBJ-PERFORMANCE-RTX5070.md](CGBJ-PERFORMANCE-RTX5070.md). The warmed indoor scene remains around 5 FPS; the two texture settings tested did not improve it. Keep the launcher defaults until a measured change wins.

## DualShock 4 input follow-up

The CGBJ `gameinput` host now has an opt-in Windows joystick slot. `Play-CGBJ.cmd` selects joystick index 0 for the connected Sony DualShock 4 USB device. A Windows capture observed the face buttons, shoulders, D-pad, both trigger axes, and both stick axes. The focused conversion test passed; a release build and a brief title launch logged `Windows joystick connection changed id=0 connected=true`. Full in-game button and movement behavior still needs a hands-on check. WinMM provides no rumble in this implementation.

The latest Task Manager capture showed 100% 3D utilization, 11.4 of 12.0 GB dedicated GPU memory, 6.0 of 7.9 GB shared GPU memory, and 14.8 of 15.8 GB system RAM. The GPU was 45 C. These readings support GPU and memory pressure as a performance concern; they do not identify a single fix or establish that temperature is limiting FPS.
