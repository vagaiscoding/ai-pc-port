# CGBJ performance on RTX 5070 (2026-11-26)

## Reproduction

- CGBJ source: `f4b48ad766ddbec1a1867022cd42f7645b0dd55e` (`DoubleGhosty/cool_fox` mirror).
- Windows release build, Series S profile, Pareto Residency launcher, warmed shader and pipeline cache.
- GPU: GeForce RTX 5070, 12 GiB dedicated memory. The same intro and early indoor sequence rendered with visible characters, textures, and lighting.
- Each sample is one runtime frame-rate interval with 4,500–6,500 draws per frame. The first three matching intervals of each run were excluded from the steady median because scene entry was still loading.

| Setting | Steady samples | Median FPS | Result |
| --- | ---: | ---: | --- |
| Launcher default | 45 | 4.9 | Best of these runs |
| `VICTORY_TEXTURE_KEEP=0` | 15 | 4.8 | No measurable improvement |
| `VICTORY_UPLOAD_AHEAD=0` | 15 | 4.5 | Slower; the scene still rendered |

These are sequential playthroughs, not synchronized frame replays. Small differences are not evidence of a speedup. The 30 FPS shown in menus and loading screens occurred at fewer than 12 draws per frame and does not represent the 3D scene.

## Bottleneck evidence

- In the warmed default run, a representative interval had 4.6 FPS, 218.6 ms/frame and 5,977 draws/frame: 59.9 ms preparing, 145.7 ms waiting for the GPU, and 37.0 ms copying results back. The GPU and CPU work overlap, so these terms should not be added to predict frame time.
- The RTX 5070 reached 10,504 MiB local GPU memory use against a budget of 10,640 MiB, plus 5,989 MiB of shared GPU memory in one heavy interval. `nvidia-smi` showed roughly 11,800 MiB in use and 99–100% utilization while the scene ran. This is consistent with severe memory pressure, though it does not isolate which resource caused the GPU wait.
- CGBJ reported host physical memory pressure with about 2–3 GiB free. Its README recommends 64 GiB RAM for its tested build, and its published prologue timing on a different machine is 13–20 FPS, so 30 FPS is not a verified CGBJ baseline.
- Pipeline warming completed 2,282 known pipelines beforehand. Heavy frames still dropped roughly 300 draws without a host pipeline, with zero pipelines being compiled in those intervals. FPS and rendering completeness are separate open problems.
- `VICTORY_CARD_STEPS` reports about 10,000 descriptor fills per heavy frame, but its implementation deduplicates source reads before recording GPU copies. That counter is not 10,000 individual GPU copy commands.

## Next engineering work

1. Profile a repeatable heavy scene with GPUView or a frame capture that does not trigger the title watchdog. Distinguish paging, draw/compute cost, and queue waits before changing GPU scheduling.
2. Inventory live VRAM by resource class and descriptor references under pressure. The idle texture sweep cannot evict textures still held by descriptor slots; any new eviction must repopulate those slots before use to preserve rendering.
3. Compare one candidate change at a time against the default on the same save, with draw count, FPS, local/shared GPU memory, and a rendered-frame check. Keep the user-facing launcher on the verified default until a candidate wins.

Raw logs, saves, captures, and game files stay local. No 30 FPS fix is claimed here.
