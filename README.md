# ai-pc-port

Windows compatibility-layer work for an Xbox Series title. Start with the CGBJ build to reproduce the latest successful launch. The separate VIctory RTX build contains the earlier DLSS, rendering, and performance work.

## Build on Windows

Install Rust through rustup and the Visual Studio C++ build tools with a Windows SDK. Both snapshots pin Rust 1.98.1. From the repository root in PowerShell:

```powershell
.\Build-Windows.ps1
```

Use `-Runtime CGBJ` or `-Runtime RTX` to build one runtime. The build script obtains the upstream BCPack format table and verifies its SHA-256 before compiling. With Cargo dependencies already cached, use `-Offline -BcpackTables <path>` to build with a local copy of that table.

## Run

Set `VICTORY_GAME_DIR` to your own game directory containing `game_scarlett_master.exe`.

```powershell
$env:VICTORY_GAME_DIR = 'D:\Games\YourDump'
.\Play-CGBJ.cmd
```

`Play-CGBJ.cmd` uses the upstream Pareto Residency preset and Series S. The CGBJ and RTX launchers use separate local caches and save folders.

- `VIctory-RTX-GPU-Fix.cmd --check` checks the title's imports without running the game. Omit `--check` to play.
- `VIctory-DLSS-Performance.cmd` uses DLSS Preset M on the Series S profile. Set `VICTORY_DLSS_DIR` to the `bin\x64` folder of your own NVIDIA Streamline 2.14.16 SDK installation.
- `VIctory-DLSS-4.5.cmd` retains the earlier DLSS diagnostic launcher. Both DLSS launchers support `--check`, `--gpu-log`, and `--full-log`. Full logging slows the run.

The [handoff](HANDOFF.md) records what has been tested, what is still broken, and where to start. Game files, local saves, graphics SDK binaries, and runtime logs stay on the machine that produced them.

## Source and licensing

- `runtime/victory-rtx` is the [VIctory](https://github.com/Marvisak/VIctory) RTX snapshot, built from the `940d32f` baseline with local changes. The patch archive preserves the earlier local commits, including reverted experiments.
- `runtime/cgbj` is the [CGBJ Windows snapshot](https://github.com/DoubleGhosty/cool_fox/commit/f4b48ad766ddbec1a1867022cd42f7645b0dd55e). The launcher and build integration live at the repository root.
- Each runtime includes its upstream GPL-3.0 license. Upstream source comments, notices, and documentation are preserved. The upstream BCPack table is obtained separately during the build.
