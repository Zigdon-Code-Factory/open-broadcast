# Open Broadcast

A free, open-source alternative to NVIDIA Broadcast that runs entirely on CPU. No GPU, no PyTorch.

## Quick Start

**Example: use a noise-cleaned microphone in Discord or Zoom on Windows.**

1. Download and run the Windows `*-setup.exe` from [Releases](https://github.com/Zigdon-Code-Factory/open-broadcast/releases/latest). For an unpublished build, download `release-binaries-windows-x64` from a successful [Release workflow run](https://github.com/Zigdon-Code-Factory/open-broadcast/actions/workflows/release.yml), extract the artifact ZIP, and run the installer inside.
2. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) separately, restart if its installer requests it, then open **Open Broadcast**. The virtual audio driver is not bundled with the app.
3. For microphone-only use, uncheck **Camera**. Under **Microphone**, choose your physical/headset microphone as **Input** and **CABLE Input** as **Output**. Use **Refresh devices** if the cable is not listed.
4. Enable **Noise removal (DeepFilterNet3)**, optionally enable **Mute when not speaking (Silero VAD)**, and click **Start**. Speak and check that the level meter responds.
5. In Discord or Zoom, select **CABLE Output** as your microphone and use that app's microphone test. Keep Open Broadcast running; closing its control panel leaves it in the system tray.

**Routing:** your microphone → Open Broadcast → **CABLE Input** → **CABLE Output** → Discord/Zoom. The two cable names refer to opposite ends of the same virtual cable.

For macOS/Linux builds or camera output, see [Install](#install) and [Virtual devices](#virtual-devices). To run from source, see [Develop](#develop).

## Features

- 🎙️ **AI Noise Suppression**: DeepFilterNet3 (full-band 48 kHz, streaming ONNX, about 0.4 ms per 10 ms frame)
- 🛡️ **Voice Activity Detection**: Silero VAD gates the output during silence
- 🖼️ **Background Blur / Replace / Remove**: MediaPipe selfie segmenter
- 📐 **Auto Frame**: face tracking and centering (MediaPipe face detector)
- 📷 **Virtual Camera**: feeds Zoom, Discord, Meet, OBS and other apps
- 🎛️ **Tray app + control panel**: device pickers, live preview, mic level meter
- 🔄 **Auto-updates**: the Windows installer build updates itself from GitHub Releases
- 🪟🍎🐧 **Cross-platform**: Windows, macOS (Apple Silicon), Linux

## Install

Download the latest build from [Releases](https://github.com/Zigdon-Code-Factory/open-broadcast/releases/latest):

| Platform | File | Notes |
|---|---|---|
| Windows | `OpenBroadcast-<v>-windows-x64-setup.exe` | Per-user install, no admin needed. **Auto-updates.** |
| Windows | `OpenBroadcast-<v>-windows-x64-portable.zip` | Unzip and run `OpenBroadcast.exe`. Checks for updates but doesn't self-install. |
| macOS | `OpenBroadcast-<v>-macos-arm64.dmg` | Ad-hoc signed. On first launch, right-click the app and choose **Open**. |
| Linux | `OpenBroadcast-<v>-linux-x64.tar.gz` | Extract and run `./OpenBroadcast`. |

All models are bundled, so the app works offline. Each release includes `SHA256SUMS.txt`.

### Virtual devices

Open Broadcast sends its output to virtual devices that other apps pick up as a mic or camera:

| | Virtual mic | Virtual camera |
|---|---|---|
| **Windows** | [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (picked automatically if installed) | [OBS Studio](https://obsproject.com/) 26+ (provides OBS Virtual Camera) or [Unity Capture](https://github.com/schellingb/UnityCapture) |
| **macOS** | [BlackHole](https://github.com/ExistentialAudio/BlackHole) | OBS Studio 30+ (start the virtual camera once in OBS) |
| **Linux** | `pactl load-module module-null-sink sink_name=OpenBroadcast` | `sudo modprobe v4l2loopback devices=1 card_label="Open Broadcast" exclusive_caps=1` |

In Zoom, Discord and similar apps, pick **CABLE Output** (Windows) as the microphone and **OBS Virtual Camera** as the camera.

## Updates

- The app checks GitHub Releases for a newer version at startup and every 6 hours after that. You can also check manually from the tray menu (**Check for Updates…**) or from **Check now** in the control panel. Turn off **Check for updates automatically** in the control panel to disable automatic checks.
- **Windows installer build**: click **Update and restart** in the control panel or tray menu. One click checks GitHub for a newer release, downloads its installer, verifies its SHA-256 against `SHA256SUMS.txt`, closes the app, upgrades the existing installation and relaunches. Your settings are kept. If there is no newer release, the app reports that you are up to date. The button is disabled while checking or downloading; downloads can be cancelled before installation.
- Automatic background checks still ask before installing. The existing **Check now** / **Install and restart** confirmation flow also remains available.
- **Portable, macOS and Linux builds**: the prompt opens the release page instead.
- Pre-release tags such as `v1.2.0-rc1` are never offered as updates.

## Develop

Requires Python 3.10+ (3.12 recommended, which is what releases use).

```bash
git clone https://github.com/Zigdon-Code-Factory/open-broadcast.git
cd open-broadcast
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,build]"
python scripts/fetch_models.py    # downloads models (SHA-256 pinned) into src/open_broadcast/models/
open-broadcast                    # or: python -m open_broadcast
```

On Linux, also install `libportaudio2` and the graphics/Qt runtime libraries (`libegl1 libgles2 libxkbcommon0 libxcb-cursor0`), including for CPU-only processing.

```bash
pytest                 # unit tests (model-dependent tests are skipped if models aren't fetched)
ruff check src tests scripts
open-broadcast --self-test   # loads every model and native library, headless
```

Useful flags: `--minimized` (start in the tray), `--start` (start processing immediately), `--no-update-check`, `-v` (debug logging).

### Audio diagnostics

For real recorded examples, install `pip install -e ".[audio-test]"` and run `python scripts/audio_examples.py`. It downloads three LibriSpeech speakers and three ESC-10 environmental clips, then creates 18 noisy/processed pairs at 0 and 10 dB SNR in `build/audio-tests/online/`. Original files, source URLs, licenses, checksums and mix settings are preserved alongside the examples. These are controlled mixtures of real recordings, not recordings of speech captured in those environments.

Run `pytest tests/test_audio.py tests/test_audio_streaming.py -v` for model, streaming, gate, callback and recovery tests.

Run `python scripts/audio_diagnostics.py` for synthetic noise suppression and processing-time measurements. Add `--speech-file path/to/speech.wav` for a 16-bit PCM speech sample, or `--device-test` for a VB-CABLE loopback test. The device test sends a quiet synthetic signal through the virtual cable only; it does not open a physical microphone or intentionally play through speakers.

Results are written to `build/audio-tests/report.json`. The optional speech test also writes noisy and processed WAV files there for listening. Synthetic signal checks do not replace subjective speech-quality testing. Processing-time measurements exclude device/driver latency; the loopback test reports that separately.

User data locations:

| | Config (`config.json`) | Logs |
|---|---|---|
| Windows | `%LOCALAPPDATA%\OpenBroadcast\config` | `%LOCALAPPDATA%\OpenBroadcast\logs` |
| macOS | `~/Library/Application Support/OpenBroadcast` | `~/Library/Logs/OpenBroadcast` |
| Linux | `~/.config/open-broadcast` | `~/.local/state/open-broadcast` |

### Build a release locally

```bash
python scripts/build.py
```

This fetches models, runs PyInstaller with [packaging/open_broadcast.spec](packaging/open_broadcast.spec), and writes the platform packages plus `SHA256SUMS.txt` to `dist/release/`. On Windows, the installer step needs [Inno Setup 6](https://jrsoftware.org/isdl.php) (`winget install JRSoftware.InnoSetup`). Without it, only the portable zip is built.

### Cut a release

1. Bump `__version__` in [src/open_broadcast/__init__.py](src/open_broadcast/__init__.py).
2. Add matching notes at `docs/releases/<version>.md`, beginning with `# Open Broadcast <version>`. See [the 0.1.0 notes](docs/releases/0.1.0.md) for the download list and known limitations.
3. Run `python scripts/release_notes.py` to validate the notes and generate `build/release-notes.md` for review.
4. Commit, then tag and push:
   ```bash
   git tag v0.2.0 && git push origin master v0.2.0
   ```
5. [.github/workflows/release.yml](.github/workflows/release.yml) then:
   - builds and tests on Windows, macOS and Linux
   - runs `--self-test` against each frozen app
   - publishes a GitHub Release with the curated notes, every artifact and `SHA256SUMS.txt`

   The workflow fails if the tag doesn't match `__version__`, the release notes are missing, or the Windows installer compiler is unavailable.

To test without publishing a release, push the commit and run `gh workflow run release.yml --ref master`, or run **Release** manually from the Actions tab. The workflow uploads separate binary artifacts for all three platforms and a `release-notes-preview` artifact. This does not create a tag or a GitHub Release.

## Audio pipeline

```
Mic ──▶ mono 48 kHz ──▶ DeepFilterNet3 (512-sample hops) ──▶ speech gate ──▶ virtual mic
               └──▶ 16 kHz ──▶ Silero VAD ──────────────────────┘
```

DeepFilterNet adds about 10.7 ms of algorithmic latency. The VAD-driven gate uses a fast attack and a 300 ms holdover, so the ends of sentences aren't cut off.

## Models

- **[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet)**: streaming ONNX export from [grazder/DeepFilterNet](https://github.com/grazder/DeepFilterNet/tree/torchDF_main). *MIT / Apache-2.0*
- **[Silero VAD](https://github.com/snakers4/silero-vad)**: *MIT*
- **[MediaPipe Selfie Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)** and **[Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)**: *Apache-2.0*

## License

MIT
