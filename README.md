# terminal-visualizer

A real-time terminal audio spectrum analyzer with a rainbow LED display.

![spectrum analyzer screenshot](https://github.com/jacksonrdlc/terminal-visualizer/raw/main/screenshot.png)

## Features

- Full rainbow spectrum (red → orange → yellow → green → blue → purple) across frequency bands
- Classic LED segment style with peak hold markers
- Captures system audio via BlackHole — visualizes whatever your computer is playing
- Smooth 30 FPS with delta rendering and vectorized numpy — low CPU and memory usage

## Install

```bash
pip install sounddevice numpy
```

## macOS System Audio Setup

To visualize music playing through your speakers (not just the microphone):

1. Install BlackHole virtual audio device:
   ```bash
   brew install blackhole-2ch
   ```

2. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
   - Click `+` → **Create Multi-Output Device**
   - Check both **BlackHole 2ch** and your output device (e.g. Scarlett, MacBook Speakers, headphones)

3. Go to **System Settings → Sound → Output** and select the **Multi-Output Device**

4. Run the visualizer:
   ```bash
   python3 sound_viz.py --device "BlackHole 2ch"
   ```

Audio will play through your speakers and the visualizer will react to it simultaneously.

## Usage

```bash
# List available input devices
python3 sound_viz.py --list

# Use a specific device by name
python3 sound_viz.py --device "BlackHole 2ch"

# Use a specific device by index
python3 sound_viz.py --device 2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | system default | Device index or name substring |
| `--bands` | 24 | Number of frequency bars |
| `--smooth` | 0.75 | Temporal smoothing (0 = raw, 0.99 = very slow) |
| `--sens` | 4.5 | Amplitude sensitivity multiplier |
| `--peak-decay` | 0.014 | Speed at which peak markers fall per frame |

Press `Ctrl+C` to quit.
