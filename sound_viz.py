#!/usr/bin/env python3
"""
sound-viz  —  terminal audio visualizer
LED segment spectrum analyzer — green/yellow/red, full-width, with peak hold

Install:  pip install sounddevice numpy
Run:      python3 sound_viz.py
          python3 sound_viz.py --list         # list audio devices
          python3 sound_viz.py --device N     # use device number N

macOS system audio (hear what your computer plays):
  Install BlackHole virtual audio device:
    brew install blackhole-2ch
  Then in Audio MIDI Setup.app:
    • Create a "Multi-Output Device" containing your speakers + BlackHole 2ch
    • Set that Multi-Output Device as your system output
  Run: python3 sound_viz.py --device "BlackHole 2ch"
"""

import sys, os, time, math, signal, argparse, threading
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sys.exit("Missing dep — run:  pip install sounddevice numpy")

# ──────────────────────────── config ─────────────────────────────────────────
SAMPLE_RATE  = 44100
CHUNK        = 2048
N_BANDS      = 24           # frequency bars (auto-scales to terminal width)
SMOOTH       = 0.75         # temporal smoothing (0=raw, 0.99=very laggy)
SENS         = 4.5          # amplitude multiplier before clipping
MIN_HZ       = 20
MAX_HZ       = 16_000
FPS          = 30
BG_COLOR     = (6, 6, 6)    # near-black background
PEAK_DECAY   = 0.014        # fraction of full height peak falls per frame

# LED segment dimensions (in pixel rows; each terminal row = 2 pixel rows via ▀)
LED_LIT_PX   = 2            # lit portion of each segment
LED_GAP_PX   = 2            # dark gap below each segment
LED_SLOT     = LED_LIT_PX + LED_GAP_PX   # 4 px total per slot

# ──────────────────────────── shared state ───────────────────────────────────
_amps     = np.zeros(N_BANDS)
_lock     = threading.Lock()
_alive    = True
_peak_run = 1e-9

# Pre-allocated pixel buffers and layout cache (rebuilt on terminal resize)
_buf_f32     = None   # float32 working buffer
_buf_u8      = None   # uint8 output buffer
_prev_buf    = None   # previous frame for delta rendering
_band_map    = None   # int32 (w,): column → band index, -1 = gap
_lit_rgb     = None   # float32 (total_seg, 3): bright colors per segment height
_dim_rgb     = None   # float32 (total_seg, 3): dim ghost colors
_peak_rgb    = None   # float32 (total_seg, 3): peak marker colors
_layout_key  = (0, 0, 0)  # (pr, w, n_bands) — triggers rebuild on change

# ──────────────────────────── audio ──────────────────────────────────────────
def audio_cb(indata, frames, t, status):
    global _peak_run
    mono  = indata[:, 0]
    win   = mono * np.hanning(len(mono))
    fft   = np.abs(np.fft.rfft(win, n=CHUNK))
    freqs = np.fft.rfftfreq(CHUNK, 1 / SAMPLE_RATE)

    edges = np.logspace(math.log10(max(MIN_HZ, 1)), math.log10(MAX_HZ), N_BANDS + 1)
    raw   = np.zeros(N_BANDS)
    for i in range(N_BANDS):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if m.any():
            raw[i] = np.sqrt(np.mean(fft[m] ** 2))

    _peak_run = max(_peak_run * 0.994, raw.max())
    if _peak_run > 1e-9:
        raw /= _peak_run
    raw = np.clip(raw * SENS, 0, 1)

    with _lock:
        _amps[:] = _amps * SMOOTH + raw * (1 - SMOOTH)


# ──────────────────────────── renderer ───────────────────────────────────────
def _seg_color_t(t: float):
    """
    Full visible spectrum left→right: red → orange → yellow → green → blue → purple.
    t=1.0 (left/low-freq) → red, t=0.0 (right/high-freq) → purple.
    """
    if t > 0.85:
        return (1.00, 0.05, 0.00)   # red
    elif t > 0.70:
        return (1.00, 0.42, 0.00)   # orange
    elif t > 0.55:
        return (0.95, 0.88, 0.00)   # yellow
    elif t > 0.38:
        return (0.05, 0.88, 0.12)   # green
    elif t > 0.20:
        return (0.00, 0.30, 1.00)   # blue
    else:
        return (0.62, 0.00, 1.00)   # purple/violet


def _rebuild_layout(pr: int, w: int, n_bands: int) -> None:
    """Pre-compute band_map and color LUTs. Called once per terminal resize."""
    global _band_map, _lit_rgb, _dim_rgb, _peak_rgb
    global _buf_f32, _buf_u8, _prev_buf, _layout_key

    bar_w     = w / n_bands
    gap       = 1 if bar_w >= 4.0 else 0
    total_seg = pr // LED_SLOT

    # Band map: which band does each pixel column belong to (-1 = gap/border)
    bm = np.full(w, -1, dtype=np.int32)
    for bi in range(n_bands):
        x0 = int(bi * bar_w)
        x1 = min(w, max(x0 + 1, int((bi + 1) * bar_w) - gap))
        bm[x0:x1] = bi
    _band_map = bm

    # Color LUTs per pixel column (based on band/frequency position, not height)
    lit  = np.zeros((w, 3), dtype=np.float32)
    dim  = np.zeros((w, 3), dtype=np.float32)
    peak = np.zeros((w, 3), dtype=np.float32)
    for col in range(w):
        bi = int(bm[col])
        if bi < 0:
            continue
        t = 1.0 - bi / max(1, n_bands - 1)   # 1 = leftmost (red), 0 = rightmost (blue)
        rc, gc, bc = _seg_color_t(t)
        lit[col]  = [rc * 255,        gc * 255,        bc * 255       ]
        dim[col]  = [rc * 255 * 0.07, gc * 255 * 0.07, bc * 255 * 0.07]
        peak[col] = [(rc + 0.45 * (1 - rc)) * 255,
                     (gc + 0.45 * (1 - gc)) * 255,
                     (bc + 0.45 * (1 - bc)) * 255]
    _lit_rgb  = lit
    _dim_rgb  = dim
    _peak_rgb = peak

    _buf_f32  = np.empty((pr, w, 3), dtype=np.float32)
    _buf_u8   = np.empty((pr, w, 3), dtype=np.uint8)
    _prev_buf = None   # force full redraw after resize
    _layout_key = (pr, w, n_bands)


def build_pixel_buf(w: int, h: int, amps: np.ndarray, peaks_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized renderer: iterate over segment heights (not bands), use numpy
    boolean masks to assign colors to all bars simultaneously.
    """
    global _layout_key

    pr = h * 2
    n  = len(amps)
    if _layout_key != (pr, w, n):
        _rebuild_layout(pr, w, n)

    buf       = _buf_f32
    buf[:]    = BG_COLOR
    total_seg = pr // LED_SLOT

    lit_counts = (amps     * total_seg      ).astype(np.int32)   # (n,)
    pk_segs    = (peaks_arr * (total_seg - 1)).astype(np.int32)  # (n,)

    bm      = _band_map                       # (w,)
    valid   = bm >= 0                         # (w,) — excludes gaps
    bm_safe = np.where(valid, bm, 0)          # avoid out-of-bounds index

    for si in range(total_seg):
        row = pr - (si + 1) * LED_SLOT
        if row < 0:
            continue
        seg_s = slice(row, min(pr, row + LED_LIT_PX))

        is_lit  = valid & (si <  lit_counts[bm_safe])
        is_peak = valid & (si == pk_segs[bm_safe]) & (peaks_arr[bm_safe] >= 0.02) & ~is_lit
        is_dim  = valid & ~is_lit & ~is_peak

        buf[seg_s, is_lit,  :] = _lit_rgb[is_lit]
        buf[seg_s, is_peak, :] = _peak_rgb[is_peak]
        buf[seg_s, is_dim,  :] = _dim_rgb[is_dim]

    np.clip(buf, 0, 255, out=buf)
    np.copyto(_buf_u8, buf, casting='unsafe')
    return _buf_u8


_UPPER_B = '▀'.encode()   # b'\xe2\x96\x80' — pre-encoded, reused every frame
_BG_ARR  = np.array(BG_COLOR, dtype=np.uint8)

def pixels_to_bytes(buf: np.ndarray) -> bytes:
    """
    Convert pixel buffer → raw bytes for sys.stdout.buffer.
    Optimisations:
      • Row-level delta: skip rows identical to previous frame entirely.
      • Run-length encoding: emit one escape code per color run, not per pixel.
        For LED-style bars each band is one solid-color run → ~N_BANDS iterations
        per changed row instead of ~W iterations.
      • bytearray accumulation avoids str allocation and UTF-8 encoding overhead.
    """
    global _prev_buf

    pr, w, _ = buf.shape
    h        = pr // 2
    full     = _prev_buf is None or _prev_buf.shape != buf.shape

    out = bytearray()

    for row in range(h):
        tp = row * 2
        bp = tp + 1

        # Row-level delta: skip if both pixel rows are identical to last frame
        if not full:
            if (np.array_equal(buf[tp], _prev_buf[tp]) and
                    (bp >= pr or np.array_equal(buf[bp], _prev_buf[bp]))):
                continue

        # Absolute cursor move to this terminal row (1-indexed)
        out.extend(f'\033[{row + 1};1H'.encode())

        fg = buf[tp].copy()
        bg = buf[bp].copy() if bp < pr else np.broadcast_to(_BG_ARR, (w, 3)).copy()

        # Clamp near-black to BG_COLOR
        fg[np.sum(fg, axis=1) < 8] = _BG_ARR
        bg[np.sum(bg, axis=1) < 8] = _BG_ARR

        # Find columns where fg or bg changes → run boundaries
        fg_diff    = np.any(fg[1:] != fg[:-1], axis=1)
        bg_diff    = np.any(bg[1:] != bg[:-1], axis=1)
        run_starts = np.concatenate([[0], np.where(fg_diff | bg_diff)[0] + 1, [w]])

        last_fg = last_bg = (-1, -1, -1)

        for i in range(len(run_starts) - 1):
            cs = int(run_starts[i])
            ce = int(run_starts[i + 1])

            tr, tg, tb = int(fg[cs, 0]), int(fg[cs, 1]), int(fg[cs, 2])
            br, bg2, bb = int(bg[cs, 0]), int(bg[cs, 1]), int(bg[cs, 2])

            if (tr, tg, tb) != last_fg:
                out.extend(f'\033[38;2;{tr};{tg};{tb}m'.encode())
                last_fg = (tr, tg, tb)
            if (br, bg2, bb) != last_bg:
                out.extend(f'\033[48;2;{br};{bg2};{bb}m'.encode())
                last_bg = (br, bg2, bb)

            out.extend(_UPPER_B * (ce - cs))

        out.extend(b'\033[0m')

    # Update delta cache
    if full:
        _prev_buf = buf.copy()
    else:
        np.copyto(_prev_buf, buf)

    return bytes(out)


# ──────────────────────────── main ───────────────────────────────────────────
def sig_handler(s, f):
    global _alive
    _alive = False


def parse_args():
    p = argparse.ArgumentParser(description='Terminal audio visualizer')
    p.add_argument('--list',       action='store_true', help='List audio input devices and exit')
    p.add_argument('--device',     default=None,        help='Device index or name substring')
    p.add_argument('--bands',      type=int,   default=N_BANDS,     help=f'Number of bands (default {N_BANDS})')
    p.add_argument('--smooth',     type=float, default=SMOOTH,      help=f'Smoothing 0-0.99 (default {SMOOTH})')
    p.add_argument('--sens',       type=float, default=SENS,        help=f'Sensitivity (default {SENS})')
    p.add_argument('--peak-decay', type=float, default=PEAK_DECAY,  help=f'Peak fall speed per frame (default {PEAK_DECAY})')
    return p.parse_args()


def resolve_device(spec):
    if spec is None:
        return None
    try:
        return int(spec)
    except ValueError:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if spec.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                return i
        sys.exit(f"Device not found: {spec!r}  (run --list to see options)")


def main():
    global _alive, N_BANDS, SMOOTH, SENS, PEAK_DECAY
    args = parse_args()

    if args.list:
        print("Input audio devices:\n")
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                marker = ' ◀ default' if i == sd.default.device[0] else ''
                print(f"  [{i:2d}]  {d['name']}{marker}")
        return

    N_BANDS    = args.bands
    SMOOTH     = args.smooth
    SENS       = args.sens
    PEAK_DECAY = args.peak_decay

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    device   = resolve_device(args.device)
    dev_info = sd.query_devices(device if device is not None else sd.default.device[0])

    sys.stdout.buffer.write(b'\033[2J\033[H\033[?25l')   # clear, hide cursor
    sys.stdout.buffer.flush()
    print(f"  sound-viz  |  {dev_info['name']}  |  Ctrl+C to quit\n")
    if 'blackhole' not in dev_info['name'].lower() and 'loopback' not in dev_info['name'].lower():
        print("  Tip: this is capturing the microphone.")
        print("  For system audio install BlackHole: brew install blackhole-2ch\n")
    time.sleep(1.5)

    stream = sd.InputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK,
        callback=audio_cb,
    )
    stream.start()

    frame_dt = 1.0 / FPS
    peaks    = np.zeros(N_BANDS)

    sys.stdout.buffer.write(b'\033[2J')
    sys.stdout.buffer.flush()

    try:
        while _alive:
            t0 = time.perf_counter()

            try:
                w, h = os.get_terminal_size()
            except OSError:
                w, h = 80, 24
            h = max(4, h - 1)

            with _lock:
                amps = _amps.copy()

            # Peak hold: instantly follow rising signal, slowly fall otherwise
            peaks = np.where(amps > peaks, amps, np.maximum(0.0, peaks - PEAK_DECAY))

            buf   = build_pixel_buf(w, h, amps, peaks)
            frame = pixels_to_bytes(buf)
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()

            elapsed = time.perf_counter() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)

    finally:
        stream.stop()
        stream.close()
        sys.stdout.buffer.write(b'\033[?25h\033[0m\033[2J\033[H')
        sys.stdout.buffer.flush()
        print("sound-viz stopped.")


if __name__ == '__main__':
    main()
