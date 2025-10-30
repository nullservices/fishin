#!/usr/bin/env python3
# fishin.py
# Motion-triggered fishing helper
# Two-tap F6 to pick region, F8 start/stop, F9 exit. Auto-loots with Auto Loot ON.

import time
import math
import random
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mss
import pyautogui
import keyboard

# ---------------------------- USER SETTINGS ----------------------------
CAST_KEY = '1'                   # fishing keybind
RUN_TOGGLE_KEY = 'f8'
SET_REGION_KEY = 'f6'            # two-tap: top-left then bottom-right
CANCEL_REGION_KEY = 'f10'
EXIT_KEY = 'f9'
DEBUG_TOGGLE_KEY = 'f7'
TEST_CENTER_CLICK_KEY = 'f2'
TEST_AUTOTARGET_CLICK_KEY = 'f3'

# Sensitivity (turn these up to reduce false clicks; down to be more sensitive)
FRAME_RATE = 20
BASELINE_SECONDS = 1.0
Z_THRESH = 5.0                   # spike vs per-cast baseline (stddev)
THRESH_MULT_ALT = 3.0            # fallback multiplier when stddev is tiny
AREA_MULT = 3.5                  # changed-pixel area multiplier vs baseline
AREA_MIN_ABS = 150               # absolute min area to accept
DIFF_PIXEL_MIN = 25              # per-pixel diff to count toward area (0..255)

# Color gate (focus motion where bobber colors usually are)
COLOR_MODE = 'auto'              # 'auto' | 'red' | 'blue' | 'off'
RED_MIN = 120
BLUE_MIN = 120
DOM_MARGIN = 40
SAT_MIN = 25

# Timing & mouse feel
LAND_BOBBER_DELAY = (1.1, 1.6)
POST_LOOT_DELAY = (0.8, 1.4)
RECAST_DELAY_RANGE = (0.6, 1.2)
RIGHT_CLICK_DELAY = (0.05, 0.12)
JITTER_PIXELS = 2

# Cursor parking (keep mouse just outside the watch box; only once per cycle)
PARK_MOUSE = True
PARK_BEFORE_CAST = True          # leave this True
PARK_AFTER_LOOT = False          # and this False so it only parks once
SAFE_INSET = 12
PARK_MIN_DIST = 40               # min px from box edge (used by edge function below)
PARK_EDGE_OFFSET = (40, 80)      # how far outside the box (min, max)

# Region sanity
MIN_REGION_W = 80
MIN_REGION_H = 80
EDGE_MARGIN = 4
# ----------------------------------------------------------------------

pyautogui.FAILSAFE = True

@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

current_region: Optional[Region] = None
running = False
stop_flag = False
debug_on = False

_region_stage = 0
_region_first: Optional[Tuple[int, int]] = None

# ---------------------------- helpers ----------------------------
def rnd(a: float, b: float) -> float:
    return random.uniform(a, b)

def log(msg: str):
    print(msg, flush=True)

def human_move_to(x: int, y: int, dur_range=(0.06, 0.12), wobble_px=2):
    """Quicker smooth move with easing."""
    sx, sy = pyautogui.position()
    ex = x + random.randint(-wobble_px, wobble_px)
    ey = y + random.randint(-wobble_px, wobble_px)
    duration = rnd(*dur_range)
    steps = max(6, int(duration * 90))
    for i in range(steps + 1):
        t = i / steps
        ease = 3*t*t - 2*t*t*t  # cubic ease-in-out
        cx = sx + (ex - sx) * ease
        cy = sy + (ey - sy) * ease
        pyautogui.moveTo(cx, cy)
        time.sleep(duration / steps)

def human_right_click(x: int, y: int):
    human_move_to(x, y, dur_range=(0.06, 0.12), wobble_px=JITTER_PIXELS)
    time.sleep(rnd(*RIGHT_CLICK_DELAY))
    pyautogui.click(button='right')

def cast():
    pyautogui.press(CAST_KEY)

def point_outside_region(reg: Region, offset_range=None) -> tuple[int, int]:
    """Pick a point just outside the outlined region."""
    if offset_range is None:
        offset_range = PARK_EDGE_OFFSET
    side = random.choice(['top', 'bottom', 'left', 'right'])
    offset = random.randint(*offset_range)
    if side == 'top':
        x = random.randint(reg.left, reg.left + reg.width)
        y = reg.top - offset
    elif side == 'bottom':
        x = random.randint(reg.left, reg.left + reg.width)
        y = reg.top + reg.height + offset
    elif side == 'left':
        x = reg.left - offset
        y = random.randint(reg.top, reg.top + reg.height)
    else:
        x = reg.left + reg.width + offset
        y = random.randint(reg.top, reg.top + reg.height)

    # keep within screen safe inset
    sw, sh = pyautogui.size()
    x = int(max(SAFE_INSET, min(sw - SAFE_INSET, x)))
    y = int(max(SAFE_INSET, min(sh - SAFE_INSET, y)))
    return x, y

# ---------------------------- region pick ----------------------------
def set_region_interactively():
    global _region_stage, _region_first, current_region
    if _region_stage == 0:
        _region_first = pyautogui.position()
        _region_stage = 1
        log("[region] top-left saved, move to bottom-right and press F6 again")
        return
    x2, y2 = pyautogui.position()
    x1, y1 = _region_first
    _region_stage = 0
    _region_first = None

    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    if width < MIN_REGION_W or height < MIN_REGION_H:
        log(f"[region] too small ({width}x{height}), try again")
        return
    current_region = Region(left, top, width, height)
    log(f"[region] set: {left},{top} {width}x{height}")

def cancel_region():
    global _region_stage, _region_first
    _region_stage = 0
    _region_first = None
    log("[region] selection canceled")

# ---------------------------- capture & metrics ----------------------------
def grab_bgr(reg: Region, sct: mss.mss) -> np.ndarray:
    shot = sct.grab({"left": reg.left, "top": reg.top, "width": reg.width, "height": reg.height})
    return np.asarray(shot)[:, :, :3]  # BGRA -> BGR

def to_gray(bgr: np.ndarray) -> np.ndarray:
    gray = (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]).astype(np.uint8)
    if EDGE_MARGIN > 0:
        gray[:EDGE_MARGIN, :] = gray[EDGE_MARGIN, :]
        gray[-EDGE_MARGIN:, :] = gray[-EDGE_MARGIN-1, :]
        gray[:, :EDGE_MARGIN] = gray[:, EDGE_MARGIN][:, None]
        gray[:, -EDGE_MARGIN:] = gray[:, -EDGE_MARGIN-1][:, None]
    return gray

def red_mask(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    dom = r - np.maximum(b, g)
    sat = r - ((b + g) // 2)
    return (r >= RED_MIN) & (dom >= DOM_MARGIN) & (sat >= SAT_MIN)

def blue_mask(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    dom = b - np.maximum(r, g)
    sat = b - ((r + g) // 2)
    return (b >= BLUE_MIN) & (dom >= DOM_MARGIN) & (sat >= SAT_MIN)

def auto_color_mask(bgr: np.ndarray) -> np.ndarray:
    rm = red_mask(bgr)
    bm = blue_mask(bgr)
    rc, bc = int(rm.sum()), int(bm.sum())
    if rc == 0 and bc == 0:
        return rm | bm
    return rm if rc >= bc else bm

def apply_color_gate(diff_u8: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    if COLOR_MODE == 'off':
        return diff_u8
    mask = red_mask(bgr) if COLOR_MODE == 'red' else blue_mask(bgr) if COLOR_MODE == 'blue' else auto_color_mask(bgr)
    return np.where(mask, diff_u8, 0).astype(np.uint8)

def compute_click_from_diff(diff: np.ndarray, reg: Region, fallback_radius: int = 6) -> Tuple[int, int]:
    h, w = diff.shape
    flat = diff.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return reg.center()

    max_idx = int(np.argmax(flat))
    max_y, max_x = divmod(max_idx, w)

    k = max(1, int(flat.size * 0.04))
    if k < flat.size:
        thresh = np.partition(flat, -k)[-k]
        mask = diff >= thresh
    else:
        mask = diff > 0

    if np.any(mask):
        ys, xs = np.nonzero(mask)
        weights = diff[ys, xs].astype(np.float32)
        s = float(weights.sum())
        if s <= 1e-6:
            y = int(ys.mean()); x = int(xs.mean())
        else:
            y = int(round((ys * weights).sum() / s))
            x = int(round((xs * weights).sum() / s))
        return reg.left + int(np.clip(x, 0, w - 1)), reg.top + int(np.clip(y, 0, h - 1))

    # fallback: small circular neighborhood around the max pixel
    rr = fallback_radius
    y0, x0 = max_y, max_x
    y_min = max(0, y0 - rr); y_max = min(h - 1, y0 + rr)
    x_min = max(0, x0 - rr); x_max = min(w - 1, x0 + rr)

    ys, xs, ws = [], [], []
    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):
            if (yy - y0) ** 2 + (xx - x0) ** 2 <= rr * rr:
                v = float(diff[yy, xx])
                if v > 0:
                    ys.append(yy); xs.append(xx); ws.append(v)

    if not ws:
        return reg.left + int(x0), reg.top + int(y0)

    ys = np.asarray(ys, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    ws = np.asarray(ws, dtype=np.float32)
    s = float(ws.sum())
    cy = int(round((ys * ws).sum() / s))
    cx = int(round((xs * ws).sum() / s))
    return reg.left + int(np.clip(cx, 0, w - 1)), reg.top + int(np.clip(cy, 0, h - 1))

def baseline_stats(reg: Region, sct: mss.mss, seconds: float):
    prev = None
    mads, areas = [], []
    end = time.time() + max(0.2, seconds)
    delay = 1.0 / FRAME_RATE
    while time.time() < end and not stop_flag:
        bgr = grab_bgr(reg, sct)
        g = to_gray(bgr)
        if prev is not None:
            pg = to_gray(prev)
            raw = np.abs(g.astype(np.int16) - pg.astype(np.int16)).astype(np.uint8)
            gd = apply_color_gate(raw, bgr)
            use = gd if gd.any() else raw
            mad = float(np.mean(use if not gd.any() else gd[gd > 0]))
            area = int((use > DIFF_PIXEL_MIN).sum())
            mads.append(mad); areas.append(area)
            if debug_on:
                print(f"[dbg] base mad={mad:.2f} area={area}")
        prev = bgr
        time.sleep(delay)
    if not mads:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(mads))
    std = float(np.std(mads, ddof=1)) if len(mads) > 1 else 0.0
    area_mean = float(np.mean(areas)) if areas else 0.0
    return mean, std, area_mean

def monitor_for_bite_and_target(reg: Region, sct: mss.mss) -> Tuple[bool, Optional[Tuple[int, int]]]:
    base_mean, base_std, base_area = baseline_stats(reg, sct, BASELINE_SECONDS)
    dyn_thresh = max(base_mean * THRESH_MULT_ALT, base_mean + Z_THRESH * max(base_std, 1e-6))
    area_thresh = max(base_area * AREA_MULT, AREA_MIN_ABS)
    raw_dyn_thresh = max(base_mean * (THRESH_MULT_ALT * 0.8), base_mean + (Z_THRESH * 0.9) * max(base_std, 1e-6))
    raw_area_thresh = max(area_thresh * 0.6, max(AREA_MIN_ABS // 2, 10))

    if debug_on:
        print(f"[thresh] gated={dyn_thresh:.2f}/{int(area_thresh)} raw={raw_dyn_thresh:.2f}/{int(raw_area_thresh)}")

    prev = None
    sustain = 0
    last_target = None
    deadline = time.time() + 22.0
    delay = 1.0 / FRAME_RATE

    while time.time() < deadline and running and not stop_flag:
        bgr = grab_bgr(reg, sct)
        g = to_gray(bgr)
        if prev is not None:
            pg = to_gray(prev)
            raw_diff = np.abs(g.astype(np.int16) - pg.astype(np.int16)).astype(np.uint8)

            gd = apply_color_gate(raw_diff, bgr) if COLOR_MODE != 'off' else raw_diff
            gated_has = bool(gd.any())

            if gated_has:
                gated_mad = float(np.mean(gd[gd > 0]))
                gated_area = int((gd > DIFF_PIXEL_MIN).sum())
            else:
                gated_mad, gated_area = 0.0, 0

            raw_mad = float(np.mean(raw_diff)) if raw_diff.size else 0.0
            raw_area = int((raw_diff > DIFF_PIXEL_MIN).sum())
            peak_val = int(raw_diff.max()) if raw_diff.size else 0

            last_target = compute_click_from_diff(gd if (gated_has and gated_area > 0) else raw_diff, reg)

            if debug_on:
                print(f"[dbg] gmad={gated_mad:.2f} garea={gated_area} | rmad={raw_mad:.2f} rarea={raw_area} peak={peak_val}")

            gated_ok = (gated_mad >= dyn_thresh and gated_area >= area_thresh)
            raw_ok   = (raw_mad >= raw_dyn_thresh and raw_area >= raw_area_thresh)
            peak_ok  = (peak_val >= max(dyn_thresh * 2.5, 100))

            if gated_ok or raw_ok or peak_ok:
                sustain += 1
                if sustain >= 1:
                    return True, last_target
            else:
                sustain = 0

        prev = bgr
        time.sleep(delay)

    return False, None

# ---------------------------- main loop ----------------------------
def fishing_loop():
    if current_region is None:
        log("[run] set a region first (F6 twice)")
        return

    log("[run] started")
    with mss.mss() as sct:
        while running and not stop_flag:
            parked_this_cycle = False

            if PARK_MOUSE and PARK_BEFORE_CAST and not parked_this_cycle:
                px, py = point_outside_region(current_region)
                human_move_to(px, py, dur_range=(0.05, 0.10))
                time.sleep(rnd(0.02, 0.05))
                parked_this_cycle = True

            cast()
            time.sleep(rnd(*LAND_BOBBER_DELAY))

            hit, target = monitor_for_bite_and_target(current_region, sct)

            if hit:
                x, y = target if target else current_region.center()
                human_right_click(x, y)
                time.sleep(rnd(*POST_LOOT_DELAY))

                # If someone later turns PARK_AFTER_LOOT on, still ensure only one move per cycle.
                if PARK_MOUSE and PARK_AFTER_LOOT and not parked_this_cycle:
                    px, py = point_outside_region(current_region)
                    human_move_to(px, py, dur_range=(0.05, 0.10))
                    time.sleep(rnd(0.02, 0.05))
                    parked_this_cycle = True

            time.sleep(rnd(*RECAST_DELAY_RANGE))
    log("[run] stopped")

# ---------------------------- hotkeys ----------------------------
def toggle_run():
    global running
    if not running:
        if current_region is None:
            log("[run] set a region first (F6 twice)")
            return
        running = True
        threading.Thread(target=fishing_loop, daemon=True).start()
        log("[run] RUNNING (F8 to stop)")
    else:
        running = False
        log("[run] PAUSED (F8 to start)")

def toggle_debug():
    global debug_on
    debug_on = not debug_on
    log(f"[dbg] {'ON' if debug_on else 'OFF'}")

def test_center_click():
    if current_region is None:
        log("[test] set a region first")
        return
    x, y = current_region.center()
    log(f"[test] right-click center {x},{y}")
    human_right_click(x, y)

def test_autotarget_click():
    if current_region is None:
        log("[test] set a region first")
        return
    with mss.mss() as sct:
        b1 = grab_bgr(current_region, sct)
        time.sleep(1.0 / FRAME_RATE)
        b2 = grab_bgr(current_region, sct)
        g1, g2 = to_gray(b1), to_gray(b2)
        diff = np.abs(g2.astype(np.int16) - g1.astype(np.int16)).astype(np.uint8)
        diff_gated = apply_color_gate(diff, b2)
        tx, ty = compute_click_from_diff(diff_gated if diff_gated.any() else diff, current_region)
        log(f"[test] right-click auto-target {tx},{ty}")
        human_right_click(tx, ty)

def main():
    log("WoW Fishing — motion spike + auto-track")
    log(f"[{SET_REGION_KEY.upper()}] pick region  [{RUN_TOGGLE_KEY.upper()}] start/stop  "
        f"[{DEBUG_TOGGLE_KEY.upper()}] debug  [{TEST_CENTER_CLICK_KEY.upper()}] test center  "
        f"[{TEST_AUTOTARGET_CLICK_KEY.upper()}] test target  [{CANCEL_REGION_KEY.upper()}] cancel  "
        f"[{EXIT_KEY.upper()}] exit")

    keyboard.add_hotkey(SET_REGION_KEY, set_region_interactively)
    keyboard.add_hotkey(CANCEL_REGION_KEY, cancel_region)
    keyboard.add_hotkey(RUN_TOGGLE_KEY, toggle_run)
    keyboard.add_hotkey(DEBUG_TOGGLE_KEY, toggle_debug)
    keyboard.add_hotkey(TEST_CENTER_CLICK_KEY, test_center_click)
    keyboard.add_hotkey(TEST_AUTOTARGET_CLICK_KEY, test_autotarget_click)

    try:
        keyboard.wait(EXIT_KEY)
    except KeyboardInterrupt:
        pass
    finally:
        global stop_flag, running
        stop_flag = True
        running = False
        log("[exit] bye")

if __name__ == "__main__":
    main()
