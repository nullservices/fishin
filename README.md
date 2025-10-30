# 🎣 Fishin — Local WoW Auto-Fishing Assistant

**Fishin** is a lightweight, fully local Python utility that helps you automate fishing**.  
It detects the bobber splash visually (via motion detection) and right-clicks to reel in — all while keeping things simple, human-like, and customizable.

---

## 🧠 Features

- 🖥️ Works on your own **local or LAN WoW server**
- 🎞️ **Motion-based detection** — no memory reads or packet hooks
- 🧩 Auto-tracks and right-clicks the bobber when it splashes
- 👁️ Adjustable sensitivity (tune to avoid ripples or false triggers)
- 🧍 Human-style cursor movement & randomized timing
- 🐭 Smart "cursor park" — moves the mouse just outside your fishing box
- ⏸️ Fully controlled by hotkeys — no overlays or GUIs needed

---

## ⚙️ Controls

| Key | Action |
|-----|---------|
| **F6 (x2)** | Select fishing region (top-left → bottom-right) |
| **F8** | Start / Stop fishing loop |
| **F7** | Toggle debug output (shows thresholds and motion metrics) |
| **F2** | Test right-click at region center |
| **F3** | Test auto-target click (motion centroid) |
| **F9** | Exit Fishin |

> Make sure *Auto Loot* is turned on in your WoW interface settings.

---

## 🧰 Requirements

| Dependency | Install Command |
|-------------|----------------|
| Python 3.10+ | — |
| mss | `pip install mss` |
| pyautogui | `pip install pyautogui` |
| numpy | `pip install numpy` |
| keyboard | `pip install keyboard` |

> **Windows:** You may need to run `python -m pip install pyautogui mss keyboard numpy` as administrator.  
> **Mac/Linux:** Works best in windowed WoW mode (no fullscreen capture).

---

## ⚙️ Configuration

Edit the **user settings** at the top of `fishin.py` to your liking:

```python
CAST_KEY = '1'          # your fishing keybind
Z_THRESH = 5.0          # increase to make it less sensitive
AREA_MULT = 3.5         # require larger splash area
PARK_BEFORE_CAST = True # move cursor outside box once per cycle
```

---

## 🧭 Usage

1. Launch WoW and stand near water.
2. Run the script:  
   ```bash
   python fishin.py
   ```
3. Press **F6 twice** to select your bobber region.
4. Press **F8** to start fishing.
5. Sit back and relax — the fish practically catch themselves!

---

## 🪄 Tips

- Adjust `Z_THRESH` and `AREA_MULT` for your water type and resolution.  
- Make the region tight around your bobber’s landing spot.  
- Run in **windowed (borderless)** mode for best capture stability.  
- If you notice false clicks on ripples, increase sensitivity a little.

---

## ⚠️ Disclaimer

This tool is intended **only for personal or private servers** where automation is allowed.  
Do **not** use this on public realms — doing so may violate their terms of service.

---

## 📜 License

MIT License © 2025 — You’re free to use, modify, and share for personal projects.

---

**Coded with 🐍 Python and patience.**  
If you enjoy it, give the project a ⭐ on GitHub!
