"""
hand_gesture_launcher.py
Gesture -> Open/Launch actions.

Requirements:
    pip install opencv-python mediapipe numpy

How it works:
    - Detects gestures (swipe_left/right/up/down, push, pull).
    - When a gesture is recognized, runs the mapped action (open URL, folder, app command).
    - Cross-platform support (Windows / macOS / Linux). Configure ACTION_MAP to change behavior.

Controls:
    ESC : quit
    c   : clear HUD / reset
    m   : toggle mapping (enable / disable gesture-action execution)
    h   : print mapping help to console
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import platform
import subprocess
import time
from collections import deque

# ---------- USER TUNABLE PARAMETERS ----------
TRACK_LANDMARK_IDX = 9     # palm center; good for swipes
MAX_HISTORY = 20
SMOOTHING_POS = 0.6
SMOOTHING_VEL = 0.5
VEL_THRESHOLD = 6.0
SWIPE_FRAMES = 6
SWIPE_DISPLACEMENT = 80
PUSH_AREA_RATIO = 1.12
PUSH_SUSTAIN = 5
GESTURE_COOLDOWN_FRAMES = 12  # frames to wait after firing an action

# ---------- ACTION MAP (customize this) ----------
# Supported action types:
#   - {"type": "url", "payload": "https://example.com"}
#   - {"type": "folder", "payload": "/path/to/folder"}  # opens file explorer
#   - {"type": "file", "payload": "/path/to/file"}      # opens a file with default app
#   - {"type": "cmd", "payload": ["command", "arg1", ...]}  # runs command (spawn, non-blocking)
#   - {"type": "shell", "payload": "some shell command string"}  # runs via shell=True
#
# Be conservative with destructive commands. This script will execute whatever you configure.
# Example sensible defaults are provided below.
ACTION_MAP = {
    "swipe_right": {"type": "url", "payload": "https://www.google.com"},
    "swipe_left":  {"type": "folder", "payload": None},  # None -> open home directory
    "swipe_up":    {"type": "cmd", "payload": ["code", "."]},  # tries to open VS Code in cwd
    "swipe_down":  {"type": "cmd", "payload": None},  # fallback open terminal
    "pull":        {"type": "folder", "payload": None},  # another folder open
}

# ---------- Helpers ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SYSTEM = platform.system().lower()

def safe_print_mapping():
    print("Gesture -> Action mapping:")
    for g,a in ACTION_MAP.items():
        print(f"  {g:12s} -> {a}")

def open_url(url):
    try:
        if SYSTEM == "windows":
            os.startfile(url)  # handles URLs too
        elif SYSTEM == "darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        print(f"[Action] opened URL: {url}")
    except Exception as e:
        print(f"[Error] open_url failed: {e}")

def open_path(path):
    # open folder or file with default OS behavior
    try:
        if path is None:
            path = os.path.expanduser("~")
        if SYSTEM == "windows":
            os.startfile(path)
        elif SYSTEM == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        print(f"[Action] opened path: {path}")
    except Exception as e:
        print(f"[Error] open_path failed: {e}")

def run_cmd(cmd_list):
    # cmd_list can be None: fallback behavior depends on OS
    try:
        if cmd_list is None:
            # fallback: open a terminal window
            if SYSTEM == "windows":
                subprocess.Popen(["cmd.exe"])
            elif SYSTEM == "darwin":
                subprocess.Popen(["open", "-a", "Terminal"])
            else:
                # try common Linux terminals
                for t in ("gnome-terminal", "konsole", "xterm", "tilix", "alacritty"):
                    try:
                        subprocess.Popen([t])
                        break
                    except FileNotFoundError:
                        continue
        else:
            # launch command non-blocking
            subprocess.Popen(cmd_list)
        print(f"[Action] ran command: {cmd_list if cmd_list else '[terminal fallback]'}")
    except Exception as e:
        print(f"[Error] run_cmd failed: {e}")

def execute_action(action):
    # action is a dict with type/payload as above
    if action is None:
        return
    t = action.get("type")
    p = action.get("payload")
    if t == "url":
        open_url(p)
    elif t in ("folder", "file", "path"):
        open_path(p)
    elif t == "cmd":
        run_cmd(p)
    elif t == "shell":
        try:
            subprocess.Popen(p, shell=True)
            print(f"[Action] shell: {p}")
        except Exception as e:
            print(f"[Error] shell failed: {e}")
    else:
        print(f"[Warning] unknown action type: {t}")

# ---------- Gesture detection (similar to previous scripts) ----------
def direction_from_velocity(vx, vy, vel_thresh=VEL_THRESHOLD):
    mag = np.hypot(vx, vy)
    if mag < vel_thresh:
        return "steady"
    ang = np.degrees(np.arctan2(vy, vx))
    if -45 <= ang <= 45:
        return "right"
    if ang >= 135 or ang <= -135:
        return "left"
    if 45 < ang < 135:
        return "down"
    if -135 < ang < -45:
        return "up"
    return "unknown"

def main():
    print("Starting gesture launcher. ESC to quit. Press 'm' to toggle mapping on/off. 'h' to show mapping.")
    safe_print_mapping()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)

    pts = deque(maxlen=MAX_HISTORY)
    area_hist = deque(maxlen=PUSH_SUSTAIN)
    smoothed_pos = None
    smoothed_vx = 0.0
    smoothed_vy = 0.0
    recent_dirs = deque(maxlen=SWIPE_FRAMES)

    mapping_enabled = True
    gesture_cooldown = 0
    last_action_text = "none"
    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            tracked_pt = None
            bbox_area = None

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                bbox_area = max(1e-9, (maxx - minx) * (maxy - miny))
                chosen = lm.landmark[TRACK_LANDMARK_IDX]
                tracked_pt = (int(chosen.x * w), int(chosen.y * h))
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # smoothing & velocity
            if tracked_pt is not None:
                pts.appendleft(tracked_pt)
                if smoothed_pos is None:
                    smoothed_pos = np.array(tracked_pt, dtype=float)
                    prev = smoothed_pos.copy()
                else:
                    prev = smoothed_pos.copy()
                    smoothed_pos = SMOOTHING_POS * np.array(tracked_pt, dtype=float) + (1 - SMOOTHING_POS) * smoothed_pos
                vx = smoothed_pos[0] - prev[0]
                vy = smoothed_pos[1] - prev[1]
                smoothed_vx = SMOOTHING_VEL * vx + (1 - SMOOTHING_VEL) * smoothed_vx
                smoothed_vy = SMOOTHING_VEL * vy + (1 - SMOOTHING_VEL) * smoothed_vy
                dir_label = direction_from_velocity(smoothed_vx, smoothed_vy)
                recent_dirs.append(dir_label)
            else:
                smoothed_vx *= 0.85
                smoothed_vy *= 0.85
                dir_label = "no_hand"

            # area history for push/pull
            if bbox_area is not None:
                area_hist.appendleft(bbox_area)
            else:
                area_hist.appendleft(None)

            detected_gesture = None
            if gesture_cooldown > 0:
                gesture_cooldown -= 1

            # Swipe detection
            if gesture_cooldown == 0 and len(pts) >= SWIPE_FRAMES:
                p_new = np.array(pts[0], dtype=float)
                p_old = np.array(pts[SWIPE_FRAMES-1], dtype=float)
                disp = p_new - p_old
                disp_mag = np.hypot(disp[0], disp[1])
                dir_counts = {}
                for d in recent_dirs:
                    dir_counts[d] = dir_counts.get(d, 0) + 1
                candidate = max(dir_counts.items(), key=lambda x: x[1])[0] if dir_counts else "steady"
                if disp_mag > SWIPE_DISPLACEMENT and candidate in ("left","right","up","down"):
                    if dir_counts.get(candidate, 0) >= SWIPE_FRAMES // 2:
                        detected_gesture = f"swipe_{candidate}"
                        gesture_cooldown = GESTURE_COOLDOWN_FRAMES

            # Push / Pull detection using area changes
            if gesture_cooldown == 0 and len(area_hist) >= PUSH_SUSTAIN and None not in area_hist:
                newest = area_hist[0]; oldest = area_hist[-1]
                if newest / (oldest + 1e-12) > PUSH_AREA_RATIO:
                    detected_gesture = "push"
                    gesture_cooldown = GESTURE_COOLDOWN_FRAMES
                elif newest / (oldest + 1e-12) < (1.0 / PUSH_AREA_RATIO):
                    detected_gesture = "pull"
                    gesture_cooldown = GESTURE_COOLDOWN_FRAMES

            # If detected and mapping enabled, execute action
            if detected_gesture and mapping_enabled:
                action = ACTION_MAP.get(detected_gesture)
                if action:
                    execute_action(action)
                    last_action_text = f"{detected_gesture} -> {action['type']}"
                else:
                    last_action_text = f"{detected_gesture} -> (no mapping)"

            # HUD overlay
            hud = f"Dir:{dir_label} Gesture:{detected_gesture if detected_gesture else '—'} Mapping:{'ON' if mapping_enabled else 'OFF'}"
            hud2 = f"Last action: {last_action_text}"
            cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
            cv2.putText(frame, hud2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # draw smoothed tracking point
            if smoothed_pos is not None:
                cx, cy = int(smoothed_pos[0]), int(smoothed_pos[1])
                cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

            # trail
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0,200,200), 2)

            cv2.imshow("Gesture Launcher", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('c'):
                pts.clear(); area_hist.clear()
                smoothed_pos = None; smoothed_vx = 0.0; smoothed_vy = 0.0
                last_action_text = "reset"
                print("[Action] cleared history")
            elif key == ord('m'):
                mapping_enabled = not mapping_enabled
                print(f"[Action] mapping_enabled = {mapping_enabled}")
            elif key == ord('h'):
                safe_print_mapping()
            # else ignore other keys

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
