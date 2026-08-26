#!/usr/bin/env python3
"""
Headless calibration + live readout. No window, works over SSH.

    python3 calibrate_headless.py --dist 0.5          # calibrate at 0.5 m
    python3 calibrate_headless.py --focal 812         # live distance readout
    python3 calibrate_headless.py --focal 812 --save frame.jpg   # dump a frame

Hold the marker at the stated distance, square to the lens. The script
averages 30 detections so a single bad frame cannot skew the result.
"""

import argparse
import math
import time

import cv2
import numpy as np

MARKER_ID = 7
MARKER_SIZE_M = 0.20      # <-- SET THIS to your printed marker's black square width
DICT = cv2.aruco.DICT_4X4_50

WIDTH, HEIGHT = 1280, 720


def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"could not open /dev/video{idx}")
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"camera open at {int(w)}x{int(h)}")
    return cap


def get_detector():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    return ("legacy", d, cv2.aruco.DetectorParameters_create())


def detect(det, gray):
    if isinstance(det, tuple):
        _, d, p = det
        return cv2.aruco.detectMarkers(gray, d, parameters=p)[:2]
    return det.detectMarkers(gray)[:2]


def marker_width_px(corners_one):
    pts = corners_one.reshape(4, 2)
    top = np.linalg.norm(pts[1] - pts[0])
    bot = np.linalg.norm(pts[2] - pts[3])
    cx = pts[:, 0].mean()
    return (top + bot) / 2.0, cx


def find(cap, det, tries=1):
    """Grab frames until the marker is seen. Returns (width_px, cx, frame_w)."""
    for _ in range(tries):
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(det, gray)
        if ids is not None and MARKER_ID in ids.flatten():
            i = list(ids.flatten()).index(MARKER_ID)
            w, cx = marker_width_px(corners[i])
            return w, cx, frame.shape[1], frame
    return None


def calibrate(cam, known):
    cap = open_cam(cam)
    det = get_detector()
    print(f"\nhold the marker flat and square at exactly {known} m")
    print("collecting 30 samples...\n")

    widths = []
    misses = 0
    while len(widths) < 30:
        r = find(cap, det, tries=1)
        if r is None:
            misses += 1
            if misses % 30 == 0:
                print(f"  no marker yet ({misses} frames)... check lighting and framing")
            if misses > 600:
                raise SystemExit("gave up -- marker never detected")
            continue
        w, cx, fw, _ = r
        widths.append(w)
        print(f"  sample {len(widths):2d}/30   width = {w:6.1f} px")
        time.sleep(0.05)

    cap.release()
    arr = np.array(widths)
    # drop outliers, keep the middle 80%
    lo, hi = np.percentile(arr, [10, 90])
    keep = arr[(arr >= lo) & (arr <= hi)]
    w_med = float(np.median(keep))
    focal = (w_med * known) / MARKER_SIZE_M

    print(f"\n  median width : {w_med:.1f} px  (spread {arr.min():.0f}-{arr.max():.0f})")
    print(f"  FOCAL_PX     : {focal:.1f}")
    print(f"\n  run: python3 calibrate_headless.py --focal {focal:.0f}\n")
    if w_med < 60:
        print("  WARNING: marker is small in frame. Print it bigger or")
        print("           calibrate closer, or distance will be noisy.\n")


def live(cam, focal, save=None):
    cap = open_cam(cam)
    det = get_detector()
    print(f"\nfocal = {focal}   marker = {MARKER_SIZE_M} m")
    print("ctrl-C to stop\n")
    try:
        while True:
            r = find(cap, det, tries=1)
            if r is None:
                print("  ---  no target")
                time.sleep(0.2)
                continue
            w, cx, fw, frame = r
            dist = (MARKER_SIZE_M * focal) / w
            bear = math.degrees(math.atan2(cx - fw / 2.0, focal))
            print(f"  dist = {dist:5.2f} m    bearing = {bear:+6.1f} deg    ({w:.0f} px)")
            if save:
                cv2.imwrite(save, frame)
                print(f"  wrote {save}")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--dist", type=float, help="calibrate: known distance in metres")
    p.add_argument("--focal", type=float, help="live readout with this focal length")
    p.add_argument("--save", type=str, help="write one frame to this path and exit")
    a = p.parse_args()

    if a.dist:
        calibrate(a.cam, a.dist)
    elif a.focal:
        live(a.cam, a.focal, a.save)
    else:
        p.print_help()