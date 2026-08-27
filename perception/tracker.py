#!/usr/bin/env python3
"""
Follow-me perception layer.
# Calibrated <today's date>, laptop webcam @ 960x540 (approx)
# Phone-screen marker. RE-CALIBRATE after switching to the printed marker.
FOCAL_PX = 704 
Detects an ArUco marker and reports (distance_m, bearing_deg) -- the exact same
two numbers a UWB anchor pair would give you. Swap this module out for a UWB
reader later and the controller downstream does not change.

Usage:
    python tracker.py --genmarker      # writes marker.png -- print it, tape it on your back
    python tracker.py --calibrate      # hold marker at exactly 2.0 m, get your focal length
    python tracker.py --focal 800      # live tracking
    python tracker.py --focal 800 --serial COM5   # live tracking + send to ESP32
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

MARKER_ID = 7
MARKER_SIZE_M = 0.17      # physical width of the printed marker, in metres. MEASURE IT.
DICT = cv2.aruco.DICT_4X4_50


def get_detector():
    """OpenCV changed the aruco API at 4.7. Support both."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    return ("legacy", d, cv2.aruco.DetectorParameters_create())


def detect(detector, gray):
    if isinstance(detector, tuple):
        _, d, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def gen_marker(path="marker.png", px=1000):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(d, MARKER_ID, px)
    else:
        img = cv2.aruco.drawMarker(d, MARKER_ID, px)
    img = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
    print(f"wrote {path} (id={MARKER_ID})")
    print(f"Print it, then MEASURE the black square's width and set MARKER_SIZE_M.")
    print(f"White border matters -- do not crop it off.")


def measure(corners_one):
    """Return (pixel_width, cx, cy) for a single detected marker."""
    pts = corners_one.reshape(4, 2)
    # average the two horizontal edges -- more robust than one
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    width_px = (top + bottom) / 2.0
    cx, cy = pts.mean(axis=0)
    return width_px, cx, cy


def solve(width_px, cx, frame_w, focal_px):
    """Pinhole model. distance = (real_size * focal) / apparent_size."""
    distance = (MARKER_SIZE_M * focal_px) / max(width_px, 1e-6)
    offset_px = cx - (frame_w / 2.0)
    bearing = math.degrees(math.atan2(offset_px, focal_px))
    return distance, bearing


def calibrate(cam_index=0, known_distance=0.5):
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    det = get_detector()
    print(f"Hold the marker flat, facing the camera, at exactly {known_distance} m.")
    print("Press SPACE to capture, q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(det, gray)
        if ids is not None and MARKER_ID in ids.flatten():
            i = list(ids.flatten()).index(MARKER_ID)
            w, cx, cy = measure(corners[i])
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"width={w:.1f}px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                focal = (w * known_distance) / MARKER_SIZE_M
                print(f"\n  focal_px = {focal:.1f}")
                print(f"  run:  python tracker.py --focal {focal:.0f}\n")
                break
        else:
            cv2.putText(frame, "no marker", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.imshow("calibrate", frame)
    cap.release()
    cv2.destroyAllWindows()


class EMA:
    """Exponential smoothing. Raw detections jitter; the cart must not."""
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None

    def __call__(self, x):
        self.value = x if self.value is None else \
            self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def track(focal_px, cam_index=0, serial_port=None, show=True):
   
    ser = None
    if serial_port:
        import serial  # pip install pyserial
        ser = serial.Serial(serial_port, 115200, timeout=0.1)
        time.sleep(2)  # ESP32 resets on port open
    
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    det = get_detector()
    d_filt, b_filt = EMA(0.35), EMA(0.35)
    last_seen = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(det, gray)

        found = ids is not None and MARKER_ID in ids.flatten()
        if found:
            i = list(ids.flatten()).index(MARKER_ID)
            wpx, cx, cy = measure(corners[i])
            dist, bear = solve(wpx, cx, w_frame, focal_px)
            dist, bear = d_filt(dist), b_filt(bear)
            last_seen = time.time()
            if ser:
                # ESP32 enforces its own timeout; this is just the data path
                ser.write(f"T,{dist:.3f},{bear:.2f}\n".encode())
            if show:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, f"{dist:.2f} m   {bear:+.1f} deg", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # STOP is the safe default. Send nothing and let the ESP32 time out,
            # but send an explicit stop too -- belt and braces.
            if ser:
                ser.write(b"S\n")
            if show:
                cv2.putText(frame, "TARGET LOST", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if ser and ser.in_waiting:
                try:
                    echo = ser.readline().decode(errors="ignore").strip()
                    if echo:
                        print(echo)
                except Exception:
                    pass    

        if show:
            cv2.line(frame, (w_frame // 2, 0), (w_frame // 2, h), (255, 255, 0), 1)
            cv2.imshow("tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if ser:
        ser.write(b"S\n")
        ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genmarker", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--focal", type=float, default=None)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--serial", type=str, default=None)
    a = p.parse_args()

    if a.genmarker:
        gen_marker()
    elif a.calibrate:
        calibrate(a.cam)
    elif a.focal:
        track(a.focal, a.cam, a.serial)
    else:
        p.print_help()
        sys.exit(1)