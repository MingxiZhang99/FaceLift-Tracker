"""
Real-Time Webcam: Compress → Landmark → GPEN-256 → Landmark Warp → Blend
==========================================================================
GPEN-256 + Landmark-guided restoration. Fully standalone, no external deps.

Problem:
  a. GPEN-256 alone distorts facial geometry — particularly enlarging eyes and
     warping features. This is especially unstable when the subject wears glasses.
  b. GPEN inference latency (~30-50ms) causes the face ROI to lag behind during
     fast head movement, resulting in visible misalignment between the restored
     region and the actual face position.

Solution:
  a. Landmark-guided correction: detect landmarks on the original crop, run GPEN,
     detect landmarks on the restored output, then apply per-region affine warps
     (eyes/nose/mouth independently) to pull distorted features back to their
     original positions.
  b. Inter-frame tracking with Kalman-smoothed MOSSE or sparse optical flow
     (Lucas-Kanade) to maintain face ROI continuity between detection frames,
     reducing effective per-frame cost to ~1-3ms while keeping the ROI locked
     onto the face during rapid motion.

Pipeline:
  Webcam → Compress → Detect Face
    → Detect landmarks (original feature positions)
    → GPEN-256 restore
    → Detect restored landmarks
    → Local affine warp (eyes/nose/mouth aligned separately)
    → Blend

Controls:
  q - quit
  c - cycle CRF (23/35/45)
  w - cycle restore weight (0.5/0.7/1.0)
  f - cycle detection frequency (1/3/5/10 frames)
  l - toggle landmark correction on/off
  s - save screenshot
"""

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions


# ─── Config ──────────────────────────────────────────────────────────────────

ROI_PADDING = 0.4
BLEND_BORDER = 12

_DIR = os.path.dirname(os.path.abspath(__file__))
_GPEN_PATH = os.path.join(_DIR, "GPEN-BFR-256.onnx")
_LANDMARK_PATH = os.path.join(_DIR, "face_landmarker.task")


# ─── FaceROI ─────────────────────────────────────────────────────────────────

@dataclass
class FaceROI:
    x1: int
    y1: int
    x2: int
    y2: int
    crop: np.ndarray


# ─── Blend ───────────────────────────────────────────────────────────────────

def blend_roi(frame: np.ndarray, restored_crop: np.ndarray, roi: FaceROI) -> np.ndarray:
    """Paste restored_crop into frame at roi location with alpha feathering."""
    out = frame.copy()
    rh, rw = roi.y2 - roi.y1, roi.x2 - roi.x1

    if restored_crop.shape[:2] != (rh, rw):
        restored_crop = cv2.resize(restored_crop, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

    mask = np.ones((rh, rw), dtype=np.float32)
    b = min(BLEND_BORDER, rh // 2, rw // 2)
    for i in range(b):
        val = i / b
        mask[i, :] = np.minimum(mask[i, :], val)
        mask[rh - 1 - i, :] = np.minimum(mask[rh - 1 - i, :], val)
        mask[:, i] = np.minimum(mask[:, i], val)
        mask[:, rw - 1 - i] = np.minimum(mask[:, rw - 1 - i], val)

    mask = mask[..., None]
    region = out[roi.y1:roi.y2, roi.x1:roi.x2].astype(np.float32)
    blended = region * (1 - mask) + restored_crop.astype(np.float32) * mask
    out[roi.y1:roi.y2, roi.x1:roi.x2] = blended.astype(np.uint8)
    return out


# ─── Landmark Detector (MediaPipe, ~2ms) ─────────────────────────────────────

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE = [1, 2, 98, 327]
MOUTH = [61, 291, 0, 17, 78, 308]
KEY_INDICES = LEFT_EYE + RIGHT_EYE + NOSE + MOUTH


class LightLandmarker:
    """MediaPipe FaceLandmarker — 478 points, ~2ms/face."""

    def __init__(self):
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_LANDMARK_PATH),
            num_faces=1,
        )
        self.detector = FaceLandmarker.create_from_options(opts)

    def detect(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """Return key landmark coords [N, 2] in pixels, or None."""
        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        return points[KEY_INDICES]


# ─── GPEN-256 Restorer ───────────────────────────────────────────────────────

class GPENRestorer:
    """GPEN-256 via ONNX Runtime."""

    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            _GPEN_PATH, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] GPEN-256 loaded (ONNX, threads={opts.intra_op_num_threads})")

    def run(self, face_bgr: np.ndarray) -> np.ndarray:
        """Input BGR face, output GPEN restored result (same size)."""
        h, w = face_bgr.shape[:2]

        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        output = self.session.run(None, {self.input_name: img})[0]

        out = np.clip(output[0], -1, 1)
        out = ((out + 1) / 2 * 255).astype(np.uint8)
        out = np.transpose(out, (1, 2, 0))
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        if (h, w) != (256, 256):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out


# ─── Landmark-Guided Correction ──────────────────────────────────────────────

def landmark_correct(restored: np.ndarray, lm_src: np.ndarray, lm_dst: np.ndarray) -> np.ndarray:
    """
    Warp restored facial features back to original positions.
    lm_src: landmarks from GPEN output
    lm_dst: landmarks from original input
    """
    h, w = restored.shape[:2]
    result = restored.copy()

    groups = [
        slice(0, 6),    # left eye
        slice(6, 12),   # right eye
        slice(12, 16),  # nose
        slice(16, 22),  # mouth
    ]

    for idx_slice in groups:
        src_pts = lm_src[idx_slice]
        dst_pts = lm_dst[idx_slice]

        if len(src_pts) < 3:
            continue

        try:
            M = cv2.getAffineTransform(
                src_pts[:3].astype(np.float32),
                dst_pts[:3].astype(np.float32)
            )
        except cv2.error:
            continue

        warped = cv2.warpAffine(restored, M, (w, h), flags=cv2.INTER_LINEAR)

        center = dst_pts.mean(axis=0).astype(int)
        pts_range = dst_pts.max(axis=0) - dst_pts.min(axis=0)
        radius = (max(int(pts_range[0] * 0.7), 10), max(int(pts_range[1] * 0.7), 10))

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, tuple(center), radius, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)

        mask_3ch = mask[:, :, np.newaxis]
        result = (warped * mask_3ch + result * (1 - mask_3ch)).astype(np.uint8)

    return result


# ─── Full Restore Pipeline ───────────────────────────────────────────────────

def restore_with_landmark(crop, gpen, landmarker, weight=0.7, use_correction=True):
    """GPEN restore + landmark correction + blend."""
    lm_orig = landmarker.detect(crop) if use_correction else None

    restored = gpen.run(crop)

    if use_correction and lm_orig is not None:
        lm_restored = landmarker.detect(restored)
        if lm_restored is not None:
            diff = np.mean(np.abs(lm_restored - lm_orig))
            if diff > 2.0:
                restored = landmark_correct(restored, lm_restored, lm_orig)

    return cv2.addWeighted(restored, weight, crop, 1 - weight, 0)


# ─── Face Tracker (Kalman + MOSSE) ───────────────────────────────────────────

class FaceTracker:
    """
    Combines Haar cascade detection with Kalman filter prediction and
    MOSSE tracking for smooth, low-latency face ROI tracking.

    - Detection frames: run cascade, reinitialize tracker + Kalman state
    - Tracking frames: MOSSE update (~0.5ms) + Kalman predict for smoothing
    """

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._tracker = None
        self._kalman = self._init_kalman()
        self._initialized = False
        self._last_bbox = None  # (x, y, w, h) in full-frame coords

    def _init_kalman(self) -> cv2.KalmanFilter:
        """4-state Kalman: [cx, cy, vx, vy], measures [cx, cy]."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def detect_and_init(self, frame: np.ndarray) -> FaceROI | None:
        """Run full detection, reinitialize tracker and Kalman."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))

        if len(faces) == 0:
            # Use Kalman prediction if detection fails
            if self._initialized:
                return self._predict_roi(frame)
            return None

        # Take largest face
        sx, sy = w / 320, h / 240
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x1, y1 = int(fx * sx), int(fy * sy)
        bw, bh = int(fw * sx), int(fh * sy)

        # Add padding
        px, py = int(bw * ROI_PADDING), int(bh * ROI_PADDING)
        rx1, ry1 = max(0, x1 - px), max(0, y1 - py)
        rx2, ry2 = min(w, x1 + bw + px), min(h, y1 + bh + py)

        self._last_bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)

        # Init MOSSE tracker
        self._tracker = cv2.legacy.TrackerMOSSE_create()
        self._tracker.init(frame, self._last_bbox)

        # Init Kalman state
        cx = rx1 + (rx2 - rx1) / 2
        cy = ry1 + (ry2 - ry1) / 2
        self._kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self._initialized = True

        crop = frame[ry1:ry2, rx1:rx2].copy()
        return FaceROI(x1=rx1, y1=ry1, x2=rx2, y2=ry2, crop=crop)

    def track(self, frame: np.ndarray) -> FaceROI | None:
        """Track face using MOSSE + Kalman prediction. ~1ms."""
        if not self._initialized:
            return None

        h, w = frame.shape[:2]

        # MOSSE update
        success, bbox = self._tracker.update(frame)

        if success:
            bx, by, bw, bh = [int(v) for v in bbox]
            cx, cy = bx + bw // 2, by + bh // 2

            # Kalman correct with MOSSE measurement
            self._kalman.correct(np.array([[cx], [cy]], dtype=np.float32))

            # Kalman predict (smoothed position)
            pred = self._kalman.predict()
            pcx, pcy = int(pred[0, 0]), int(pred[1, 0])

            # Use Kalman-smoothed center with MOSSE size
            rx1 = max(0, pcx - bw // 2)
            ry1 = max(0, pcy - bh // 2)
            rx2 = min(w, rx1 + bw)
            ry2 = min(h, ry1 + bh)

            self._last_bbox = (rx1, ry1, bw, bh)
            crop = frame[ry1:ry2, rx1:rx2].copy()
            return FaceROI(x1=rx1, y1=ry1, x2=rx2, y2=ry2, crop=crop)

        # MOSSE lost track — use pure Kalman prediction
        return self._predict_roi(frame)

    def _predict_roi(self, frame: np.ndarray) -> FaceROI | None:
        """Pure Kalman prediction when tracker fails."""
        if self._last_bbox is None:
            return None

        h, w = frame.shape[:2]
        _, _, bw, bh = self._last_bbox

        pred = self._kalman.predict()
        pcx, pcy = int(pred[0, 0]), int(pred[1, 0])

        rx1 = max(0, pcx - bw // 2)
        ry1 = max(0, pcy - bh // 2)
        rx2 = min(w, rx1 + bw)
        ry2 = min(h, ry1 + bh)

        if rx2 - rx1 < 20 or ry2 - ry1 < 20:
            return None

        crop = frame[ry1:ry2, rx1:rx2].copy()
        return FaceROI(x1=rx1, y1=ry1, x2=rx2, y2=ry2, crop=crop)


# ─── Face Tracker: Optical Flow ──────────────────────────────────────────────

class OpticalFlowTracker:
    """
    Tracks face ROI using sparse optical flow (Lucas-Kanade) on corner points
    inside the ROI. Estimates translation from median point displacement.

    - Detection frames: detect face, extract good corners inside ROI
    - Tracking frames: calcOpticalFlowPyrLK on corners → median shift → move ROI
    - More accurate than MOSSE for fast lateral movement
    - ~2-3ms per frame
    """

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._prev_gray = None
        self._points = None  # tracked points [N, 1, 2]
        self._last_bbox = None  # (x1, y1, w, h)
        self._initialized = False

        # Lucas-Kanade params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        # Good features params
        self._feature_params = dict(
            maxCorners=50,
            qualityLevel=0.05,
            minDistance=7,
            blockSize=7
        )

    def detect_and_init(self, frame: np.ndarray) -> FaceROI | None:
        """Run detection, extract corners inside face ROI for tracking."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 240))
        faces = self._cascade.detectMultiScale(small, 1.15, 5, minSize=(40, 40))

        if len(faces) == 0:
            if self._initialized:
                return self._track_flow(frame)
            return None

        # Take largest face
        sx, sy = w / 320, h / 240
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x1, y1 = int(fx * sx), int(fy * sy)
        bw, bh = int(fw * sx), int(fh * sy)

        px, py = int(bw * ROI_PADDING), int(bh * ROI_PADDING)
        rx1, ry1 = max(0, x1 - px), max(0, y1 - py)
        rx2, ry2 = min(w, x1 + bw + px), min(h, y1 + bh + py)

        self._last_bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)

        # Extract corners inside ROI for optical flow tracking
        roi_gray = gray[ry1:ry2, rx1:rx2]
        corners = cv2.goodFeaturesToTrack(roi_gray, **self._feature_params)

        if corners is not None and len(corners) > 0:
            # Offset corners to full-frame coordinates
            corners[:, 0, 0] += rx1
            corners[:, 0, 1] += ry1
            self._points = corners
        else:
            # Fallback: use ROI center + corners as points
            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            self._points = np.array([[[cx, cy]]], dtype=np.float32)

        self._prev_gray = gray
        self._initialized = True

        crop = frame[ry1:ry2, rx1:rx2].copy()
        return FaceROI(x1=rx1, y1=ry1, x2=rx2, y2=ry2, crop=crop)

    def track(self, frame: np.ndarray) -> FaceROI | None:
        """Track using optical flow."""
        if not self._initialized:
            return None
        return self._track_flow(frame)

    def _track_flow(self, frame: np.ndarray) -> FaceROI | None:
        """Compute optical flow displacement and shift ROI."""
        if self._prev_gray is None or self._points is None or self._last_bbox is None:
            return None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._points, None, **self._lk_params
        )

        if new_points is None or status is None:
            self._prev_gray = gray
            return self._fallback_roi(frame)

        # Filter good points
        good_mask = status.flatten() == 1
        if good_mask.sum() < 3:
            self._prev_gray = gray
            return self._fallback_roi(frame)

        old_good = self._points[good_mask]
        new_good = new_points[good_mask]

        # Median displacement (robust to outliers)
        dx = np.median(new_good[:, 0, 0] - old_good[:, 0, 0])
        dy = np.median(new_good[:, 0, 1] - old_good[:, 0, 1])

        # Shift ROI
        bx, by, bw, bh = self._last_bbox
        nx1 = int(max(0, bx + dx))
        ny1 = int(max(0, by + dy))
        nx2 = min(w, nx1 + bw)
        ny2 = min(h, ny1 + bh)

        self._last_bbox = (nx1, ny1, bw, bh)
        self._points = new_good.reshape(-1, 1, 2)
        self._prev_gray = gray

        if nx2 - nx1 < 20 or ny2 - ny1 < 20:
            return None

        crop = frame[ny1:ny2, nx1:nx2].copy()
        return FaceROI(x1=nx1, y1=ny1, x2=nx2, y2=ny2, crop=crop)

    def _fallback_roi(self, frame: np.ndarray) -> FaceROI | None:
        """Return last known ROI when flow fails."""
        if self._last_bbox is None:
            return None
        h, w = frame.shape[:2]
        bx, by, bw, bh = self._last_bbox
        rx2, ry2 = min(w, bx + bw), min(h, by + bh)
        if rx2 - bx < 20 or ry2 - by < 20:
            return None
        crop = frame[by:ry2, bx:rx2].copy()
        return FaceROI(x1=bx, y1=by, x2=rx2, y2=ry2, crop=crop)


# ─── Compression ─────────────────────────────────────────────────────────────

def compress_frame(frame: np.ndarray, crf: int) -> tuple[np.ndarray, int]:
    quality = max(5, int(95 - (crf - 18) * (90 / 33)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), len(buf)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(_GPEN_PATH):
        print(f"ERROR: Cannot find {_GPEN_PATH}")
        return
    if not os.path.exists(_LANDMARK_PATH):
        print(f"ERROR: Cannot find {_LANDMARK_PATH}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    tracker_kalman = FaceTracker()
    tracker_optflow = OpticalFlowTracker()
    use_optflow = False  # Start with Kalman+MOSSE, press 't' to toggle

    gpen = GPENRestorer()
    landmarker = LightLandmarker()
    print("[INFO] Landmark detector loaded (~2ms/face)")
    print("[INFO] Trackers: Kalman+MOSSE (~1ms) | OpticalFlow (~2-3ms)")
    print("[INFO] Press 't' to toggle between trackers")

    crf = 45
    crf_cycle = [23, 35, 45]
    weight = 0.7
    weight_cycle = [0.5, 0.7, 1.0]
    detect_every = 3
    detect_freq_cycle = [1, 3, 5, 10]
    use_correction = True

    fps = 0.0
    frame_idx = 0
    fps_count = 0
    fps_time = time.time()
    bitrate_kbps = 0.0
    bytes_acc = 0
    bitrate_time = time.time()
    restore_ms = 0.0
    save_idx = 0
    cached_rois: list[FaceROI] = []

    print("=" * 60)
    print("GPEN-256 + Landmark Guided Face Restoration")
    print("=" * 60)
    print("Keys: q=quit c=CRF w=weight f=det-freq l=landmark t=tracker s=save")
    print(f"Landmark correction: {'ON' if use_correction else 'OFF'}")
    print("=" * 60)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        original = frame.copy()

        # Compression
        compressed, enc_bytes = compress_frame(frame, crf)
        bytes_acc += enc_bytes

        # Face detection (full) or tracking (fast)
        tracker = tracker_optflow if use_optflow else tracker_kalman
        if frame_idx % detect_every == 0:
            roi = tracker.detect_and_init(compressed)
        else:
            roi = tracker.track(compressed)

        cached_rois = [roi] if roi is not None else []

        # GPEN + Landmark restoration
        restored_frame = compressed.copy()
        t_restore = time.time()
        for roi in cached_rois:
            restored_crop = restore_with_landmark(
                roi.crop, gpen, landmarker, weight, use_correction
            )
            restored_frame = blend_roi(restored_frame, restored_crop, roi)
        restore_ms = (time.time() - t_restore) * 1000

        # Display
        compressed_display = compressed.copy()
        for roi in cached_rois:
            cv2.rectangle(compressed_display, (roi.x1, roi.y1), (roi.x2, roi.y2),
                          (0, 255, 0), 2)

        # Metrics
        frame_idx += 1
        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now
        if now - bitrate_time >= 1.0:
            bitrate_kbps = bytes_acc * 8 / (now - bitrate_time) / 1000.0
            bytes_acc = 0
            bitrate_time = now
        loop_ms = (now - t0) * 1000.0

        # OSD
        lm_status = "LM:ON" if use_correction else "LM:OFF"
        cv2.putText(original, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(compressed_display, f"Compressed CRF={crf}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        trk_name = "OptFlow" if use_optflow else "Kalman+MOSSE"
        cv2.putText(restored_frame, f"GPEN-256 [{lm_status}] [{trk_name}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        osd = [
            f"FPS: {fps:.1f} | Loop: {loop_ms:.0f}ms",
            f"Bitrate: {bitrate_kbps:.0f}kbps | W:{weight}",
            f"Faces: {len(cached_rois)} | Restore: {restore_ms:.0f}ms | {lm_status}",
        ]
        for i, line in enumerate(osd):
            cv2.putText(restored_frame, line, (10, 460 - (len(osd) - 1 - i) * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

        display = np.hstack([original, compressed_display, restored_frame])
        cv2.imshow("GPEN-256 + Landmark", display)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            idx = (crf_cycle.index(crf) + 1) % len(crf_cycle) if crf in crf_cycle else 0
            crf = crf_cycle[idx]
            print(f"CRF -> {crf}")
        elif key == ord('w'):
            idx = (weight_cycle.index(weight) + 1) % len(weight_cycle) if weight in weight_cycle else 0
            weight = weight_cycle[idx]
            print(f"Weight -> {weight}")
        elif key == ord('f'):
            idx = (detect_freq_cycle.index(detect_every) + 1) % len(detect_freq_cycle) \
                if detect_every in detect_freq_cycle else 0
            detect_every = detect_freq_cycle[idx]
            print(f"Detect every -> {detect_every} frames")
        elif key == ord('l'):
            use_correction = not use_correction
            print(f"Landmark correction -> {'ON' if use_correction else 'OFF'}")
        elif key == ord('t'):
            use_optflow = not use_optflow
            print(f"Tracker -> {'OpticalFlow' if use_optflow else 'Kalman+MOSSE'}")
        elif key == ord('s'):
            cv2.imwrite(f"webcam_gpen_{save_idx:03d}.png", display)
            print(f"Saved webcam_gpen_{save_idx:03d}.png")
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
