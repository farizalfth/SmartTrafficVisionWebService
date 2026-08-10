# -*- coding: utf-8 -*-
"""AI Engine Server — Smart Traffic Vision.

Server Python minimal yang hanya menjalankan deteksi YOLO11 dan
menulis hasilnya ke Firebase Realtime Database. Tidak lagi memakai MySQL.

Endpoint:
    /video_feed?cctv_id=<id>   Live stream hasil deteksi AI (MJPEG)
    /api/analyze_cctv?cctv_id=<id>  Deteksi satu frame + bounding box (base64)
    /api/cctv_list             Daftar CCTV dari Firebase

Data CCTV dibaca dari Firebase node `cctv` (hasil migrasi MySQL).
"""
import base64
import io
import os
import re
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ===== FIREBASE =====
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-traffic-vision-app-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# ===== FLASK =====
app = Flask(__name__)
CORS(app)

SERVER_START_TIME = time.time()

# ===== GAMBAR ARTIKEL (dilayani dari local disk) =====
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== YOLO =====
print("Memuat Model AI (YOLO11)...")
model = YOLO("yolo11n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]

# ===== PARAMETER ESTIMASI LALU LINTAS (Level 1-3) =====
# Level 1: kepadatan berbasis "occupancy ratio" (luas box ÷ area ROI), bukan jumlah objek.
# Level 2: kalibrasi per kamera (kapasitas, skala piksel/meter, ROI) dari Firebase.
# Level 3: tracking centroid antar frame untuk estimasi kecepatan & deteksi antrean.
DEFAULT_KAPASITAS = 15          # asumsi kendaraan maksimal jika tidak dikalibrasi
DEFAULT_PX_PER_M = 30           # skala default piksel per meter (kalibrasi per kamera lebih akurat)
FRAME_BURST = 10                # jumlah frame per siklus di detection_loop (untuk tracking)
SPEED_SLOW_MPS = 1.5            # < 1.5 m/s (~5 km/j) dianggap antrean/berhenti
EMA_ALPHA = 0.5                 # faktor smoothing eksponensial
STATUS_KEEP = 6                 # riwayat status utk voting mayoritas
OCC_LOW = 30                    # occupancy < 30% => trafik ringan (pasti Lancar)
OCC_HIGH = 55                   # occupancy >= 55% => padat/kemacetan
SPEED_MIN_SAMPLES = 3           # minimal sampel tracking agar kecepatan/antrean dipercaya
SIGNAL_FRESH_SEC = 120          # CCTV dianggap "sedang dideteksi" bila data live segar dlm 120 dtk

# State smoothing per kamera
_smooth_state = {}


def ema(cid, key, value, alpha=EMA_ALPHA):
    """Exponential moving average per kamera per metrik."""
    st = _smooth_state.setdefault(str(cid), {})
    prev = st.get(key)
    new = value if prev is None else prev * (1 - alpha) + value * alpha
    st[key] = new
    return new


def vote_status(cid, status):
    """Voting mayoritas status terakhir untuk mencegah flicker."""
    st = _smooth_state.setdefault(str(cid), {})
    hist = st.setdefault("status_hist", deque(maxlen=STATUS_KEEP))
    hist.append(status)
    return max(set(hist), key=hist.count)


def get_camera_config(cctv):
    """Baca konfigurasi kalibrasi per kamera dari Firebase (field pada node cctv)."""
    try:
        kapasitas = float(cctv.get("kapasitas") or DEFAULT_KAPASITAS)
    except (TypeError, ValueError):
        kapasitas = DEFAULT_KAPASITAS
    try:
        px_per_m = float(cctv.get("px_per_m") or DEFAULT_PX_PER_M)
    except (TypeError, ValueError):
        px_per_m = DEFAULT_PX_PER_M
    roi = cctv.get("roi")
    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        try:
            roi = [float(v) for v in roi]
            if roi[0] >= roi[2] or roi[1] >= roi[3]:
                roi = None
        except (TypeError, ValueError):
            roi = None
    else:
        roi = None
    return {"kapasitas": kapasitas, "px_per_m": px_per_m, "roi": roi}


def analyze_frame(frame, results, roi=None):
    """Hitung metrik satu frame: counts, total, occupancy (%), dan daftar box.

    Occupancy = rasio luas bounding box kendaraan terhadap area ROI jalan.
    ROI format: [left, top, right, bottom] ternormalisasi 0-1. Jika tidak ada,
    ROI default = seluruh frame.
    """
    frame_h, frame_w = frame.shape[:2]

    if roi and len(roi) == 4:
        x0 = roi[0] * frame_w
        y0 = roi[1] * frame_h
        x1 = roi[2] * frame_w
        y1 = roi[3] * frame_h
        roi_w = max(1.0, x1 - x0)
        roi_h = max(1.0, y1 - y0)
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(frame_w), float(frame_h)
        roi_w, roi_h = float(frame_w), float(frame_h)
    roi_area = roi_w * roi_h

    counts = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
    boxes = []
    occ_area = 0.0

    for r in results:
        b = r.boxes
        if b is None:
            continue
        for i, box in enumerate(b.xyxy):
            bx1, by1, bx2, by2 = (float(v) for v in box)
            ix0 = max(bx1, x0)
            iy0 = max(by1, y0)
            ix1 = min(bx2, x1)
            iy1 = min(by2, y1)
            if ix1 > ix0 and iy1 > iy0:
                occ_area += (ix1 - ix0) * (iy1 - iy0)
            cls = int(b.cls[i])
            if cls == 2:
                counts["mobil"] += 1
            elif cls == 3:
                counts["motor"] += 1
            elif cls == 5:
                counts["bus"] += 1
            elif cls == 7:
                counts["truk"] += 1
            boxes.append((bx1, by1, bx2, by2))

    total = sum(counts.values())
    occupancy = min(100.0, (occ_area / roi_area) * 100.0)
    return counts, total, occupancy, boxes


def iou_xyxy(a, b):
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = (a[2] - a[0]) * (a[3] - a[1]) + 1e-6
    ab = (b[2] - b[0]) * (b[3] - b[1]) + 1e-6
    return inter / (aa + ab - inter)


class SpeedTracker:
    """Tracking centroid + IOU ringan untuk estimasi kecepatan antar frame.

    Tidak bergantung tracker eksternal; aman untuk stream & burst per kamera.
    """

    def __init__(self, px_per_m=30.0, keep_secs=2.0):
        self.px_per_m = max(1.0, float(px_per_m))
        self.keep_secs = keep_secs
        self.next_id = 0
        self.tracks = {}  # tid -> {"box": (x1,y1,x2,y2), "t": epoch}

    def update(self, boxes, t, dt):
        """boxes: list (x1,y1,x2,y2). dt: selang antar frame (detik). Return speeds (m/s)."""
        if dt is None or dt <= 0 or dt > 1.5:
            dt = None
        speeds = []
        matched = set()

        for b in boxes:
            best_tid, best_iou = None, 0.15
            for tid, tr in self.tracks.items():
                if tid in matched:
                    continue
                i = iou_xyxy(b, tr["box"])
                if i > best_iou:
                    best_iou, best_tid = i, tid
            if best_tid is not None:
                matched.add(best_tid)
                prev_box = self.tracks[best_tid]["box"]
                self.tracks[best_tid] = {"box": b, "t": t}
                if dt is not None:
                    dx = (b[0] + b[2]) / 2 - (prev_box[0] + prev_box[2]) / 2
                    dy = (b[1] + b[3]) / 2 - (prev_box[1] + prev_box[3]) / 2
                    dist_px = (dx * dx + dy * dy) ** 0.5
                    if dist_px < 2.0:
                        dist_px = 0.0
                    speeds.append((dist_px / self.px_per_m) / dt)
            else:
                self.tracks[self.next_id] = {"box": b, "t": t}
                self.next_id += 1

        for tid in list(self.tracks):
            if t - self.tracks[tid]["t"] > self.keep_secs:
                del self.tracks[tid]
        return speeds


def speed_stats(speeds):
    """Ringkas kecepatan: (kecepatan rata-rata m/s atau None, rasio antrean 0-1)."""
    if not speeds:
        return None, 0.0
    avg = sum(speeds) / len(speeds)
    queue = sum(1 for s in speeds if s < SPEED_SLOW_MPS) / len(speeds)
    return avg, queue


def classify_status(total, occupancy, speed_kmh=None, queue_ratio=0.0, speed_samples=0):
    """Klasifikasi Lancar/Padat/Macet.

    Kepadatan (occupancy) adalah sinyal utama yang andal; kecepatan & antrean
    hanya dipakai bila ada cukup sampel tracking. Tanpa penjagaan ini, malam
    hari dengan sedikit kendaraan bisa salah diklasifikasikan "Macet" hanya
    karena kecepatan tak terukur (0 km/j) dan antrean menggelembung ke 100%.
    """
    if total == 0:
        return "Lancar"
    # Trafik ringan (occupancy rendah) selalu Lancar — kecepatan/antrean di
    # sini hampir pasti artefak tracking (sedikit kendaraan / lampu merah).
    if occupancy < OCC_LOW:
        return "Lancar"
    confident = speed_kmh is not None and speed_samples >= SPEED_MIN_SAMPLES
    if occupancy >= OCC_HIGH:
        # Padat tinggi; hanya boleh "Padat" bila arus benar-benar melaju.
        if confident and speed_kmh >= 30 and queue_ratio < 0.5:
            return "Padat"
        return "Macet"
    # Occupancy sedang (OCC_LOW..OCC_HIGH).
    if confident:
        if speed_kmh < 10 or (queue_ratio >= 0.5 and speed_kmh < 12):
            return "Macet"
        if speed_kmh < 30:
            return "Padat"
    return "Padat"

try:
    from cap_from_youtube import cap_from_youtube
except ImportError:
    cap_from_youtube = None


def fetch_cctv_list():
    """Baca daftar CCTV dari Firebase (node cctv)."""
    raw = firebase_db.reference("cctv").get() or {}
    cameras = []
    for key, r in raw.items():
        if not isinstance(r, dict):
            continue
        cameras.append({
            "id": r.get("id") or int(key.lstrip("c")),
            "name": r.get("name", ""),
            "status": r.get("status") or "Aktif",
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "url": r.get("url", ""),
            "youtube_link": r.get("url", ""),
            "kapasitas": r.get("kapasitas") or DEFAULT_KAPASITAS,
            "px_per_m": r.get("px_per_m") or DEFAULT_PX_PER_M,
            "roi": r.get("roi"),
        })
    return sorted(cameras, key=lambda c: c["id"])


def generate_live_stream(cctv):
    """Streaming MJPEG hasil deteksi + tulis data real-time ke Firebase."""
    cctv_id = cctv["id"]
    if cap_from_youtube is None:
        raise RuntimeError("cap_from_youtube tidak tersedia")
    cap = cap_from_youtube(cctv["youtube_link"], "360p")
    last_accumulate_time = 0
    cfg = get_camera_config(cctv)
    tracker = SpeedTracker(cfg["px_per_m"])
    prev_t = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        t = time.time()
        dt = (t - prev_t) if prev_t else None
        prev_t = t

        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        counts_now, total_now, occ, boxes = analyze_frame(frame, results, cfg["roi"])
        speeds = tracker.update(boxes, t, dt)
        avg_speed_mps, queue_ratio = speed_stats(speeds)
        speed_kmh = (avg_speed_mps * 3.6) if avg_speed_mps is not None else None

        occ_s = ema(cctv_id, "occ", occ)
        total_s = ema(cctv_id, "total", float(total_now))
        prev_speed = _smooth_state.get(str(cctv_id), {}).get("speed")
        speed_s = ema(cctv_id, "speed", speed_kmh if speed_kmh is not None else (prev_speed if prev_speed is not None else 0.0))
        queue_s = ema(cctv_id, "queue", queue_ratio)

        kepadatan_persen = min(100, int(round(occ_s)))
        status_val = classify_status(int(round(total_s)), occ_s, speed_s, queue_s)
        status_val = vote_status(cctv_id, status_val)

        waktu_sekarang_unix = time.time()
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_today = now.strftime("%Y-%m-%d")

        # --- A. UPDATE LIVE (Dashboard Real-time) ---
        try:
            ref_live = firebase_db.reference(f"traffic_stats/{cctv_id}/live")
            ref_live.update({
                "total": int(round(total_s)),
                "kepadatan_persen": kepadatan_persen,
                "occupancy_persen": round(occ_s, 1),
                "detail": counts_now,
                "kecepatan_kmh": round(speed_s, 1),
                "queue": round(queue_s, 2),
                "last_update": timestamp_str,
                "last_update_ts": int(waktu_sekarang_unix),
                "status": status_val,
            })
        except Exception:
            pass

        # --- B. AKUMULASI HARIAN ---
        if waktu_sekarang_unix - last_accumulate_time > 5:
            try:
                ref_daily = firebase_db.reference(f"traffic_stats/{cctv_id}/daily_reports/{date_today}")
                daily_data = ref_daily.get()

                if not daily_data:
                    first_detection = timestamp_str
                    old_total_daily = 0
                    old_detail_daily = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
                    duration_str = "0 menit"
                else:
                    first_detection = daily_data.get("first_detection", timestamp_str)
                    old_total_daily = daily_data.get("total_hari_ini", 0)
                    old_detail_daily = daily_data.get("detail", {"mobil": 0, "motor": 0, "bus": 0, "truk": 0})
                    start_dt = datetime.strptime(first_detection, "%Y-%m-%d %H:%M:%S")
                    diff = now - start_dt
                    duration_str = f"{int(diff.total_seconds() // 60)} menit"

                new_total_daily = old_total_daily + total_now
                new_detail_daily = {
                    "mobil": old_detail_daily.get("mobil", 0) + counts_now["mobil"],
                    "motor": old_detail_daily.get("motor", 0) + counts_now["motor"],
                    "bus": old_detail_daily.get("bus", 0) + counts_now["bus"],
                    "truk": old_detail_daily.get("truk", 0) + counts_now["truk"],
                }

                ref_daily.set({
                    "first_detection": first_detection,
                    "last_detection": timestamp_str,
                    "duration_active": duration_str,
                    "total_hari_ini": new_total_daily,
                    "detail": new_detail_daily,
                    "last_update": timestamp_str,
                    "status_terakhir": status_val,
                    "kepadatan_terakhir_persen": kepadatan_persen,
                })
                ref_live.update({
                    "total_akumulasi_hari_ini": new_total_daily,
                    "session_duration": duration_str,
                })
                last_accumulate_time = waktu_sekarang_unix
            except Exception as e:
                print(f"Error Harian: {e}")

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()


@app.route("/video_feed")
def video_feed():
    cctv_id = request.args.get("cctv_id")
    target = next((c for c in fetch_cctv_list() if str(c["id"]) == str(cctv_id)), None)
    if not target:
        return "CCTV tidak ditemukan", 404
    try:
        return Response(
            generate_live_stream(target),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        return f"Stream error: {e}", 500


@app.route("/api/analyze_cctv")
def analyze_cctv():
    cctv_id = request.args.get("cctv_id")
    target = next((c for c in fetch_cctv_list() if str(c["id"]) == str(cctv_id)), None)
    if not target:
        return jsonify({"error": "Not Found"}), 404

    if cap_from_youtube is None:
        return jsonify({"error": "cap_from_youtube tidak tersedia"}), 500

    try:
        cap = cap_from_youtube(target["youtube_link"], "360p")
        success, frame = cap.read()
        cap.release()
        if not success:
            return jsonify({"error": "Gagal mengambil frame"}), 500

        cfg = get_camera_config(target)
        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        counts, total, occ, _boxes = analyze_frame(frame, results, cfg["roi"])
        kepadatan = min(100, int(round(occ)))
        status = classify_status(total, occ)

        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")

        return jsonify({
            "counts": counts,
            "total": total,
            "status": status,
            "kepadatan": kepadatan,
            "occupancy_persen": round(occ, 1),
            "kecepatan_kmh": None,
            "image": img_base64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Terima upload gambar artikel, simpan ke static/uploads, kembalikan nama file."""
    if "image" not in request.files:
        return jsonify({"error": "File 'image' tidak ditemukan"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "File kosong"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "Ekstensi tidak diizinkan (png/jpg/jpeg/gif/webp)"}), 400
    if f.content_length and f.content_length > 2 * 1024 * 1024:
        return jsonify({"error": "Ukuran gambar maksimal 2MB"}), 400
    filename = f"{int(time.time())}_{secure_filename(f.filename)}"
    f.save(os.path.join(UPLOAD_DIR, filename))
    return jsonify({"filename": filename})


@app.route("/api/cctv_list")
def cctv_list():
    return jsonify(fetch_cctv_list())


@app.route("/api/cctv_config/<int:cid>", methods=["POST"])
def cctv_config(cid):
    """Simpan konfigurasi kalibrasi per CCTV (kapasitas, px_per_m, roi) ke Firebase.

    Body JSON: {kapasitas?, px_per_m?, roi?} — roi berupa [left, top, right, bottom]
    ternormalisasi 0-1. Kosongkan dengan [] untuk pakai seluruh frame.
    """
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    if not data:
        return jsonify({"error": "Body JSON kosong"}), 400

    payload = {}
    kapasitas = data.get("kapasitas")
    if kapasitas is not None:
        try:
            kapasitas = float(kapasitas)
        except (TypeError, ValueError):
            return jsonify({"error": "kapasitas harus angka"}), 400
        if not (1 <= kapasitas <= 200):
            return jsonify({"error": "kapasitas harus 1-200"}), 400
        payload["kapasitas"] = kapasitas

    px_per_m = data.get("px_per_m")
    if px_per_m is not None:
        try:
            px_per_m = float(px_per_m)
        except (TypeError, ValueError):
            return jsonify({"error": "px_per_m harus angka"}), 400
        if not (5 <= px_per_m <= 300):
            return jsonify({"error": "px_per_m harus 5-300"}), 400
        payload["px_per_m"] = px_per_m

    roi = data.get("roi")
    if roi is not None:
        if roi in ([], "", "none"):
            payload["roi"] = []
        elif isinstance(roi, (list, tuple)) and len(roi) == 4:
            try:
                roi_f = [float(v) for v in roi]
            except (TypeError, ValueError):
                return jsonify({"error": "roi harus 4 angka"}), 400
            if not all(0.0 <= v <= 1.0 for v in roi_f) or roi_f[0] >= roi_f[2] or roi_f[1] >= roi_f[3]:
                return jsonify({"error": "roi harus 0-1 dan left<right, top<bottom"}), 400
            payload["roi"] = roi_f
        else:
            return jsonify({"error": "roi harus [left, top, right, bottom]"}), 400

    if not payload:
        return jsonify({"error": "Tidak ada field valid untuk disimpan"}), 400

    try:
        key = f"c{int(cid)}"
        firebase_db.reference(f"cctv/{key}").update(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "saved": payload})


# ===== PENGUKURAN SINYAL CCTV (latensi nyata via thumbnail YouTube) =====
_SIGNAL_CACHE = {}
_SIGNAL_TTL = 15


def measure_signal(cctv):
    """Ukur sinyal CCTV secara real: latensi HEAD thumbnail YouTube.

    Returns (signal_persen, latency_ms, online).
    Signal dipakai warna hijau/kuning/merah di frontend.
    """
    cid = str(cctv["id"])
    if str(cctv.get("status", "")).lower() not in ("aktif", "online"):
        return 0.0, None, False

    now = time.time()
    cached = _SIGNAL_CACHE.get(cid)
    if cached and now - cached["t"] < _SIGNAL_TTL:
        return cached["signal"], cached["latency"], cached["online"]

    url = cctv.get("url") or cctv.get("youtube_link") or ""
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url or "")
    if not m:
        result = (0.0, None, False)
    else:
        thumb = f"https://i.ytimg.com/vi/{m.group(1)}/hqdefault.jpg"
        try:
            req = urllib.request.Request(thumb, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
            start = time.time()
            with urllib.request.urlopen(req, timeout=5) as resp:
                latency = (time.time() - start) * 1000
            if latency < 300:
                signal = 100.0
            elif latency < 700:
                signal = 88.0
            elif latency < 1200:
                signal = 70.0
            elif latency < 2000:
                signal = 50.0
            else:
                signal = 30.0
            result = (signal, round(latency, 1), True)
        except Exception:
            result = (0.0, None, False)

    _SIGNAL_CACHE[cid] = {"t": now, "signal": result[0], "latency": result[1], "online": result[2]}
    return result


@app.route("/api/server_status")
def server_status():
    """Status AI Server + sinyal real setiap CCTV (diukur langsung, bukan hardcode).

    Sinyal hanya dimasukkan untuk CCTV yang benar-benar sedang dideteksi
    (data live segar), sehingga sinyal muncul satu per satu seiring deteksi
    berjalan — bukan sekaligus semua CCTV.
    """
    uptime_seconds = int(time.time() - SERVER_START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    cams = fetch_cctv_list()
    now_ts = int(time.time())

    signals = []
    online_count = 0
    total_signal = 0.0
    latest_detection = None
    for c in cams:
        cid = str(c["id"])
        live = firebase_db.reference(f"traffic_stats/{cid}/live").get() or {}
        if not isinstance(live, dict):
            live = {}
        last = live.get("last_update")
        last_ts = live.get("last_update_ts")
        if last and (latest_detection is None or last > latest_detection):
            latest_detection = last
        # CCTV yang belum/berhenti dideteksi tidak ditampilkan sinyalnya.
        try:
            fresh = last_ts is not None and (now_ts - int(last_ts)) <= SIGNAL_FRESH_SEC
        except (TypeError, ValueError):
            fresh = False
        if not fresh:
            continue
        signal, latency, ok = measure_signal(c)
        if ok:
            online_count += 1
            total_signal += signal
        signals.append({
            "id": c["id"],
            "name": c["name"],
            "online": ok,
            "signal": round(signal, 1),
            "latency": latency,
            "last_update": last,
        })

    total = len(cams)
    stability = round(total_signal / len(signals), 1) if signals else 0.0
    if stability >= 90:
        label = "SANGAT STABIL"
    elif stability >= 70:
        label = "STABIL"
    elif stability >= 50:
        label = "OPTIMAL"
    elif stability >= 30:
        label = "DIPANTAU"
    else:
        label = "KRITIS"

    return jsonify({
        "status": "ONLINE",
        "status_label": label,
        "uptime": f"{hours}h {minutes}m",
        "stability": stability,
        "cctv_total": total,
        "cctv_online": online_count,
        "cctv_offline": total - online_count,
        "cctv_signals": signals,
        "last_update": latest_detection or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "server_time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
    })


def detection_loop():
    """Loop deteksi otomatis untuk semua CCTV setiap beberapa detik.

    Membaca burst beberapa frame berturut-turut per kamera agar tracking
    centroid bekerja untuk estimasi kecepatan & deteksi antrean.
    """
    while True:
        for cctv in fetch_cctv_list():
            try:
                cctv_id = cctv["id"]
                if str(cctv.get("status", "")).lower() not in ("aktif", "online"):
                    continue
                if cap_from_youtube is None:
                    continue
                cfg = get_camera_config(cctv)
                tracker = SpeedTracker(cfg["px_per_m"])
                cap = cap_from_youtube(cctv["youtube_link"], "360p")

                counts_sum = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
                totals = []
                occ_sum = 0.0
                occ_n = 0
                speeds_all = []
                last_counts = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
                prev_t = None
                frames = 0

                while frames < FRAME_BURST:
                    success, frame = cap.read()
                    if not success:
                        break
                    t = time.time()
                    dt = (t - prev_t) if prev_t else None
                    prev_t = t
                    results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
                    counts, total, occ, boxes = analyze_frame(frame, results, cfg["roi"])
                    for k in counts_sum:
                        counts_sum[k] += counts[k]
                    totals.append(total)
                    occ_sum += occ
                    occ_n += 1
                    last_counts = counts
                    speeds_all += tracker.update(boxes, t, dt)
                    frames += 1
                cap.release()

                if occ_n == 0:
                    continue

                avg_speed_mps, queue_ratio = speed_stats(speeds_all)
                speed_kmh = (avg_speed_mps * 3.6) if avg_speed_mps is not None else None
                occ_avg = occ_sum / occ_n
                total_avg = sum(totals) / len(totals)

                occ_s = ema(cctv_id, "occ", occ_avg)
                total_s = ema(cctv_id, "total", float(total_avg))
                queue_s = ema(cctv_id, "queue", queue_ratio)
                # Kecepatan hanya diperbarui bila ada pengukuran nyata; tanpa
                # pengukuran pertahankan nilai terakhir (None jika belum pernah).
                speed_s = (ema(cctv_id, "speed", speed_kmh) if speed_kmh is not None
                           else _smooth_state.get(str(cctv_id), {}).get("speed"))

                kepadatan = min(100, int(round(occ_s)))
                status = classify_status(int(round(total_s)), occ_s, speed_s, queue_s, len(speeds_all))
                status = vote_status(cctv_id, status)

                update = {
                    "total": int(round(total_s)),
                    "kepadatan_persen": kepadatan,
                    "occupancy_persen": round(occ_s, 1),
                    "detail": last_counts,
                    "queue": round(queue_s, 2),
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_update_ts": int(time.time()),
                    "status": status,
                }
                if speed_s is not None:
                    update["kecepatan_kmh"] = round(speed_s, 1)
                firebase_db.reference(f"traffic_stats/{cctv_id}/live").update(update)
            except Exception as e:
                print(f"Deteksi error {cctv.get('id')}: {e}")
        time.sleep(10)


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
