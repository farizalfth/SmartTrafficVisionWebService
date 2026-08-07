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
KAPASITAS_MAKSIMAL = 15

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
        })
    return sorted(cameras, key=lambda c: c["id"])


def generate_live_stream(cctv):
    """Streaming MJPEG hasil deteksi + tulis data real-time ke Firebase."""
    cctv_id = cctv["id"]
    if cap_from_youtube is None:
        raise RuntimeError("cap_from_youtube tidak tersedia")
    cap = cap_from_youtube(cctv["youtube_link"], "360p")
    last_accumulate_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)

        counts_now = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 2:
                    counts_now["mobil"] += 1
                elif cls == 3:
                    counts_now["motor"] += 1
                elif cls == 5:
                    counts_now["bus"] += 1
                elif cls == 7:
                    counts_now["truk"] += 1

        total_now = sum(counts_now.values())
        waktu_sekarang_unix = time.time()
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_today = now.strftime("%Y-%m-%d")

        kepadatan_persen = min(100, int((total_now / KAPASITAS_MAKSIMAL) * 100))
        if kepadatan_persen < 40:
            status_val = "Lancar"
        elif kepadatan_persen <= 75:
            status_val = "Padat"
        else:
            status_val = "Macet"

        # --- A. UPDATE LIVE (Dashboard Real-time) ---
        try:
            ref_live = firebase_db.reference(f"traffic_stats/{cctv_id}/live")
            ref_live.update({
                "total": total_now,
                "kepadatan_persen": kepadatan_persen,
                "detail": counts_now,
                "last_update": timestamp_str,
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

        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        counts = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 2:
                    counts["mobil"] += 1
                elif cls == 3:
                    counts["motor"] += 1
                elif cls == 5:
                    counts["bus"] += 1
                elif cls == 7:
                    counts["truk"] += 1

        total = sum(counts.values())
        kepadatan = min(100, int((total / KAPASITAS_MAKSIMAL) * 100))
        if kepadatan < 40:
            status = "Lancar"
        elif kepadatan <= 75:
            status = "Padat"
        else:
            status = "Macet"

        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")

        return jsonify({
            "counts": counts,
            "total": total,
            "status": status,
            "kepadatan": kepadatan,
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
    """Status AI Server + sinyal real setiap CCTV (diukur langsung, bukan hardcode)."""
    uptime_seconds = int(time.time() - SERVER_START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    cams = fetch_cctv_list()

    signals = []
    online_count = 0
    total_signal = 0.0
    latest_detection = None
    for c in cams:
        cid = str(c["id"])
        signal, latency, ok = measure_signal(c)
        live = firebase_db.reference(f"traffic_stats/{cid}/live").get() or {}
        last = live.get("last_update") if isinstance(live, dict) else None
        if last and (latest_detection is None or last > latest_detection):
            latest_detection = last
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
    stability = round(total_signal / total, 1) if total else 99.5
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
    """Loop deteksi otomatis untuk semua CCTV setiap beberapa detik."""
    while True:
        for cctv in fetch_cctv_list():
            try:
                if str(cctv.get("status", "")).lower() not in ("aktif", "online"):
                    continue
                if cap_from_youtube is None:
                    continue
                cap = cap_from_youtube(cctv["youtube_link"], "360p")
                success, frame = cap.read()
                cap.release()
                if not success:
                    continue
                results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
                counts = {"mobil": 0, "motor": 0, "bus": 0, "truk": 0}
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 2:
                            counts["mobil"] += 1
                        elif cls == 3:
                            counts["motor"] += 1
                        elif cls == 5:
                            counts["bus"] += 1
                        elif cls == 7:
                            counts["truk"] += 1
                total = sum(counts.values())
                kepadatan = min(100, int((total / KAPASITAS_MAKSIMAL) * 100))
                if kepadatan < 40:
                    status = "Lancar"
                elif kepadatan <= 75:
                    status = "Padat"
                else:
                    status = "Macet"
                firebase_db.reference(f"traffic_stats/{cctv['id']}/live").update({
                    "total": total,
                    "kepadatan_persen": kepadatan,
                    "detail": counts,
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status,
                })
            except Exception as e:
                print(f"Deteksi error {cctv.get('id')}: {e}")
        time.sleep(10)


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
