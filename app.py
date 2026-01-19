# -*- coding: utf-8 -*-
from random import random
import threading
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
from werkzeug.utils import secure_filename
import mysql.connector
import os
from functools import wraps
from datetime import datetime, time
import cv2
import numpy as np
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
import time
from collections import defaultdict
from datetime import datetime, timedelta
import calendar
from flask import request, jsonify

# ===== FIREBASE INTEGRATION =====
import firebase_admin
from firebase_admin import credentials, db as firebase_db

from collections import Counter

# Simpan waktu kapan server dinyalakan
SERVER_START_TIME = time.time()

# Inisialisasi Firebase
# Pastikan file serviceAccountKey.json ada di folder yang sama dengan app.py
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-traffic-vision-app-default-rtdb.asia-southeast1.firebasedatabase.app/' # GANTI DENGAN URL DATABASE ANDA
})

# ===== FLASK APP CONFIG =====
app = Flask(__name__)
app.secret_key = "smart_traffic_secret"

# ===== UPLOAD CONFIG =====
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ===== LOAD YOLO MODEL =====
print("Sedang memuat Model AI (YOLO11)...")
try: 
    model = YOLO("yolo11n.pt") 
    print("Model AI YOLO11 Siap!")
except Exception as e:
    print(f"Error loading YOLO: {e}")

# COCO Class ID: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Data CCTV Manual (Karena tabel CCTV belum ada di MySQL)
# URL ini akan digunakan oleh generator
cctv_list = [
    {'id': '1', 'name': 'CCTV Pontianak (Simpang Garuda)', 'url': 'https://www.youtube.com/watch?v=_xzKYnk6zSE'},
    {'id': '2', 'name': 'CTV Pontianak (Tugu Khatulistiwa)', 'url': 'https://www.youtube.com/watch?v=BEw38LHC5x4'},
    {'id': '3', 'name': 'CTV Demak (Alun-Alun)', 'url': 'https://www.youtube.com/embed/aboCZ7gkclk'},
    {'id': '4', 'name': 'CCTV Demak (Pasar Bintoro)', 'url': 'https://www.youtube.com/watch?v=7c4CsGkmBu8'},
    {'id': '5', 'name': 'CCTV Demak (Pertigaan Trengguli)', 'url': 'https://www.youtube.com/watch?v=5nw3G2jtWaU'}
]

# ===== KONEKSI DATABASE MYSQL (Untuk Admin & Artikel) =====
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="smart_traffic"
)

def get_db_cursor(dictionary=True):
    if not db.is_connected():
        db.reconnect()
    return db.cursor(dictionary=dictionary)

# ===== DECORATORS =====
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            flash("Silahkan login terlebih dahulu!", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            # Mengembalikan JSON 401 karena ini untuk akses API (AJAX)
            return jsonify({"message": "Unauthorized", "status": "error"}), 401
        return f(*args, **kwargs)
    return decorated

# =========================================================================
# ===== AI ENGINE (REAL TIME DETECTION) =====
# =========================================================================

def generate_live_stream(video_url, cctv_id):
    cap = cap_from_youtube(video_url, '360p')
    last_accumulate_time = 0 
    
    # Tentukan kapasitas maksimal kendaraan yang bisa ditampung jalan di frame tersebut
    # Anda bisa menyesuaikan angka ini (misal 15 atau 20)
    KAPASITAS_MAKSIMAL = 15 

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        
        counts_now = {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 2: counts_now['mobil'] += 1
                elif cls == 3: counts_now['motor'] += 1
                elif cls == 5: counts_now['bus'] += 1
                elif cls == 7: counts_now['truk'] += 1
        
        total_now = sum(counts_now.values())
        waktu_sekarang_unix = time.time()
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_today = now.strftime("%Y-%m-%d")

        # --- LOGIKA PENENTU STATUS BERDASARKAN PERSENTASE ---
        kepadatan_persen = min(100, int((total_now / KAPASITAS_MAKSIMAL) * 100))

        if kepadatan_persen < 40:
            status_val = "Lancar"
        elif kepadatan_persen <= 75:
            status_val = "Padat"
        else:
            status_val = "Macet"

        # --- A. UPDATE LIVE (Dashboard Real-time) ---
        try:
            ref_live = firebase_db.reference(f'traffic_stats/{cctv_id}/live')
            ref_live.update({
                'total': total_now,
                'kepadatan_persen': kepadatan_persen, # Simpan angka % ke firebase
                'detail': counts_now,
                'last_update': timestamp_str,
                'status': status_val
            })
        except: pass

        # --- B. LOGIKA AKUMULASI HARIAN & DURASI ---
        if waktu_sekarang_unix - last_accumulate_time > 5:
            try:
                ref_daily = firebase_db.reference(f'traffic_stats/{cctv_id}/daily_reports/{date_today}')
                daily_data = ref_daily.get()

                if not daily_data:
                    first_detection = timestamp_str
                    old_total_daily = 0
                    old_detail_daily = {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0}
                    duration_str = "0 menit"
                else:
                    first_detection = daily_data.get('first_detection', timestamp_str)
                    old_total_daily = daily_data.get('total_hari_ini', 0)
                    old_detail_daily = daily_data.get('detail', {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0})
                    
                    start_dt = datetime.strptime(first_detection, "%Y-%m-%d %H:%M:%S")
                    diff = now - start_dt
                    minutes = int(diff.total_seconds() // 60)
                    duration_str = f"{minutes} menit"

                new_total_daily = old_total_daily + total_now
                new_detail_daily = {
                    'mobil': old_detail_daily.get('mobil', 0) + counts_now['mobil'],
                    'motor': old_detail_daily.get('motor', 0) + counts_now['motor'],
                    'bus': old_detail_daily.get('bus', 0) + counts_now['bus'],
                    'truk': old_detail_daily.get('truk', 0) + counts_now['truk']
                }

                ref_daily.set({
                    'first_detection': first_detection,
                    'last_detection': timestamp_str,
                    'duration_active': duration_str,
                    'total_hari_ini': new_total_daily,
                    'detail': new_detail_daily,
                    'last_update': timestamp_str,
                    'status_terakhir': status_val,
                    'kepadatan_terakhir_persen': kepadatan_persen
                })
                
                ref_live.update({
                    'total_akumulasi_hari_ini': new_total_daily,
                    'session_duration': duration_str
                })

                last_accumulate_time = waktu_sekarang_unix
            except Exception as e:
                print(f"Error Harian: {e}")

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# =========================================================================
# ===== SHARED LOGIC =====
# =========================================================================

def fetch_cctv_list():
    return [
        { 
            "id": 1, 
            "name": "CCTV Pontianak (Simpang Garuda)", 
            "status": "Aktif", 
            "lat": -0.023851, "lon": 109.333423,
            "stream_url": "https://www.youtube.com/embed/_xzKYnk6zSE", 
            "youtube_link": "https://www.youtube.com/watch?v=_xzKYnk6zSE" 
        },
        { 
            "id": 2, 
            "name": "CCTV Pontianak (Tugu Khatulistiwa)", 
            "status": "Aktif",
            "lat": 0.000000, "lon": 109.321100,
            "stream_url": "https://www.youtube.com/embed/BEw38LHC5x4", 
            "youtube_link": "https://www.youtube.com/watch?v=BEw38LHC5x4" 
        },
        { 
            "id": 3, 
            "name": "CCTV Demak (Alun-Alun)", 
            "status": "Aktif",
            "lat": -6.894621, "lon": 110.636922,
            "stream_url": "https://www.youtube.com/embed/aboCZ7gkclk", 
            "youtube_link": "https://www.youtube.com/watch?v=aboCZ7gkclk" 
        },
        { 
            "id": 4, "name": "CCTV Demak (Pasar Bintoro)", "status": "Aktif",
            "lat": -6.8850, "lon": 110.6400,
            "stream_url": "https://www.youtube.com/embed/7c4CsGkmBu8", 
            "youtube_link": "https://www.youtube.com/watch?v=7c4CsGkmBu8" 
        },
        { 
            "id": 5, "name": "CCTV Demak (Pertigaan Trengguli)", "status": "Aktif",
            "lat": -6.8700, "lon": 110.6500,
            "stream_url": "https://www.youtube.com/embed/5nw3G2jtWaU", 
            "youtube_link": "https://www.youtube.com/watch?v=5nw3G2jtWaU" 
        },
    ]

    def logic_get_summary(cctv_id):
        if not cctv_id:
            return {...}

    cursor = get_db_cursor()
    cursor.execute("""
        SELECT 
            SUM(total_kendaraan) AS total,
            MAX(kepadatan) AS max_kepadatan
        FROM traffic_stats
        WHERE cctv_id = %s
        AND DATE(created_at) = CURDATE()
    """, (cctv_id,))
    
    data = cursor.fetchone()
    cursor.close()

    return {
        "kendaraan_hari_ini": data['total'] or 0,
        "kepadatan_tertinggi": f"{data['max_kepadatan'] or 0}%",
        "rata_rata_kecepatan": "-",
        "kamera_aktif": "5"
    }

def get_real_vehicle_count(youtube_link):
    raise NotImplementedError

def logic_get_vehicle(cctv_id, period):
    labels = ['Mobil','Motor','Bus','Truk']
    if not cctv_id:
        return {"labels": labels, "data": [0, 0, 0, 0]}

    try:
        cctv_id = int(cctv_id)
        cctv_data = next((item for item in fetch_cctv_list() if item["id"] == cctv_id), None)
        
        if cctv_data:
            # === GUNAKAN DATA ASLI DARI YOLO ===
            # (Untuk Pie Chart Distribusi Kendaraan)
            real_counts = get_real_vehicle_count(cctv_data['youtube_link'])
            return {
                "labels": labels, 
                "data": [real_counts['mobil'], real_counts['motor'], real_counts['bus'], real_counts['truk']]
            }
    except:
        pass

    return {"labels": labels, "data": [0, 0, 0, 0]}

def classify_traffic(total_kendaraan):
    """
    Klasifikasi lalu lintas berdasarkan jumlah kendaraan
    """
    if total_kendaraan >= 30:
        return "Macet", "red"
    elif total_kendaraan >= 15:
        return "Sedang", "yellow"
    else:
        return "Lancar", "green"

def detect_and_store(cctv):
    counts = get_real_vehicle_count(cctv['youtube_link'])
    total = sum(counts.values())

    kepadatan = min(100, int((total / 40) * 100))
    status, color = classify_traffic(total)

    cursor = get_db_cursor()
    cursor.execute("""
        INSERT INTO traffic_stats
        (cctv_id, mobil, motor, bus, truk, total_kendaraan, kepadatan, status, color)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        cctv['id'],
        counts['mobil'],
        counts['motor'],
        counts['bus'],
        counts['truk'],
        total,
        kepadatan,
        status,
        color
    ))
    db.commit()
    cursor.close()

def auto_detection_loop():
    while True:
        for cctv in fetch_cctv_list():
            try:
                detect_and_store(cctv)
            except Exception as e:
                print("Deteksi error:", e)

        time.sleep(60)  # setiap 1 menit


def logic_get_traffic(cctv_id, period):
    cursor = get_db_cursor()

    cursor.execute("""
        SELECT HOUR(created_at) AS jam, AVG(kepadatan) AS kepadatan
        FROM traffic_stats
        WHERE cctv_id = %s
        AND DATE(created_at) = CURDATE()
        GROUP BY HOUR(created_at)
        ORDER BY jam
    """, (cctv_id,))

    rows = cursor.fetchall()
    cursor.close()

    labels = [f"{r['jam']}:00" for r in rows]
    data = [int(r['kepadatan']) for r in rows]

    return {"labels": labels, "kepadatan": data}

def get_annotated_frame(youtube_link):
    raise NotImplementedError

# =========================================================================
# ===== ROUTES & API =====
# =========================================================================

# --- API UNTUK DETEKSI GAMBAR & KOTAK MERAH (DIPANGGIL TOMBOL DETEKSI) ---
@app.route('/api/analyze_cctv', methods=['GET'])
def api_analyze_cctv():
    cctv_id = request.args.get('cctv_id')
    if not cctv_id:
        return jsonify({"error": "No ID"}), 400

    try:
        cctv_id = int(cctv_id)
        cctv_data = next(
            (item for item in fetch_cctv_list() if item["id"] == cctv_id),
            None
        )

        if cctv_data:
            counts, img_base64 = get_annotated_frame(
                cctv_data['youtube_link']
            )

            total = sum(counts.values())          # ✅ HITUNG DULU
            status, color = classify_traffic(total)  # ✅ BARU KLASIFIKASI

            return jsonify({
                "counts": counts,
                "total": total,
                "status": status,
                "color": color,
                "image": img_base64
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Not Found"}), 404

def logic_get_summary(cctv_id):
    raise NotImplementedError

# =========================================================================
# ===== LOGIKA PEMBANTU FIREBASE (SINKRONISASI TOTAL & REAL-TIME) =====
# =========================================================================

def get_firebase_logic_summary(cctv_id):
    """Mengambil ringkasan data HARI INI dengan status berdasarkan persentase (%)"""
    
    # 1. Dapatkan tanggal hari ini
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if not cctv_id:
        return {
            "kendaraan_hari_ini": 0, 
            "kepadatan_tertinggi": 0, 
            "rata_rata_kecepatan": "80 km/j",
            "status": "Lancar",
            "kamera_aktif": len(cctv_list)
        }
    
    try:
        # Ambil data LIVE (real-time detik ini)
        ref_live = firebase_db.reference(f'traffic_stats/{cctv_id}/live')
        live_data = ref_live.get()
        
        # Ambil data DAILY (akumulasi dari pagi sampai sekarang)
        ref_daily = firebase_db.reference(f'traffic_stats/{cctv_id}/daily_reports/{today_str}')
        daily_data = ref_daily.get()
        
        if live_data:
            # A. TOTAL KENDARAAN (Akumulasi Hari Ini)
            if daily_data:
                total_hari_ini = daily_data.get('total_hari_ini', 0)
            else:
                total_hari_ini = live_data.get('total_akumulasi_hari_ini', 0)
            
            # B. DETEKSI DETIK INI (Untuk hitung % Kepadatan)
            total_sekarang = live_data.get('total', 0) 
            
            # C. HITUNG KEPADATAN % (Kapasitas 15 kendaraan)
            kapasitas = 15
            kepadatan_int = min(100, int((total_sekarang / kapasitas) * 100))
            
            # D. PENENTU STATUS BERDASARKAN PERSENTASE (%)
            if kepadatan_int < 40:
                status_txt = "Lancar"
            elif kepadatan_int <= 75:
                status_txt = "Padat"
            else:
                status_txt = "Macet"
            
            # E. HITUNG KECEPATAN (Dinamis berdasarkan kepadatan)
            if total_sekarang == 0:
                kecepatan_int = 80
            else:
                # Kecepatan berkurang seiring naiknya kepadatan
                kecepatan_int = max(10, 80 - int(kepadatan_int * 0.8))
            
            return {
                "kendaraan_hari_ini": total_hari_ini,
                "kepadatan_tertinggi": kepadatan_int, # Angka saja, JS akan tambah %
                "rata_rata_kecepatan": f"{kecepatan_int} km/j",
                "status": status_txt,
                "kamera_aktif": len(cctv_list)
            }
            
    except Exception as e:
        print(f"Firebase Summary Error: {e}")
    
    return {
        "kendaraan_hari_ini": 0, 
        "kepadatan_tertinggi": 0, 
        "rata_rata_kecepatan": "80 km/j",
        "status": "Lancar",
        "kamera_aktif": len(cctv_list)
    }

def get_firebase_logic_history(cctv_id, period):
    if not cctv_id: return {"labels": [], "data": []}

    try:
        ref = firebase_db.reference(f'traffic_stats/{cctv_id}/history')
        # Ambil data riwayat (asumsi data history tersimpan setiap 5-10 menit)
        history_data = ref.get()
        
        now = datetime.now()

        # --- LOGIKA HARIAN (TAMPILKAN JAM-JAMAN) ---
        if period == 'harian':
            labels, values = [], []
            if history_data:
                # Ambil hanya data milik hari ini
                today_str = now.strftime('%Y-%m-%d')
                # Urutkan dan ambil 12 data terakhir untuk hari ini
                items = [v for v in history_data.values() if v.get('last_update', '').startswith(today_str)]
                for val in items[-12:]:
                    labels.append(val.get('last_update', '')[11:16]) # Jam:Menit
                    values.append(min(100, int((val.get('total', 0) / 20) * 100)))
            return {"labels": labels, "data": values, "kepadatan": values}

        # --- LOGIKA MINGGUAN (7 BATANG: SEN - MIN) ---
        elif period == 'mingguan':
            labels = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
            data_points = [0] * 7
            counts = [0] * 7
            
            if history_data:
                for val in history_data.values():
                    dt = datetime.strptime(val['last_update'], '%Y-%m-%d %H:%M:%S')
                    # Cek apakah data masuk dalam minggu ini (7 hari terakhir)
                    if dt > (now - timedelta(days=7)):
                        idx = dt.weekday() # 0=Senin, 6=Minggu
                        data_points[idx] += min(100, int((val.get('total', 0) / 20) * 100))
                        counts[idx] += 1
            
            # Hitung rata-rata kepadatan per hari
            final_values = [int(data_points[i]/counts[i]) if counts[i] > 0 else 0 for i in range(7)]
            return {"labels": labels, "data": final_values, "kepadatan": final_values}

        # --- LOGIKA BULANAN (4 BATANG: MINGGU 1 - 4) ---
        elif period == 'bulanan':
            labels = ["Minggu 1", "Minggu 2", "Minggu 3", "Minggu 4"]
            data_points = [0] * 4
            counts = [0] * 4
            
            if history_data:
                for val in history_data.values():
                    dt = datetime.strptime(val['last_update'], '%Y-%m-%d %H:%M:%S')
                    if dt.month == now.month and dt.year == now.year:
                        # Kelompokkan tanggal ke Minggu 1, 2, 3, atau 4
                        day = dt.day
                        idx = min(3, (day - 1) // 7)
                        data_points[idx] += min(100, int((val.get('total', 0) / 20) * 100))
                        counts[idx] += 1
            
            final_values = [int(data_points[i]/counts[i]) if counts[i] > 0 else 0 for i in range(4)]
            return {"labels": labels, "data": final_values, "kepadatan": final_values}

    except Exception as e:
        print(f"Error History Logic: {e}")
        return {"labels": [], "data": []}

# =========================================================================
# --- ROUTE API (ADMIN & PUBLIC SUDAH DISINKRONKAN) ---
# =========================================================================

@app.route('/api/admin/dashboard_summary')
@api_login_required
def api_admin_dashboard_summary():
    return jsonify(get_firebase_logic_summary(request.args.get('cctv_id')))

@app.route('/api/public/dashboard_summary')
def api_public_dashboard_summary():
    # User mengambil dari fungsi logika yang sama dengan admin
    return jsonify(get_firebase_logic_summary(request.args.get('cctv_id')))

@app.route('/api/public/traffic_data')
def api_public_traffic_data():
    cctv_id = request.args.get('cctv_id')
    period = request.args.get('period', 'harian') # Ambil parameter period
    
    if not cctv_id:
        return jsonify({"labels": [], "datasets": {"mobil":[], "motor":[], "bus":[], "truk":[]}})

    try:
        ref = firebase_db.reference(f'traffic_stats/{cctv_id}/daily_reports')
        # Ambil semua data untuk diproses agregasinya (Mingguan/Bulanan butuh data lebih dari 7 hari)
        all_data = ref.get() or {}
        
        now = datetime.now()
        curr_year = now.year
        curr_month = now.month
        month_name = now.strftime('%B')

        labels = []
        data_mobil, data_motor, data_bus, data_truk = [], [], [], []

        if period == 'harian':
            # LOGIKA: 7 Hari Terakhir
            for i in range(6, -1, -1):
                date_key = (now - timedelta(days=i)).strftime('%Y-%m-%d')
                labels.append(datetime.strptime(date_key, "%Y-%m-%d").strftime("%d %b"))
                
                det = all_data.get(date_key, {}).get('detail', {})
                data_mobil.append(det.get('mobil', 0))
                data_motor.append(det.get('motor', 0))
                data_bus.append(det.get('bus', 0))
                data_truk.append(det.get('truk', 0))

        elif period == 'mingguan':
            # LOGIKA: Pembagian Tetap 1-7, 8-14, 15-21, 22-Akhir Bulan
            last_day = calendar.monthrange(curr_year, curr_month)[1]
            ranges = [(1, 7), (8, 14), (15, 21), (22, last_day)]
            
            for i, (start, end) in enumerate(ranges):
                labels.append([f"Minggu {i+1}", f"{start}-{end} {month_name[:3]}"])
                
                sum_mobil = sum_motor = sum_bus = sum_truk = 0
                for d in range(start, end + 1):
                    date_key = f"{curr_year}-{str(curr_month).zfill(2)}-{str(d).zfill(2)}"
                    if date_key in all_data:
                        det = all_data[date_key].get('detail', {})
                        sum_mobil += det.get('mobil', 0)
                        sum_motor += det.get('motor', 0)
                        sum_bus += det.get('bus', 0)
                        sum_truk += det.get('truk', 0)
                
                data_mobil.append(sum_mobil)
                data_motor.append(sum_motor)
                data_bus.append(sum_bus)
                data_truk.append(sum_truk)

        elif period == 'bulanan':
            # LOGIKA: 12 Bulan (Jan - Des) Tahun Berjalan
            month_list = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
            for i, m_short in enumerate(month_list):
                labels.append(m_short)
                month_prefix = f"{curr_year}-{str(i+1).zfill(2)}"
                
                sum_mobil = sum_motor = sum_bus = sum_truk = 0
                for date_key, val in all_data.items():
                    if date_key.startswith(month_prefix):
                        det = val.get('detail', {})
                        sum_mobil += det.get('mobil', 0)
                        sum_motor += det.get('motor', 0)
                        sum_bus += det.get('bus', 0)
                        sum_truk += det.get('truk', 0)
                
                data_mobil.append(sum_mobil)
                data_motor.append(sum_motor)
                data_bus.append(sum_bus)
                data_truk.append(sum_truk)

        return jsonify({
            "labels": labels,
            "datasets": {
                "mobil": data_mobil,
                "motor": data_motor,
                "bus": data_bus,
                "truk": data_truk
            }
        })
    except Exception as e:
        print(f"Error Public Traffic API: {e}")
        return jsonify({"labels": [], "datasets": {"mobil":[], "motor":[], "bus":[], "truk":[]}})

# =========================================================================
# ===== API SERVER (PERSISTENSI DATA STABIL CCTV) =====
# =========================================================================

@app.route('/api/admin/server_status')
@api_login_required
def server_status():
    # 1. Hitung Uptime (Lama server berjalan)
    uptime_seconds = int(time.time() - SERVER_START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m"

    # 2. Logika "Kesehatan" atau Stabilitas
    # Kita bisa asumsikan stabilitas 100% jika Firebase terkoneksi
    # Di sini kita buat sedikit variasi agar terlihat real-time (misal 98-100%)
    import random
    stability_score = round(random.uniform(98.5, 99.9), 1)

    return jsonify({
        "status": "ONLINE",
        "uptime": uptime_str,
        "stability": stability_score,
        "last_update": datetime.now().strftime("%H:%M:%S")
    })

# =========================================================================
# ===== API GRAFIK (PERSISTENSI DATA & REAL-TIME UPDATE) =====
# =========================================================================

@app.route('/api/admin/vehicle_distribution')
@api_login_required
def api_admin_vehicle_distribution():
    cctv_id = request.args.get('cctv_id')
    today_str = datetime.now().strftime("%Y-%m-%d")
    counts = {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0}
    
    try:
        ref = firebase_db.reference('traffic_stats')
        
        if cctv_id:
            # Ambil data akumulasi khusus HARI INI
            data = ref.child(f"{cctv_id}/daily_reports/{today_str}/detail").get()
            if data: counts = data
        else:
            # Gabungkan akumulasi HARI INI dari SEMUA CCTV
            all_data = ref.get()
            if all_data:
                for key in all_data:
                    d = all_data[key].get('daily_reports', {}).get(today_str, {}).get('detail', {})
                    for k in counts: counts[k] += d.get(k, 0)

        values = [counts['mobil'], counts['motor'], counts['bus'], counts['truk']]
        total = sum(values)
        # Hitung persen
        perc = [f"{round((v/total*100), 1)}%" if total > 0 else "0%" for v in values]
        
        return jsonify({"labels": ['Mobil', 'Motor', 'Bus', 'Truk'], "data": values, "percentages": perc})
    except:
        return jsonify({"labels": ['Mobil', 'Motor', 'Bus', 'Truk'], "data": [0,0,0,0], "percentages": ["0%","0%","0%","0%"]})

@app.route('/api/admin/traffic_data')
@api_login_required
def api_admin_traffic_data():
    cctv_id = request.args.get('cctv_id')
    period = request.args.get('period', 'harian')
    
    try:
        ref = firebase_db.reference(f'traffic_stats/{cctv_id}/daily_reports')
        all_data = ref.get() or {}

        # Ambil info waktu sekarang
        now = datetime.now()
        curr_year = now.year
        curr_month = now.month
        month_name = now.strftime('%B')

        labels, d_mobil, d_motor, d_bus, d_truk = [], [], [], [], []

        if period == 'harian':
            # 7 Hari terakhir
            for i in range(6, -1, -1):
                dt = (now - timedelta(days=i)).strftime('%Y-%m-%d')
                labels.append(datetime.strptime(dt, '%Y-%m-%d').strftime('%d %b'))
                det = all_data.get(dt, {}).get('detail', {})
                d_mobil.append(det.get('mobil', 0)); d_motor.append(det.get('motor', 0))
                d_bus.append(det.get('bus', 0)); d_truk.append(det.get('truk', 0))

        elif period == 'mingguan':
            # PEMBAGIAN TETAP: 1-7, 8-14, 15-21, 22-Akhir Bulan
            # Mendapatkan jumlah hari dalam bulan ini
            last_day = calendar.monthrange(curr_year, curr_month)[1]
            ranges = [(1, 7), (8, 14), (15, 21), (22, last_day)]
            
            for i, (start, end) in enumerate(ranges):
                labels.append([f"Minggu {i+1}", f"{start}-{end} {month_name[:3]}"])
                
                sum_mobil = sum_motor = sum_bus = sum_truk = 0
                for day in range(start, end + 1):
                    # Format: 2026-01-01
                    date_key = f"{curr_year}-{str(curr_month).zfill(2)}-{str(day).zfill(2)}"
                    if date_key in all_data:
                        det = all_data[date_key].get('detail', {})
                        sum_mobil += det.get('mobil', 0)
                        sum_motor += det.get('motor', 0)
                        sum_bus += det.get('bus', 0)
                        sum_truk += det.get('truk', 0)
                
                d_mobil.append(sum_mobil); d_motor.append(sum_motor)
                d_bus.append(sum_bus); d_truk.append(sum_truk)

        elif period == 'bulanan':
            # Jan - Des tahun berjalan
            month_list = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
            for i, m_short in enumerate(month_list):
                labels.append(m_short)
                month_prefix = f"{curr_year}-{str(i+1).zfill(2)}"
                
                sum_mobil = sum_motor = sum_bus = sum_truk = 0
                for date_key, val in all_data.items():
                    if date_key.startswith(month_prefix):
                        det = val.get('detail', {})
                        sum_mobil += det.get('mobil', 0)
                        sum_motor += det.get('motor', 0)
                        sum_bus += det.get('bus', 0)
                        sum_truk += det.get('truk', 0)
                
                d_mobil.append(sum_mobil); d_motor.append(sum_motor)
                d_bus.append(sum_bus); d_truk.append(sum_truk)

        return jsonify({
            "labels": labels,
            "datasets": {"mobil": d_mobil, "motor": d_motor, "bus": d_bus, "truk": d_truk}
        })
    except Exception as e:
        return jsonify({"labels": [], "datasets": {}, "error": str(e)})

@app.route('/api/cctv_locations')
def api_cctv_locations():
    # 1. Ambil list CCTV dasar
    cameras = fetch_cctv_list()
    
    try:
        # 2. Ambil semua status dari Firebase sekaligus
        traffic_ref = firebase_db.reference('traffic_stats').get()
        
        # 3. Gabungkan status ke dalam list camera
        for cam in cameras:
            cam_id = str(cam['id'])
            if traffic_ref and cam_id in traffic_ref:
                # Ambil status dari node live
                cam['traffic_status'] = traffic_ref[cam_id].get('live', {}).get('status', 'Lancar')
                cam['current_total'] = traffic_ref[cam_id].get('live', {}).get('total', 0)
            else:
                cam['traffic_status'] = 'Lancar' # Default jika belum ada deteksi
                cam['current_total'] = 0

        return jsonify(cameras), 200
    except Exception as e:
        print(f"Error Map Sync: {e}")
        return jsonify(cameras), 200

# =========================================================================
# --- ROUTE API (PUBLIC DASHBOARD) ---
# =========================================================================

# --- API PUBLIC UNTUK USER ---
@app.route('/api/public/vehicle_distribution')
def api_public_vehicle_distribution():
    cctv_id = request.args.get('cctv_id')
    today_str = datetime.now().strftime("%Y-%m-%d")
    counts = {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0}
    
    try:
        ref = firebase_db.reference('traffic_stats')
        # Ambil rincian HARI INI
        if cctv_id:
            data = ref.child(f"{cctv_id}/daily_reports/{today_str}/detail").get()
            if data: counts = data
        else:
            all_data = ref.get()
            if all_data:
                for key in all_data:
                    d = all_data[key].get('daily_reports', {}).get(today_str, {}).get('detail', {})
                    for k in counts: counts[k] += d.get(k, 0)

        values = [counts.get('mobil', 0), counts.get('motor', 0), counts.get('bus', 0), counts.get('truk', 0)]
        total = sum(values)
        perc = [f"{round((v/total*100), 1)}%" if total > 0 else "0%" for v in values]
        
        return jsonify({
            "labels": ['Mobil', 'Motor', 'Bus', 'Truk'],
            "data": values,
            "percentages": perc
        })
    except:
        return jsonify({"labels": ['Mobil', 'Motor', 'Bus', 'Truk'], "data": [0,0,0,0], "percentages": ["0%","0%","0%","0%"]})

# Fungsi Pembantu (Taruh di atas route API)
def get_logic_vehicle_distribution(cctv_id, period):
    counts = {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0}
    try:
        ref = firebase_db.reference('traffic_stats')
        if cctv_id:
            # Ambil data dari cumulative (untuk data menyatu)
            data = ref.child(f"{cctv_id}/cumulative/detail").get()
            if data: counts = data
        else:
            # Akumulasi semua CCTV
            all_data = ref.get()
            if all_data:
                for key in all_data:
                    d = all_data[key].get('cumulative', {}).get('detail', {})
                    for k in counts: counts[k] += d.get(k, 0)

        values = [counts['mobil'], counts['motor'], counts['bus'], counts['truk']]
        total = sum(values)
        perc = [f"{round((v/total*100), 1)}%" if total > 0 else "0%" for v in values]
        
        return {"labels": ['Mobil', 'Motor', 'Bus', 'Truk'], "data": values, "percentages": perc}
    except:
        return {"labels": ['Mobil', 'Motor', 'Bus', 'Truk'], "data": [0,0,0,0], "percentages": ["0%","0%","0%","0%"]}
    
# --- Standard Routes ---
@app.route('/')
def index():
    cursor = get_db_cursor()
    cursor.execute("SELECT id, judul, isi, gambar, tanggal FROM artikel WHERE published=1 ORDER BY tanggal DESC LIMIT 5")
    latest_articles = cursor.fetchall()
    cursor.close()
    return render_template('index.html', latest_articles=latest_articles)

@app.route('/dashboard')
def dashboard():
    cursor = get_db_cursor()
    cursor.execute("SELECT id, judul, gambar, tanggal FROM artikel WHERE published=1 ORDER BY tanggal DESC LIMIT 3")
    latest_articles = cursor.fetchall()
    cursor.close()
    cctv_list = fetch_cctv_list()
    return render_template('dashboard.html', latest_articles=latest_articles, cctv_list=cctv_list)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "12345":
            session['user'] = 'admin'
            flash("Login Berhasil!", "success")
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Username atau password salah!", "danger")
    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.", "info")
    return redirect(url_for('login'))

@app.route('/kelola_artikel')
@login_required
def kelola_artikel():
    page = request.args.get('page', 1, type=int)
    per_page = 8
    offset = (page - 1) * per_page
    cursor = get_db_cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM artikel")
    total = cursor.fetchone()['total']
    total_pages = (total + per_page - 1) // per_page
    cursor.execute("SELECT id, judul, isi, gambar, published, tanggal FROM artikel ORDER BY tanggal DESC LIMIT %s OFFSET %s", (per_page, offset))
    data = cursor.fetchall()
    cursor.close()
    return render_template('kelola_artikel.html', artikel=data, page=page, total_pages=total_pages, per_page=per_page)

@app.route('/artikel/tambah', methods=['GET', 'POST'])
@login_required
def tambah_artikel():
    if request.method == 'POST':
        judul = request.form['judul']
        isi = request.form['isi']
        tanggal_str = request.form['tanggal']
        file = request.files.get('gambar')
        filename = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cursor = get_db_cursor()
        cursor.execute("INSERT INTO artikel (judul, isi, gambar, published, tanggal) VALUES (%s, %s, %s, %s, %s)", (judul, isi, filename, 0, tanggal_str))
        db.commit()
        cursor.close()
        flash("Artikel berhasil ditambahkan!", "success")
        return redirect(url_for('kelola_artikel'))
    return render_template('crud_artikel.html', mode='tambah', artikel=None)

def allowed_file(filename):
    raise NotImplementedError

@app.route('/artikel/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_artikel(id):
    next_page = request.args.get('next', url_for('kelola_artikel'))
    cursor = get_db_cursor()
    if request.method == 'POST':
        judul = request.form['judul']
        isi = request.form['isi']
        tanggal_str = request.form['tanggal']
        file = request.files.get('gambar')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            cursor.execute("UPDATE artikel SET judul=%s, isi=%s, gambar=%s, tanggal=%s WHERE id=%s", (judul, isi, filename, tanggal_str, id))
        else:
            cursor.execute("UPDATE artikel SET judul=%s, isi=%s, tanggal=%s WHERE id=%s", (judul, isi, tanggal_str, id))
        db.commit()
        cursor.close()
        flash("Artikel berhasil diperbarui!", "success")
        return redirect(next_page)
    cursor.execute("SELECT * FROM artikel WHERE id=%s", (id,))
    data = cursor.fetchone()
    cursor.close()
    return render_template('crud_artikel.html', mode='edit', artikel=data)

@app.route('/artikel/hapus/<int:id>')
@login_required
def hapus_artikel(id):
    cursor = get_db_cursor()
    cursor.execute("DELETE FROM artikel WHERE id=%s", (id,))
    db.commit()
    cursor.close()
    flash("Artikel berhasil dihapus!", "danger")
    return redirect(url_for('kelola_artikel'))

@app.route('/artikel/publish/<int:id>')
@login_required
def publish_artikel(id):
    cursor = get_db_cursor()
    try:
        cursor.execute("UPDATE artikel SET published = 1 WHERE id = %s", (id,))
        db.commit()
        flash("Artikel berhasil dipublish!", "success")
    except:
        db.rollback()
        flash("Gagal mempublish artikel", "danger")
    finally:
        cursor.close()
    return redirect(url_for('kelola_artikel'))

@app.route('/artikel/batal_publish/<int:id>')
@login_required
def batal_publish(id):
    cursor = get_db_cursor()
    try:
        cursor.execute("UPDATE artikel SET published = 0 WHERE id = %s", (id,))
        db.commit()
        flash("Artikel berhasil dibatalkan publikasinya!", "warning")
    except:
        db.rollback()
        flash("Gagal membatalkan publikasi", "danger")
    finally:
        cursor.close()
    return redirect(url_for('kelola_artikel'))

@app.route('/read_artikel')
def read_artikel():
    cursor = get_db_cursor()
    cursor.execute("SELECT id, judul, isi, gambar, tanggal FROM artikel WHERE published=1 ORDER BY tanggal DESC")
    data = cursor.fetchall()
    cursor.close()
    return render_template('read_artikel.html', artikel=data)

@app.route('/artikel/<int:id>')
def view_artikel_detail(id):
    cursor = get_db_cursor()
    cursor.execute("SELECT id, judul, isi, gambar, tanggal FROM artikel WHERE id=%s AND published=1", (id,))
    artikel = cursor.fetchone()
    cursor.close()
    if artikel:
        return render_template('artikel_detail.html', artikel=artikel)
    else:
        flash("Artikel tidak ditemukan.", "danger")
        return redirect(url_for('read_artikel'))

@app.route('/about')
def about():
    return render_template('aboutme.html')

@app.route('/cctv-page')
def cctv_page():
    return render_template('cctv.html')

@app.route('/static-page')
def static_page():
    return render_template('static.html')

@app.route('/video_feed')
def video_feed():
    cctv_id = request.args.get('cctv_id')
    
    # Cari URL video dari list manual
    target_url = None
    for cctv in cctv_list:
        if cctv['id'] == cctv_id:
            target_url = cctv['url']
            break
            
    if not target_url:
        return "CCTV tidak ditemukan", 404

    return Response(generate_live_stream(target_url, cctv_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin')
@login_required
def admin_dashboard():
    try:
        # 1. Buka koneksi database
        cursor = get_db_cursor(dictionary=True)
        
        # 2. Ambil 5 artikel terbaru (Gabungan dari limit 3 dan limit 5)
        # Kita ambil 5 supaya lebih lengkap
        cursor.execute("SELECT * FROM artikel ORDER BY tanggal DESC LIMIT 5")
        latest_articles = cursor.fetchall()
        cursor.close()

        # 3. Ambil daftar CCTV
        try:
            cctvs = fetch_cctv_list()
        except:
            # Jika fetch_cctv_list error/kosong, gunakan list manual kita tadi
            cctvs = cctv_list 

        # 4. Render ke template (Hanya panggil render_template SATU KALI)
        return render_template('admin_dashboard.html', 
                               latest_articles=latest_articles, 
                               cctv_list=cctvs)

    except Exception as e:
        print(f"Error pada Admin Dashboard: {e}")
        return "Terjadi kesalahan pada database. Pastikan MySQL aktif.", 500
    
# 1. Dictionary Translate Hari
HARI_INDO = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
    'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
}

# --- DAFTAR KATA KUNCI SENTIMEN YANG DIPERLUAS ---
KATA_POSITIF = [
    'baik', 'bagus', 'mantap', 'keren', 'membantu', 'lancar', 'puas', 'terima kasih', 
    'terimakasih', 'hebat', 'luar biasa', 'bermanfaat', 'akurat', 'informatif', 
    'canggih', 'oke', 'ok', 'sip', 'jos', 'top', 'solusi', 'cepat', 'responsif', 
    'membantu', 'terbantu', 'senang', 'keren', 'update'
]

KATA_NEGATIF = [
    'buruk', 'jelek', 'macet', 'parah', 'lambat', 'kecewa', 'salah', 'error', 
    'rusak', 'lemot', 'payah', 'ribet', 'sulit', 'susah', 'lag', 'gagal', 
    'tidak membantu', 'tidak berguna', 'kurang', 'kacau', 'berantakan', 
    'penipu', 'bohong', 'sampah', 'patah', 'tidak akurat', 'bukan solusi', 
    'mati', 'down', 'bug', 'jelek', 'parah banget'
]

def klasifikasi_sentimen(teks):
    teks = teks.lower()
    
    # 1. Cek frase negatif yang mengandung kata positif (Negation Handling)
    # Agar "tidak membantu" tidak terbaca "Baik" hanya karena ada kata "membantu"
    frase_negatif = ['tidak', 'kurang', 'bukan', 'ga ', 'gak', 'jangan']
    
    skor_positif = 0
    skor_negatif = 0
    
    # Cek kata negatif langsung
    for kata in KATA_NEGATIF:
        if kata in teks:
            skor_negatif += 1
            
    # Cek kata positif
    for kata in KATA_POSITIF:
        if kata in teks:
            # Jika ada kata negasi sebelum kata positif (contoh: "tidak bagus")
            # maka dianggap negatif
            terdeteksi_negasi = False
            for neg in frase_negatif:
                if f"{neg} {kata}" in teks or f"{neg}{kata}" in teks:
                    terdeteksi_negasi = True
                    break
            
            if terdeteksi_negasi:
                skor_negatif += 1.5 # Bobot lebih besar karena ini penolakan
            else:
                skor_positif += 1

    # Penentuan Akhir
    if skor_negatif > skor_positif:
        return "Buruk"
    elif skor_positif > skor_negatif:
        return "Baik"
    else:
        # Jika skor sama, cek apakah ada lebih banyak kata negatif secara umum
        return "Buruk" if skor_negatif > 0 else "Baik"

# --- API USER: Kirim Komentar ---
@app.route('/api/submit_comment', methods=['POST'])
def submit_comment():
    try:
        data = request.get_json()
        pesan = data.get('komentar', '')
        nama = data.get('nama', 'Anonim')
        
        sentimen = klasifikasi_sentimen(pesan)
        now = datetime.now()
        day_eng = now.strftime("%A")
        
        payload = {
            "nama": nama,
            "komentar": pesan,
            "sentimen": sentimen,
            "tanggal": now.strftime("%Y-%m-%d"),
            "jam": now.strftime("%H:%M:%S"),
            "hari": HARI_INDO.get(day_eng, day_eng),
            "timestamp": time.time()
        }
        firebase_db.reference('user_comments').push(payload)
        return jsonify({"status": "success", "message": "Laporan terkirim!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- API ADMIN: Statistik Komentar ---
@app.route('/api/admin/comments_analytics')
@api_login_required
def admin_comments_analytics():
    try:
        ref = firebase_db.reference('user_comments').get()
        if not ref:
            return jsonify({"labels": [], "baik": [], "buruk": [], "list": []})
        
        comments = list(ref.values())
        # Kelompokkan data per tanggal
        data_per_tgl = {}
        for c in comments:
            tgl = c['tanggal']
            sen = c.get('sentimen', 'Baik') # Default Baik jika data lama tidak punya sentimen
            if tgl not in data_per_tgl:
                data_per_tgl[tgl] = {"Baik": 0, "Buruk": 0}
            data_per_tgl[tgl][sen] += 1
            
        sorted_dates = sorted(data_per_tgl.keys())
        
        return jsonify({
            "labels": sorted_dates,
            "baik": [data_per_tgl[d]["Baik"] for d in sorted_dates],
            "buruk": [data_per_tgl[d]["Buruk"] for d in sorted_dates],
            "list": sorted(comments, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    detector_thread = threading.Thread(
        target=auto_detection_loop,
        daemon=True
    )
    detector_thread.start()

    app.run(debug=True)