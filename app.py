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

# ===== FIREBASE INTEGRATION =====
import firebase_admin
from firebase_admin import credentials, db as firebase_db

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
    {'id': '1', 'name': 'CCTV Pontianak (Simpang Garuda)', 'url': 'https://www.youtube.com/watch?v=1s9cRcqZf58'},
    {'id': '2', 'name': 'CTV Pontianak (Tugu Khatulistiwa)', 'url': 'https://www.youtube.com/watch?v=oqSqC-gOALo'},
    {'id': '3', 'name': 'CTV Demak (Alun-Alun)', 'url': 'https://www.youtube.com/watch?v=mHk5UKckU7M'},
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
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        
        # Hitung deteksi detik ini
        counts_now = {'mobil': 0, 'motor': 0, 'bus': 0, 'truk': 0}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 2: counts_now['mobil'] += 1
                elif cls == 3: counts_now['motor'] += 1
                elif cls == 5: counts_now['bus'] += 1
                elif cls == 7: counts_now['truk'] += 1
        
        total_now = sum(counts_now.values())
        waktu_sekarang = time.time()

        # --- A. UPDATE LIVE (SETIAP FRAME) ---
        try:
            ref_live = firebase_db.reference(f'traffic_stats/{cctv_id}/live')
            ref_live.update({
                'total': total_now, # Ini yang membuat Kepadatan BERUBAH
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except:
            pass

        # --- B. LOGIKA AKUMULASI (SETIAP 5 DETIK) ---
        if waktu_sekarang - last_accumulate_time > 5:
            try:
                ref_cum = firebase_db.reference(f'traffic_stats/{cctv_id}/cumulative')
                old_data = ref_cum.get()
                if not old_data: old_data = {'grand_total': 0, 'detail': {'mobil':0,'motor':0,'bus':0,'truk':0}}

                old_total = old_data.get('grand_total', 0)
                new_total = old_total + total_now
                
                # Simpan Akumulasi
                ref_cum.set({
                    'grand_total': new_total,
                    'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Update total_akumulasi di node live
                ref_live.update({'total_akumulasi': new_total})
                
                last_accumulate_time = waktu_sekarang
            except Exception as e:
                print(f"Error Akumulasi: {e}")

        # Streaming Video
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
            "id": 1, "name": "CCTV Pontianak (Simpang Garuda)", "status": "Aktif", 
            "lat": -0.0245, "lon": 109.3406,
            "stream_url": "https://www.youtube.com/embed/1s9cRcqZf58", 
            "youtube_link": "https://www.youtube.com/watch?v=1s9cRcqZf58" 
        },
        { 
            "id": 2, "name": "CCTV Pontianak (Tugu Khatulistiwa)", "status": "Aktif",
            "lat": 0.0000, "lon": 109.3300,
            "stream_url": "https://www.youtube.com/embed/oqSqC-gOALo", 
            "youtube_link": "https://www.youtube.com/watch?v=oqSqC-gOALo" 
        },
        { 
            "id": 3, "name": "CCTV Demak (Alun-Alun)", "status": "Aktif",
            "lat": -6.8906, "lon": 110.6385,
            "stream_url": "https://www.youtube.com/embed/mHk5UKckU7M", 
            "youtube_link": "https://www.youtube.com/watch?v=mHk5UKckU7M" 
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
    """Fungsi tunggal untuk mengambil data ringkasan (Admin & User)"""
    if not cctv_id:
        return {
            "kendaraan_hari_ini": 0, "kepadatan_tertinggi": "0%", 
            "rata_rata_kecepatan": "80 km/j", "kamera_aktif": len(cctv_list)
        }
    
    try:
        # Ambil data LIVE dari Firebase
        ref = firebase_db.reference(f'traffic_stats/{cctv_id}/live')
        data = ref.get()
        
        if data:
            # 1. TOTAL KENDARAAN (Akumulasi yang menyatu)
            total_akumulasi = data.get('total_akumulasi', 0)
            
            # 2. DETEKSI DETIK INI (Untuk Kepadatan & Kecepatan)
            # PENTING: Jika YOLO mendeteksi 0, maka kepadatan 0%. Jika ada 1 mobil, angka akan berubah.
            total_sekarang = data.get('total', 0) 
            
            # 3. HITUNG KEPADATAN (Gunakan kapasitas 15 agar lebih sensitif)
            kapasitas = 15
            kepadatan_int = min(100, int((total_sekarang / kapasitas) * 100))
            
            # 4. HITUNG KECEPATAN (Dinamis)
            # 80 km/j saat kosong, turun drastis saat ada kendaraan
            if total_sekarang == 0:
                kecepatan_int = 80
            else:
                kecepatan_int = max(10, 80 - int(kepadatan_int * 0.8))
            
            return {
                "kendaraan_hari_ini": total_akumulasi,
                "kepadatan_tertinggi": kepadatan_int,
                "rata_rata_kecepatan": f"{kecepatan_int} km/j",
                "kamera_aktif": len(cctv_list)
            }
    except Exception as e:
        print(f"Firebase Summary Error: {e}")
    
    return {"kendaraan_hari_ini": 0, "kepadatan_tertinggi": "0%", "rata_rata_kecepatan": "80 km/j", "kamera_aktif": len(cctv_list)}

def get_firebase_logic_history(cctv_id, period):
    """Fungsi tunggal untuk data riwayat grafik (Admin & User)"""
    if not cctv_id: return {"labels": ["00:00"], "data": [0], "kepadatan": [0]}
    
    try:
        ref = firebase_db.reference(f'traffic_stats/{cctv_id}/history')
        # Ambil 10 data terakhir agar grafik terus bergerak
        history_data = ref.order_by_key().limit_to_last(10).get()
        
        labels = []
        totals = []
        
        if history_data:
            for key in sorted(history_data.keys()):
                val = history_data[key]
                full_time = val.get('last_update', '00:00:00')
                time_only = full_time.split(' ')[1] if ' ' in full_time else full_time
                labels.append(time_only)
                totals.append(val.get('total', 0))
                
        return {"labels": labels, "data": totals, "kepadatan": totals}
    except Exception as e:
        print(f"Firebase History Error: {e}")
        return {"labels": ["00:00"], "data": [0], "kepadatan": [0]}

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

@app.route('/api/admin/traffic_data')
@api_login_required
def api_admin_traffic_data():
    return jsonify(get_firebase_logic_history(request.args.get('cctv_id'), request.args.get('period')))

@app.route('/api/public/traffic_data')
def api_public_traffic_data():
    return jsonify(get_firebase_logic_history(request.args.get('cctv_id'), request.args.get('period')))

@app.route('/api/public/analytics_data')
def api_public_analytics_data():
    return jsonify(get_firebase_logic_history(request.args.get('cctv_id'), request.args.get('period')))

@app.route('/api/admin/vehicle_distribution')
@api_login_required
def api_admin_vehicle_distribution():
    cctv_id = request.args.get('cctv_id')
    ref = firebase_db.reference(f'traffic_stats/{cctv_id}/live')
    data = ref.get()
    labels = ['Mobil', 'Motor', 'Bus', 'Truk']
    if data and 'detail' in data:
        d = data['detail']
        values = [d.get('mobil', 0), d.get('motor', 0), d.get('bus', 0), d.get('truk', 0)]
    else:
        values = [0, 0, 0, 0]
    return jsonify({"labels": labels, "data": values})

@app.route('/api/cctv_locations')
def get_cctv_locations():
    return jsonify(cctv_list), 200
    
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
    
if __name__ == '__main__':
    detector_thread = threading.Thread(
        target=auto_detection_loop,
        daemon=True
    )
    detector_thread.start()

    app.run(debug=True)