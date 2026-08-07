# Smart Traffic Vision

Sistem pemantauan lalu lintas **real-time** berbasis **AI (Computer Vision)** yang memanfaatkan kamera CCTV YouTube untuk mendeteksi dan menghitung kendaraan (mobil, motor, bus, truk) secara otomatis menggunakan **YOLO11**.

Proyek ini adalah website utama (Flask) dari sistem Smart Traffic Vision, yang melengkapi aplikasi mobile-nya. Data statistik lalu lintas disimpan dan disinkronkan melalui **Firebase Realtime Database** dan **MySQL**.

## Fitur Utama

- **Deteksi Kendaraan Real-Time (YOLO11)**
  - Mendeteksi dan menghitung mobil, motor, bus, dan truk dari stream CCTV YouTube.
  - Penandaan objek (bounding box) pada video langsung (`/video_feed`).
  - Klasifikasi status lalu lintas: **Lancar**, **Padat**, dan **Macet** berdasarkan kepadatan.

- **Dashboard Analitik Interaktif**
  - Grafik kepadatan lalu lintas per jam (harian) dan periode lain.
  - Distribusi kendaraan (pie chart).
  - Ringkasan KPI real-time dari Firebase (`total`, `kepadatan_persen`, `status`).
  - Agregasi **"Semua Data CCTV"** otomatis dari seluruh kamera.

- **Peta CCTV (Leaflet)**
  - Menampilkan lokasi CCTV pada peta interaktif.
  - Status konektivitas CCTV yang terdeteksi **secara nyata** (bukan acak) dengan mengukur latensi akses thumbnail YouTube.

- **Deteksi Koneksi CCTV Real**
  - Cek online/offline CCTV sungguhan berbasis HTTP latency dengan cache (TTL 6 detik).
  - Konversi latensi menjadi persentase sinyal.
  - Status "Nonaktif" yang diatur admin tetap dihormati (override manual).

- **Manajemen CCTV (Admin)**
  - CRUD data kamera (tambah, edit, hapus).
  - Kontrol status Aktif/Nonaktif.

- **Manajemen Artikel (Admin)**
  - CRUD artikel informasi lalu lintas.
  - Publikasi / batalkan publikasi.
  - Upload gambar (PNG, JPG, JPEG, GIF).

- **Komentar & Analisis Sentimen**
  - Pengguna dapat mengirim komentar/masukan.
  - Klasifikasi sentimen otomatis (Baik/Buruk) berbasis kata kunci dengan penanganan negasi (mis. "tidak membantu").
  - Grafik analitik sentimen per tanggal untuk admin.

- **Autentikasi Admin**
  - Login/logout dengan sesi Flask.
  - Proteksi halaman admin dan API (`login_required` & `api_login_required`).

## Teknologi yang Digunakan

| Bagian | Teknologi |
| ------ | --------- |
| Backend | Python 3, Flask |
| AI / Computer Vision | Ultralytics YOLO11 (`yolo11n.pt`), OpenCV |
| Database | MySQL (admin, artikel, data CCTV) |
| Realtime Database | Firebase Realtime Database (statistik lalu lintas, komentar) |
| Frontend | Bootstrap 5, Chart.js, Leaflet, Lucide Icons, Google Fonts |
| Video Stream | YouTube (yt-dlp) |

## Struktur Proyek

```
SmartTrafficVisionWeb/
├── app.py                     # Aplikasi utama Flask + engine deteksi YOLO
├── cap_from_youtube.py        # Helper mengambil stream YouTube via yt-dlp
├── yolo11n.pt                 # Model AI YOLO11 (tidak di-commit)
├── serviceAccountKey.json     # Kredensial Firebase (tidak di-commit)
├── static/
│   ├── css/                   # Style & icon
│   ├── js/                    # Bootstrap & skrip frontend
│   └── uploads/               # Upload gambar artikel
└── templates/
    ├── index.html             # Landing page
    ├── dashboard.html         # Dashboard publik
    ├── cctv.html              # Halaman CCTV + peta
    ├── static.html            # Data statistik
    ├── read_artikel.html      # Daftar artikel
    ├── artikel_detail.html    # Detail artikel
    ├── aboutme.html           # Tentang proyek
    ├── admin_login.html       # Login admin
    ├── admin_dashboard.html   # Dashboard admin
    ├── crud_artikel.html      # CRUD artikel
    ├── kelola_artikel.html    # Kelola artikel
    └── main.js                # Logika frontend
```

## Prasyarat

- Python 3.8+
- MySQL Server (database `smart_traffic`)
- Akun Firebase (Realtime Database)
- Koneksi internet (untuk akses stream CCTV YouTube)

## Instalasi

```bash
# 1. Clone repository
git clone https://github.com/farizalfth/SmartTrafficVisionWeb.git
cd SmartTrafficVisionWeb

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependensi
pip install flask mysql-connector-python opencv-python numpy ultralytics yt-dlp firebase-admin requests

# 4. Siapkan kredensial Firebase
#    Letakkan serviceAccountKey.json di folder proyek
#    Sesuaikan databaseURL pada app.py

# 5. Siapkan MySQL
#    Buat database: smart_traffic
#    Tabel cctv dibuat otomatis saat aplikasi pertama kali dijalankan

# 6. Unduh model YOLO
#    Letakkan yolo11n.pt di folder proyek

# 7. Jalankan aplikasi
python app.py
```

Akses aplikasi di `http://127.0.0.1:5000`.

> **Catatan:** `yolo11n.pt` dan `serviceAccountKey.json` tidak di-commit ke repository (lihat `.gitignore`). Anda perlu menyiapkannya secara manual sesuai langkah di atas.

## Endpoint Utama

| Route | Keterangan |
| ----- | ---------- |
| `/` | Landing page |
| `/dashboard` | Dashboard lalu lintas publik |
| `/cctv-page` | Halaman CCTV & peta interaktif |
| `/static-page` | Halaman statistik |
| `/read_artikel` | Daftar artikel |
| `/admin` | Dashboard admin (perlu login) |
| `/login` | Login admin |
| `/video_feed?cctv_id=<id>` | Live stream hasil deteksi AI (MJPEG) |
| `/api/analyze_cctv?cctv_id=<id>` | Deteksi satu frame + bounding box |
| `/api/public/dashboard_summary` | Ringkasan KPI publik |
| `/api/public/traffic_data` | Data kepadatan lalu lintas |
| `/api/cctv_locations` | Lokasi & status CCTV untuk peta |
| `/api/submit_comment` | Kirim komentar (analisis sentimen) |
| `/api/admin/comments_analytics` | Statistik sentimen komentar (admin) |

## Status Lalu Lintas

Status ditentukan dari persentase kepadatan terhadap kapasitas maksimal jalan (15 kendaraan per frame):

- `< 40%` → **Lancar**
- `40% – 75%` → **Padat**
- `> 75%` → **Macet**

## Disinkronkan dengan Aplikasi Mobile

Web ini membaca dan menulis pada node Firebase yang sama dengan aplikasi mobile Smart Traffic Vision, sehingga data lalu lintas dan komentar pengguna tetap konsisten di kedua platform.

## Lisensi

Proyek ini dibuat untuk keperluan tugas akhir / penelitian. Silakan digunakan dan dikembangkan lebih lanjut.
