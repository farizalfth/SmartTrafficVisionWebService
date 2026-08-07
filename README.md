# Smart Traffic Vision Web

Sistem pemantauan lalu lintas **real-time** berbasis **AI (Computer Vision)** yang memanfaatkan stream CCTV YouTube untuk mendeteksi dan menghitung kendaraan (mobil, motor, bus, truk) secara otomatis menggunakan **YOLO11**. Hasil deteksi ditulis ke **Firebase Realtime Database** dan ditampilkan oleh frontend **React + Vite**.

## Alur Sistem

```
┌──────────────┐  baca/tulis langsung     ┌───────────────────────────┐
│    Browser   │ ◀──────────────────────▶ │  Firebase Realtime DB     │
│  (React SPA) │                          │  cctv · artikel · admin   │
└──────┬───────┘                          │  traffic_stats · komentar │
       │  MJPEG / API                     └────────────▲──────────────┘
       ▼                                               │ menulis hasil YOLO
┌───────────────────────────────┐                      │
│  AI Server (Python + Flask)   │ ─────────────────────┘
│  ai_server.py · YOLO11        │
└───────────────────────────────┘
```

1. **AI Server** (`ai_server.py`) menjalankan loop deteksi untuk setiap CCTV aktif di Firebase. Tiap ±10 detik ia mengambil frame dari stream YouTube (`cap_from_youtube`), mendeteksi kendaraan dengan **YOLO11**, lalu menulis `total`, `kepadatan_persen`, `detail`, dan `status` ke node `traffic_stats/<cctv_id>/live`.
2. **Frontend** membaca data langsung dari Firebase untuk dashboard, peta, analitik, dan artikel — jadi tetap berfungsi penuh walau AI server sedang mati (data hanya "beku").
3. **Status server** (`/api/server_status`) mengukur **sinyal real** tiap CCTV (latensi ping ke thumbnail YouTube) dan menampilkan stabilitas sistem, bukan angka hardcode.

## Fitur Utama

- **Deteksi Kendaraan Real-Time (YOLO11)**: hitung mobil/motor/bus/truk dari stream CCTV YouTube; bounding box di `/video_feed`; klasifikasi **Lancar / Padat / Macet** (kepadatan < 40% / 40–75% / > 75%).
- **Dashboard Real-Time** (`/dashboard`): status tiap CCTV, jumlah kendaraan, dan grafik dari data `traffic_stats`.
- **Peta CCTV** (`/cctv-page`): peta Leaflet + kartu CCTV, **jam server digital dengan pilihan zona WIB / WITA / WIT** (tersinkron `server_time`), dan indikator sinyal real tiap titik.
- **Analitik Lalu Lintas** (`/static-page`): laporan **Harian (7 hari) / Mingguan / Bulanan**, KPI total kendaraan, grafik batang per kategori, dan tabel detail.
- **Berita & Artikel** (`/read_artikel`, `/artikel/:id`): daftar artikel dengan pencarian + artikel unggulan, halaman baca dengan hero cover, **penghitung views**, tombol **share** (Web Share API, WhatsApp, Facebook, Twitter, salin tautan), progress baca, dan artikel terkait.
- **Manajemen Admin**: login sederhana (cek node `admin`), kelola CCTV dan artikel (CRUD + upload gambar), pantau status AI server, dan tonton stream deteksi AI.
- **Komentar pengguna** tersimpan di Firebase.

## Struktur Proyek

```
SmartTrafficVisionWeb/
├── ai_server.py               # AI Server: Flask + YOLO11, deteksi CCTV → Firebase
├── migrate_to_firebase.py     # Skrip migrasi MySQL (XAMPP) → Firebase RTDB
├── upload_images.py           # (Opsional) Skrip lama upload gambar → Firebase Storage (butuh Blaze)
├── exports/mysql_export.json  # Backup data hasil migrasi
├── serviceAccountKey.json     # Kredensial Firebase (TIDAK di-commit)
├── yolo11n.pt                 # Model AI YOLO11 (TIDAK di-commit)
├── static/uploads/            # Gambar artikel (dilayani AI server di /static)
└── frontend/                  # Aplikasi React (Vite)
    ├── .env.example           # Konfigurasi Firebase + URL AI server
    ├── vercel.json            # SPA rewrites untuk Vercel
    └── src/
        ├── lib/firebase.js    # Helper baca/tulis Firebase + upload gambar
        ├── lib/traffic.js     # Logika agregasi statistik (port dari app.py)
        ├── components/        # Navbar, Footer, Charts, CctvMap, Reveal, dll.
        └── pages/             # Halaman user & admin (lihat tabel rute)
```

## Rute Frontend

| Rute | Halaman | Akses |
| ---- | ------- | ----- |
| `/` | Beranda | Publik |
| `/dashboard` | Dashboard lalu lintas real-time | Publik |
| `/static-page` | Analitik lalu lintas (harian/mingguan/bulanan) | Publik |
| `/cctv-page` | Peta CCTV + jam server | Publik |
| `/read_artikel` | Daftar artikel | Publik |
| `/artikel/:id` | Detail artikel | Publik |
| `/about` | Tentang | Publik |
| `/login` | Login admin | Publik |
| `/admin` | Panel admin (status server, kelola CCTV) | Admin* |
| `/kelola_artikel` | Daftar artikel admin | Admin* |
| `/artikel/tambah` · `/artikel/edit/:id` | Tambah / edit artikel | Admin* |

\* *Rute admin dilindungi `ProtectedRoute` — memerlukan login admin (`localStorage.stv_admin`).*

## Konfigurasi Firebase

1. Buat project Firebase dan aktifkan **Realtime Database**.
2. Unduh `serviceAccountKey.json` (Project Settings → Service accounts) untuk AI server.
3. Untuk frontend, buat **Web App** di Firebase console lalu isi `frontend/.env`:

```env
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=smart-traffic-vision-app
VITE_FIREBASE_DATABASE_URL=https://smart-traffic-vision-app-default-rtdb.asia-southeast1.firebasedatabase.app/
VITE_FIREBASE_MESSAGING_SENDER_ID=...
VITE_FIREBASE_APP_ID=...
# URL tempat ai_server.py dijalankan. Wajib URL publik (tunnel/port-forward/host)
# agar gambar artikel & API bisa diakses pengunjung saat frontend online.
VITE_AI_SERVER_URL=http://localhost:5000
# (Opsional) Prefix gambar artikel lama; jika kosong memakai VITE_AI_SERVER_URL
VITE_IMAGE_BASE_URL=
```

## Struktur Realtime Database

```
cctv/
  c1 ... c5            # {id, name, url, lat, lon, status} — key ber-prefix agar tidak jadi array
artikel/
  a1 ... aN            # {id, judul, tanggal, isi, gambar, published, views}
admin/                 # {username, password} — dipakai login admin
traffic_stats/
  <cctv_id>/
    live/              # {total, kepadatan_persen, detail, status, last_update}
    daily_reports/<YYYY-MM-DD>/   # akumulasi harian
user_comments/         # komentar pengguna
```

> Key Firebase tidak boleh angka murni (RTDB mengubahnya jadi array), karena itu memakai prefiks `c1..c5` dan `a1..aN`.

## Menjalankan AI Server

```bash
python -m venv venv && source venv/bin/activate
pip install flask flask-cors opencv-python numpy ultralytics yt-dlp firebase-admin cap_from_youtube
python ai_server.py        # http://localhost:5000
```

Saat server berjalan, `detection_loop` otomatis mendeteksi kendaraan tiap ±10 detik dan menulis hasilnya ke `traffic_stats/<cctv_id>/live`.

### Endpoint AI Server

| Route | Keterangan |
| ----- | ---------- |
| `/video_feed?cctv_id=<id>` | Live stream hasil deteksi AI (MJPEG) |
| `/api/analyze_cctv?cctv_id=<id>` | Deteksi satu frame + bounding box (base64) |
| `/api/cctv_list` | Daftar CCTV dari Firebase |
| `/api/upload` | Upload gambar artikel ke `static/uploads/` |
| `/api/server_status` | Status server + **sinyal real tiap CCTV** (latensi ke thumbnail YouTube), stabilitas, uptime, `server_time` |

Label stabilitas pada `/api/server_status`: **SANGAT STABIL** (≥90), **STABIL** (≥70), **OPTIMAL** (≥50), **DIPANTAU** (≥30), **KRITIS** (<30).

## Menjalankan Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # build produksi ke dist/
```

## Alur Admin

1. Buka `/login`, isi `username` & `password` (divalidasi ke node `admin` di Firebase).
2. Sukses → `localStorage.stv_admin = "1"`, redirect ke `/admin`.
3. `/admin`: pantau **status AI server** (sinyal, stabilitas, uptime dari `/api/server_status`), pilih CCTV untuk menonton **stream deteksi AI**, dan kelola CCTV (tambah/edit/hapus).
4. `/kelola_artikel`: kelola artikel — **Tambah** (`/artikel/tambah`) atau **Edit** (`/artikel/edit/:id`) dengan upload gambar ke AI server; publikasikan dengan cek **Published**.
5. Logout menghapus `stv_admin` dan kembali ke `/login`.

## Upload Gambar Artikel

Gambar artikel **tidak memakai Firebase Storage** (butuh paket Blaze/kartu kredit). Gambar disimpan di **disk AI server**:

1. Frontend upload via `POST /api/upload` → disimpan ke `static/uploads/`.
2. AI server (Flask) menyajikan folder itu otomatis di `/static/uploads/<nama>`.
3. URL lengkap disimpan di field `gambar` artikel; gambar lama yang hanya berisi nama file di-resolve lewat `VITE_AI_SERVER_URL`.

Karena itu **`VITE_AI_SERVER_URL` harus URL publik** agar pengunjung Vercel bisa memuat gambar dan memanggil API. (Solusi publik gratis: tunnel Cloudflare/ngrok atau port-forward router.)

## Deploy

### Frontend → Vercel

1. Push `frontend/` ke GitHub (atau import folder ke Vercel).
2. Isi **Environment Variables** di dashboard Vercel sesuai `.env.example`.
3. Build command `npm run build`, output dir `dist`. `vercel.json` sudah mengatur SPA rewrites.

### AI Server

Vercel tidak bisa menjalankan Python (khususnya OpenCV/YOLO). Hosting AI server bisa via:
- **Render / Railway / Fly.io** (deploy `ai_server.py` sebagai service web, port 5000), atau
- **Laptop/LAN sendiri** — set `VITE_AI_SERVER_URL` ke IP lokal. Dashboard tetap berfungsi tanpa AI server karena statistik dibaca dari Firebase.

## Migrasi Data (MySQL → Firebase)

```bash
python migrate_to_firebase.py   # export MySQL → exports/mysql_export.json + upload ke Firebase
```

Membaca database MySQL `smart_traffic` (tabel cctv, artikel, admin) dan menulis ke node Firebase dengan key ber-prefix. Sesuaikan kredensial MySQL di bagian atas file.

## Status Lalu Lintas

Status ditentukan dari persentase kepadatan terhadap kapasitas maksimal jalan (15 kendaraan per frame):

- `< 40%` → **Lancar**
- `40% – 75%` → **Padat**
- `> 75%` → **Macet**

## Lisensi

Proyek ini dibuat untuk keperluan tugas akhir / penelitian. Silakan digunakan dan dikembangkan lebih lanjut.
