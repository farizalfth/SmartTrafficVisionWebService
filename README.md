# Smart Traffic Vision Web

Sistem pemantauan lalu lintas **real-time** berbasis **AI (Computer Vision)** yang memanfaatkan stream CCTV YouTube untuk mendeteksi dan menghitung kendaraan (mobil, motor, bus, truk) secara otomatis menggunakan **YOLO11**. Hasil deteksi ditulis ke **Firebase Realtime Database** (CCTV, statistik, komentar) dan data artikel disimpan di **Supabase** (Postgres + Storage), lalu ditampilkan oleh frontend **React + Vite**.

## Alur Sistem

```
┌──────────────┐  artikel & gambar ⇄   ┌───────────────────────────┐
│    Browser   │ ◀───────────────────▶ │  Supabase (Postgres +     │
│  (React SPA) │                       │   Storage "Image Artikel")│
│              │  baca/tulis langsung  ┌───────────────────────────┐
│              │ ◀───────────────────▶ │  Firebase Realtime DB     │
└──────┬───────┘                       │  cctv · traffic_stats ·   │
       │  MJPEG / API                  │  komentar · auth          │
       ▼                               └────────────▲──────────────┘
┌───────────────────────────────┐                  │ menulis hasil YOLO
│  AI Server (Python + Flask)   │ ─────────────────┘
│  ai_server.py · YOLO11        │
└───────────────────────────────┘
```

1. **AI Server** (`ai_server.py`) menjalankan loop deteksi untuk setiap CCTV aktif di Firebase. Tiap ±10 detik ia mengambil burst frame dari stream YouTube (`cap_from_youtube`), mendeteksi kendaraan dengan **YOLO11**, menghitung **occupancy**, **kecepatan rata-rata**, dan **rasio antrean** (tracking centroid antar frame), lalu menulis `total`, `occupancy_persen`, `kepadatan_persen`, `kecepatan_kmh`, `queue`, `detail`, dan `status` ke node `traffic_stats/<cctv_id>/live`. Hasilnya diperhalus dengan **EMA smoothing** dan **voting mayoritas status** agar tidak berkedip.
2. **Frontend** membaca data langsung dari Firebase (CCTV, statistik, komentar) dan Supabase (artikel + gambar) — jadi tetap berfungsi penuh walau AI server sedang mati (data hanya "beku").
3. **Status server** (`/api/server_status`) mengukur **sinyal real** tiap CCTV (latensi ping ke thumbnail YouTube) dan menampilkan stabilitas sistem, bukan angka hardcode.
4. **Kalibrasi per kamera**: konfigurasi `kapasitas`, `px_per_m`, dan `roi` tersimpan per CCTV di Firebase (dikelola dari panel admin) dan dipakai AI server saat menghitung kepadatan, kecepatan, dan antrean.

## Fitur Utama

- **Deteksi Kendaraan Real-Time (YOLO11)**: hitung mobil/motor/bus/truk dari stream CCTV YouTube; bounding box di `/video_feed`; klasifikasi **Lancar / Padat / Macet** berbasis **occupancy** (sinyal utama) dengan kecepatan & antrean sebagai pendukung bila tracking cukup sampel.
- **Dashboard Real-Time** (`/dashboard`): ringkasan KPI, status tiap CCTV (pil traffic-light), jumlah & jenis kendaraan, kepadatan, kecepatan, antrean, live stream deteksi AI, dan grafik dari data `traffic_stats`.
- **Peta CCTV** (`/cctv-page`): peta Leaflet dengan marker berwarna sesuai status (hijau/kuning/merah/abu "Tidak Ada Data"), jam server digital dengan pilihan zona **WIB / WITA / WIT** (tersinkron `server_time`), dan indikator sinyal real tiap titik.
- **Analitik Lalu Lintas** (`/static-page`): laporan **Harian (7 hari) / Mingguan / Bulanan**, KPI total kendaraan, grafik batang per kategori, dan tabel detail.
- **Berita & Artikel** (`/read_artikel`, `/artikel/:id`): daftar artikel dengan pencarian + artikel unggulan, halaman baca dengan hero cover, **penghitung views**, tombol **share** (Web Share API, WhatsApp, Facebook, Twitter, salin tautan), progress baca, dan artikel terkait.
- **Manajemen Admin**: login via **Firebase Authentication** (email/password), kelola CCTV dan artikel (CRUD + upload gambar), **kalibrasi per kamera** (kapasitas, skala piksel/meter, ROI jalan), pantau status AI server, dan tonton stream deteksi AI.
- **Komentar pengguna** tersimpan di Firebase dengan deteksi sentimen otomatis (Baik/Netral/Buruk).
- **UI/UX responsif**: layout dirapikan khusus untuk HP/tablet (kontrol streaming, KPI, kartu ringkasan, peta, dan halaman login).

## Struktur Proyek

```
SmartTrafficVisionWeb/
├── ai_server.py               # AI Server: Flask + YOLO11, deteksi CCTV → Firebase + endpoint kalibrasi
├── cap_from_youtube.py        # Pembuka stream YouTube untuk OpenCV
├── migrate_to_firebase.py     # Skrip migrasi MySQL (XAMPP) → Firebase RTDB
├── upload_images.py           # (Opsional) Skrip lama upload gambar → Firebase Storage (butuh Blaze)
├── exports/mysql_export.json  # Backup data hasil migrasi
├── serviceAccountKey.json     # Kredensial Firebase (TIDAK di-commit)
├── yolo11n.pt                 # Model AI YOLO11 (TIDAK di-commit)
├── static/uploads/            # Gambar artikel (dilayani AI server di /static)
└── frontend/                  # Aplikasi React (Vite)
    ├── .env.example           # Konfigurasi Firebase + Supabase + URL AI server
    ├── supabase.sql           # SQL pembuatan tabel `artikel` di Supabase (jalankan di dashboard)
    ├── scripts/seed_artikel.mjs # Seed artikel ke Supabase (hapus lama + isi artikel baru)
    ├── vercel.json            # SPA rewrites untuk Vercel
    └── src/
        ├── lib/firebase.js    # Helper baca/tulis Firebase, Firebase Auth (login admin), upload Supabase
        ├── lib/supabase.js    # Helper CRUD artikel via Supabase Postgres (REST)
        ├── lib/traffic.js     # Logika agregasi statistik + status efektif (kesegaran data)
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

\* *Rute admin dilindungi `ProtectedRoute` — memerlukan sesi **Firebase Authentication** (login email/password).*

## Teknologi

| Bagian | Teknologi |
| ------ | --------- |
| Frontend | React + Vite, Bootstrap, Chart.js, Leaflet, react-router-dom |
| Backend & API | Flask (Python) — REST API + AI server |
| AI Detection | Computer Vision & YOLO11 (OpenCV, ultralytics) |
| Firebase | Realtime Database — CCTV, traffic stats, komentar; Authentication — login admin |
| Supabase | Postgres (tabel `artikel`) + Storage bucket `Image Artikel` (gambar publik) |
| Streaming Video | Flask MJPEG video feed (`/video_feed`) |
| Deploy | Vercel (frontend), Render/Railway/Fly.io atau server sendiri (AI server) |

## Konfigurasi Firebase

1. Buat project Firebase dan aktifkan **Realtime Database**.
2. Unduh `serviceAccountKey.json` (Project Settings → Service accounts) untuk AI server.
3. Untuk frontend, buat **Web App** di Firebase console lalu isi `frontend/.env`:

```env
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=smart-traffic-vision-app
VITE_FIREBASE_DATABASE_URL=https://smart-traffic-vision-app-default-rtdb.asia-southeast1.firebasedatabase.app/
VITE_FIREBASE_STORAGE_BUCKET=...
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
  c1 ... c5            # {id, name, url, lat, lon, status, kapasitas, px_per_m, roi}
                       #   key ber-prefix agar tidak jadi array;
                       #   kapasitas/px_per_m/roi = kalibrasi per kamera
traffic_stats/
  <cctv_id>/
    live/              # {total, occupancy_persen, kepadatan_persen, kecepatan_kmh,
                       #  queue, detail, status, last_update, last_update_ts}
    daily_reports/<YYYY-MM-DD>/   # akumulasi harian
user_comments/         # komentar pengguna
```

- `status` (Lancar/Padat/Macet) hanya dianggap valid bila data masih **segar** (`last_update`/`last_update_ts` < 90 detik); jika basi, UI menampilkan **"Tidak Ada Data"** (abu-abu) — tidak mengarang status.
- `kecepatan_kmh` hanya ditulis bila ada pengukuran tracking nyata; tanpa pengukuran field ini dihapus dan UI menampilkan "—".

> Login admin memakai **Firebase Authentication** (Email/Password), bukan node `admin` di RTDB.

> Key Firebase tidak boleh angka murni (RTDB mengubahnya jadi array), karena itu memakai prefiks `c1..c5`.
>
> **Artikel sudah tidak lagi di Firebase** — data artikel disimpan di **Supabase Postgres** (tabel `artikel`), lihat [Supabase Artikel](#supabase-artikel-data-crud) di bawah.

## Menjalankan AI Server

```bash
python -m venv venv && source venv/bin/activate
pip install flask flask-cors opencv-python numpy ultralytics yt-dlp firebase-admin
python ai_server.py        # http://localhost:5000
```

> Modul `cap_from_youtube` sudah tersedia sebagai file lokal (`cap_from_youtube.py`); jika ingin versi pip, tambahkan `cap_from_youtube` pada daftar install. Model **`yolo11n.pt`** harus berada di folder yang sama dengan `ai_server.py`.

Saat server berjalan, `detection_loop` otomatis mendeteksi kendaraan tiap ±10 detik dan menulis hasilnya ke `traffic_stats/<cctv_id>/live`.

### Endpoint AI Server

| Route | Keterangan |
| ----- | ---------- |
| `/video_feed?cctv_id=<id>` | Live stream hasil deteksi AI (MJPEG) |
| `/api/analyze_cctv?cctv_id=<id>` | Deteksi satu frame + bounding box (base64) |
| `/api/cctv_list` | Daftar CCTV dari Firebase |
| `/api/cctv_config/<id>` (POST) | Simpan kalibrasi kamera: `{kapasitas?, px_per_m?, roi?}` ke Firebase |
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

1. Buka `/login`, isi **email** & **password** (verifikasi via **Firebase Authentication**).
2. Sukses → sesi Firebase Auth aktif, redirect ke `/admin` (proteksi sesi di `ProtectedRoute`).
3. `/admin`: pantau **status AI server** (sinyal, stabilitas, uptime dari `/api/server_status`), pilih CCTV untuk menonton **stream deteksi AI**, dan kelola CCTV (tambah/edit/hapus).
4. **Kalibrasi kamera**: pada modal CCTV admin atur `kapasitas` (perkiraan jumlah kendaraan saat jalan penuh), `px_per_m` (skala piksel per meter untuk estimasi kecepatan), dan `roi` (batas area jalan ternormalisasi 0–1) → tersimpan ke Firebase via `POST /api/cctv_config/<id>` dan langsung dipakai loop deteksi.
5. `/kelola_artikel`: kelola artikel — **Tambah** (`/artikel/tambah`) atau **Edit** (`/artikel/edit/:id`); data tersimpan di **Supabase** (tabel `artikel`) dan gambar di-upload ke Supabase Storage.
6. Logout memanggil Firebase `signOut` dan kembali ke `/login`.

> **Setup Firebase Auth (sekali saja):** Firebase Console → *Authentication* → *Sign-in method* → aktifkan provider **Email/Password**, lalu tambahkan user admin (email + password) di tab *Users*. Pastikan `VITE_FIREBASE_AUTH_DOMAIN` terisi di `.env` frontend.

## Supabase Artikel (Data CRUD)

Sejak frontend memakai Supabase, **artikel tidak lagi disimpan di Firebase RTDB**. CRUD artikel (tambah/edit/hapus/publish) dan gambar disimpan di Supabase:

- **Data artikel** → tabel Postgres `public.artikel` (id, judul, tanggal, isi, gambar, published, views).
- **Gambar artikel** → Supabase Storage bucket publik `Image Artikel`.

### Setup Supabase Artikel

1. Buat project di https://supabase.com (paket gratis 500 MB).
2. **Storage → New bucket** → nama `Image Artikel` → centang **Public bucket**.
3. **SQL Editor → New query** → jalankan isi `frontend/supabase.sql` (membuat tabel `artikel` + RLS policy akses anon).
4. **Project Settings → API** → salin **Project URL** dan **anon public key** ke `frontend/.env`:
   ```env
   VITE_SUPABASE_URL=https://xxxx.supabase.co
   VITE_SUPABASE_ANON_KEY=eyJhbGciOi...
   VITE_SUPABASE_BUCKET=Image Artikel
   ```
   (Variabel yang sama juga diisi di dashboard Vercel.)

### Seed / hapus artikel via script

```bash
cd frontend
node --env-file=.env scripts/seed_artikel.mjs      # hapus semua artikel lama + isi artikel baru
node --env-file=.env scripts/seed_artikel.mjs --keep-existing   # hanya tambah, jangan hapus lama
```

## Upload Gambar Artikel

Gambar artikel baru disimpan ke **Supabase Storage** (bukan Firebase Storage, bukan disk AI server) supaya punya **URL publik** yang bisa dimuat semua pengunjung situs yang sudah di-deploy (Vercel) di perangkat mana pun:

1. Admin memilih file gambar saat tambah/edit artikel.
2. `uploadArticleImage()` di `lib/firebase.js` meng-upload langsung dari browser ke bucket Supabase.
3. Field `gambar` menyimpan **URL publik** `https://<ref>.supabase.co/storage/v1/object/public/Image%20Artikel/<nama>`.

### Setup Supabase Storage

1. Buat project di https://supabase.com (paket gratis 500 MB).
2. **Storage → New bucket** → nama `Image Artikel` → centang **Public bucket**.
3. **Project Settings → API** → salin **Project URL** dan **anon public key**.
4. Isi env var di `frontend/.env` (dan di Vercel):
   ```env
   VITE_SUPABASE_URL=https://xxxx.supabase.co
   VITE_SUPABASE_ANON_KEY=eyJhbGciOi...
   VITE_SUPABASE_BUCKET=Image Artikel
   ```

### Gambar artikel lama (berupa nama file saja)

Agar tetap tampil, **salin gambar ke `frontend/public/static/uploads/`** lalu `npm run build` — gambar ikut ter-bundle ke `dist` dan dilayani CDN Vercel dari origin situs. `imageUrl()` otomatis memakai `window.location.origin` jika base (`VITE_AI_SERVER_URL` / `VITE_IMAGE_BASE_URL`) berupa `localhost`/IP LAN yang tidak terjangkau publik, dan memakai base aslinya jika sudah URL publik.

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

Status ditentukan berbasis **occupancy** (persentase area jalan yang terisi kendaraan terhadap ROI) sebagai sinyal utama, dengan kecepatan & antrean sebagai pendukung bila tracking menghasilkan sampel cukup (≥3):

- **Occupancy < 30%** → **Lancar** (trafik ringan; kecepatan/antrean tidak dipakai karena tidak andal di trafik sepi).
- **Occupancy 30–55%** → **Padat**, naik ke **Macet** bila kecepatan < 10 km/j atau antrean ≥ 50% dengan kecepatan < 12 km/j.
- **Occupancy ≥ 55%** → **Macet**, turun ke **Padat** hanya bila arus benar-benar melaju (kecepatan ≥ 30 km/j dan antrean < 50%).

Hasil diperhalus **EMA smoothing** + **voting mayoritas** (riwayat 6 siklus) agar tidak berkedip. Data yang **tidak segar** (>90 detik) ditampilkan sebagai **"Tidak Ada Data"** — bukan status mengarang.

## Lisensi

Proyek ini dibuat untuk keperluan tugas akhir / penelitian. Silakan digunakan dan dikembangkan lebih lanjut.
