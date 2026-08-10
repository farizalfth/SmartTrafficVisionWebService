# LAPORAN PROYEK

## SMART TRAFFIC VISION
### Sistem Pemantauan Lalu Lintas Real-Time Berbasis AI untuk Mendukung Kota Berkelanjutan (SDG 11)

---

**Disusun oleh:**
[Nama Lengkap]
[NIS / Kelas]

**Sekolah / Instansi:**
[Nama Sekolah / Instansi]

**[Tahun Ajaran]**

---

<!-- ================== DAFTAR ISI ================== -->

## DAFTAR ISI

**HALAMAN SAMPUL** .................................................................................................. 1

**DAFTAR ISI** .............................................................................................................. 2

**BAB I. PENDAHULUAN** ............................................................................................... 3
&nbsp;&nbsp;&nbsp;&nbsp;A. Latar Belakang ......................................................................................... 3
&nbsp;&nbsp;&nbsp;&nbsp;B. Tujuan ...................................................................................................... 4
&nbsp;&nbsp;&nbsp;&nbsp;C. Manfaat .................................................................................................... 5

**BAB II. PEMBAHASAN** ................................................................................................ 6
&nbsp;&nbsp;&nbsp;&nbsp;A. Penjelasan tentang Website .......................................................................... 6
&nbsp;&nbsp;&nbsp;&nbsp;B. Metode Pengembangan / Perancangan ........................................................ 7
&nbsp;&nbsp;&nbsp;&nbsp;C. Teknologi / Tools yang Digunakan ............................................................... 8
&nbsp;&nbsp;&nbsp;&nbsp;D. Arsitektur Sistem / User Flow ...................................................................... 9
&nbsp;&nbsp;&nbsp;&nbsp;E. Fitur dan Fungsi ........................................................................................ 10
&nbsp;&nbsp;&nbsp;&nbsp;F. Permasalahan dan Solusi ........................................................................... 11
&nbsp;&nbsp;&nbsp;&nbsp;G. Dampak dan Implementasi .......................................................................... 13

**BAB III. PENUTUP** .................................................................................................... 14
&nbsp;&nbsp;&nbsp;&nbsp;A. Kesimpulan ................................................................................................ 14
&nbsp;&nbsp;&nbsp;&nbsp;B. Saran ......................................................................................................... 14

**DAFTAR PUSTAKA** ..................................................................................................... 15

**LAMPIRAN** ................................................................................................................. 16

---

<!-- ================== BAB I ================== -->

## BAB I. PENDAHULUAN

### A. Latar Belakang

Pertumbuhan jumlah kendaraan di perkotaan meningkat pesat setiap tahun, sementara
kapasitas jalan dan fasilitas lalu lintas tidak selalu bertambah seimbang. Akibatnya,
kemacetan, kepadatan lalu lintas, dan risiko kecelakaan menjadi masalah yang hampir
dirasakan setiap hari oleh masyarakat di kota-kota besar maupun kota berkembang.
Kemacetan tidak hanya membuang waktu dan bahan bakar, tetapi juga meningkatkan emisi
gas buang yang memperburuk kualitas udara dan mempercepat perubahan iklim.

Berdasarkan **Sustainable Development Goals (SDGs)**, khususnya **SDG 11 — Kota dan
Permukiman yang Berkelanjutan**, pemerintah dan masyarakat dituntut untuk menciptakan
kota yang aman, inklusif, tangguh, dan berkelanjutan. Salah satu target dalam SDG 11
adalah menyediakan **sistem transportasi yang aman, terjangkau, dan berkelanjutan**
(target 11.2), serta meningkatkan penataan kota yang partisipatif dan terintegrasi
(target 11.3). Untuk mewujudkan hal tersebut, dibutuhkan data lalu lintas yang akurat
dan terkini sebagai dasar pengambilan keputusan.

Saat ini, pemantauan lalu lintas secara manual masih banyak dilakukan, misalnya dengan
petugas di lapangan atau hanya mengandalkan kamera CCTV tanpa analisis otomatis. Cara
ini membutuhkan banyak tenaga, mudah terlewat, dan tidak dapat memberikan data real-time
secara terus-menerus. Padahal, teknologi **Computer Vision** dan **Kecerdasan Buatan (AI)**
telah berkembang pesat dan dapat digunakan untuk mendeteksi serta menghitung kendaraan
secara otomatis dari gambar atau video kamera pengawas.

Berdasarkan permasalahan tersebut, proyek ini membangun **Smart Traffic Vision**, sebuah
website pemantauan lalu lintas real-time berbasis AI. Sistem ini menggunakan model
**YOLO11** untuk mendeteksi kendaraan (mobil, motor, bus, dan truk) dari stream kamera
CCTV, kemudian menghitung kepadatan, kecepatan rata-rata, dan potensi antrean di setiap
titik pantau. Hasilnya ditampilkan dalam bentuk dashboard, peta lokasi, dan grafik yang
mudah dipahami oleh masyarakat umum maupun pengelola jalan.

Dengan demikian, Smart Traffic Vision menjadi salah satu bentuk penerapan teknologi
untuk mendukung **kota yang aman dan berkelanjutan (SDG 11)** melalui penyediaan
informasi lalu lintas yang cepat, akurat, dan dapat diakses oleh siapa saja.

### B. Tujuan

Tujuan dari pembuatan proyek Smart Traffic Vision adalah sebagai berikut:

1. Merancang dan membangun website pemantauan lalu lintas yang mampu menampilkan
   kondisi lalu lintas **secara real-time** dari beberapa titik kamera CCTV.
2. Menggunakan teknologi **Computer Vision (YOLO11)** untuk mendeteksi dan menghitung
   kendaraan secara otomatis beserta metrik lalu lintas seperti kepadatan
   (occupancy), kecepatan rata-rata, dan rasio antrean.
3. Menyediakan **dashboard informasi** yang mudah dipahami masyarakat tentang kondisi
   lalu lintas di suatu daerah.
4. Memberikan **panel admin** untuk mengelola data kamera CCTV, kalibrasi area deteksi,
   mengelola artikel, dan memantau status server AI.
5. Menerapkan prinsip **SDG 11** dengan menghadirkan solusi teknologi yang mendukung
   kota aman, nyaman, dan berkelanjutan melalui pengelolaan lalu lintas yang lebih
   baik.

### C. Manfaat

Manfaat yang dapat diperoleh dari proyek ini antara lain:

1. **Bagi Masyarakat**
   - Mendapatkan informasi kondisi lalu lintas secara real-time sehingga dapat
     merencanakan perjalanan dengan lebih baik dan menghindari kemacetan.
   - Meningkatkan kesadaran akan pentingnya ketertiban berlalu lintas.

2. **Bagi Pemerintah / Pengelola Jalan**
   - Mendapatkan data lalu lintas yang objektif dan terus-menerus sebagai bahan
     pengambilan keputusan, misalnya penempatan lampu lalu lintas, rekayasa jalan,
     dan penanganan titik rawan macet.
   - Memantau beberapa titik CCTV dari satu sistem terpusat.

3. **Bagi Pengembang Teknologi / Pelajar**
   - Menjadi media pembelajaran penerapan AI, Computer Vision, pengolahan data
     real-time, serta pengembangan website modern.

4. **Bagi Lingkungan**
   - Mendukung pengurangan emisi kendaraan dengan mengurangi waktu berhenti di
     kemacetan, sehingga sejalan dengan SDG 11 dan upaya penanganan perubahan iklim.

---

<!-- ================== BAB II ================== -->

## BAB II. PEMBAHASAN

### A. Penjelasan tentang Website

**Smart Traffic Vision** adalah website pemantauan lalu lintas real-time yang memanfaatkan
kecerdasan buatan untuk mendeteksi kendaraan dari kamera CCTV (termasuk stream CCTV
YouTube) dan menampilkan hasilnya dalam bentuk dashboard yang informatif.

Website ini memiliki dua sisi utama:

1. **Sisi Publik**
   Berisi halaman beranda, dashboard statistik lalu lintas, peta lokasi CCTV,
   halaman CCTV, dan halaman artikel/berita. Masyarakat dapat melihat jumlah
   kendaraan, tingkat kepadatan, kecepatan rata-rata, serta status lalu lintas
   (Lancar, Padat, atau Macet) di setiap titik pantau.

2. **Sisi Admin**
   Berisi halaman login (Firebase Authentication), dashboard admin untuk
   mengelola data kamera CCTV (tambah, edit, hapus), kalibrasi area deteksi
   (kapasitas, skala piksel/meter, ROI), melihat stream deteksi AI secara
   langsung, memantau status server AI (stabilitas dan sinyal per CCTV),
   mengelola artikel, serta mengelola ulasan/komentar masyarakat.

Data lalu lintas dihasilkan oleh **AI Server** yang menjalankan model YOLO11 terhadap
stream video CCTV, kemudian hasilnya ditulis ke **Firebase Realtime Database**. Sisi
frontend membaca data dari Firebase sehingga informasi selalu diperbarui secara
otomatis setiap beberapa detik.

### B. Metode Pengembangan / Perancangan

Metode yang digunakan dalam pengembangan proyek ini adalah metode **Waterfall**
dengan pendekatan **iteratif pada tahap perbaikan tampilan**. Tahapan yang dilakukan:

1. **Analisis Kebutuhan**
   Mengidentifikasi masalah lalu lintas, kebutuhan informasi publik, kebutuhan
   pengelola jalan, serta fitur yang diperlukan (pemantauan CCTV, deteksi otomatis,
   dashboard, artikel, dan manajemen data).

2. **Perancangan (Desain)**
   - Merancang arsitektur sistem (AI server, Firebase, frontend React).
   - Merancang struktur data pada Firebase Realtime Database.
   - Merancang antarmuka (UI/UX) untuk halaman publik dan admin, termasuk
     tampilan responsif untuk perangkat HP.

3. **Implementasi (Pembuatan Kode)**
   - Membangun AI server dengan Python (Flask + OpenCV + Ultralytics YOLO11).
   - Membangun frontend dengan React + Vite dan library pendukung (Chart.js,
     Leaflet, Bootstrap).
   - Menghubungkan seluruh komponen dengan Firebase dan Supabase.

4. **Pengujian**
   - Menguji deteksi kendaraan dari video CCTV.
   - Menguji alur data dari AI server ke Firebase hingga tampilan dashboard.
   - Menguji tampilan pada berbagai ukuran layar (desktop dan HP).

5. **Perbaikan (Iteratif)**
   - Menyempurnakan tampilan mobile agar rapi dan tidak "zoom".
   - Memperbaiki koneksi AI server pada versi yang sudah di-deploy.
   - Menyempurnakan tampilan sinyal per CCTV.

### C. Teknologi / Tools yang Digunakan

| No | Komponen | Teknologi / Tools | Fungsi |
|----|----------|-------------------|--------|
| 1 | Frontend | React + Vite | Membangun antarmuka pengguna |
| 2 | Frontend | Bootstrap + Bootstrap Icons | Menata tampilan dan ikon |
| 3 | Frontend | Chart.js (react-chartjs-2) | Menampilkan grafik lalu lintas |
| 4 | Frontend | Leaflet | Peta lokasi CCTV |
| 5 | Backend / AI | Python + Flask | Server API dan AI server |
| 6 | AI | Ultralytics YOLO11 | Deteksi kendaraan (Computer Vision) |
| 7 | AI | OpenCV (cv2) | Pengolahan video dan gambar |
| 8 | AI | yt-dlp / cap_from_youtube | Mengambil stream video YouTube |
| 9 | Database | Firebase Realtime Database | Data CCTV, lalu lintas, komentar |
| 10 | Database | Firebase Authentication | Login admin |
| 11 | Database | Supabase (Postgres + Storage) | Data dan gambar artikel |
| 12 | Deploy | Vercel | Hosting frontend |

### D. Arsitektur Sistem / User Flow

Arsitektur sistem Smart Traffic Vision dapat digambarkan sebagai berikut:

```
                    ┌──────────────────────────────┐
                    │   CCTV (stream YouTube)       │
                    └──────────────┬───────────────┘
                                   │ video stream
                                   ▼
                    ┌──────────────────────────────┐
                    │     AI SERVER (Flask)         │
                    │  • YOLO11 deteksi kendaraan   │
                    │  • hitung kepadatan/kecepatan │
                    └──────────────┬───────────────┘
                                   │ tulis data (total, kepadatan,
                                   │ kecepatan, status, sinyal)
                                   ▼
                    ┌──────────────────────────────┐
                    │   Firebase Realtime Database │
                    └──────────────┬───────────────┘
                                   │ baca data real-time
                                   ▼
                    ┌──────────────────────────────┐
                    │   FRONTEND (React + Vite)     │
                    │  Dashboard, Peta, Artikel,    │
                    │  Panel Admin                  │
                    └──────────────────────────────┘
```

**Alur Pengguna (User Flow):**

1. **Sisi Publik**
   - Pengguna membuka halaman beranda atau dashboard.
   - Sistem membaca data lalu lintas dari Firebase secara otomatis (setiap 5 detik).
   - Pengguna dapat melihat peta CCTV, memilih titik kamera, dan melihat status
     lalu lintas (Lancar/Padat/Macet) serta jumlah kendaraan.
   - Pengguna dapat membaca artikel dan memberikan komentar/ulasan.

2. **Sisi Admin**
   - Admin login menggunakan email dan kata sandi (Firebase Authentication).
   - Admin memilih CCTV untuk melihat **stream deteksi AI** dengan menekan tombol
     "Deteksi Kendaraan".
   - AI server membuka stream CCTV, mendeteksi kendaraan, dan menampilkan hasilnya
     secara langsung (MJPEG).
   - Data deteksi otomatis ditulis ke Firebase untuk keperluan dashboard dan grafik.
   - Admin dapat menambah/mengedit/menghapus CCTV, melakukan kalibrasi
     (kapasitas, px/meter, ROI), serta memantau status dan sinyal server AI.
   - Admin mengelola artikel dan ulasan masyarakat.

### E. Fitur dan Fungsi

1. **Deteksi Kendaraan Otomatis (YOLO11)**
   Mendeteksi mobil, motor, bus, dan truk dari stream CCTV secara real-time.

2. **Dashboard Real-Time**
   Menampilkan total kendaraan, kepadatan (%), kecepatan rata-rata (km/j), status
   lalu lintas, grafik tren harian/mingguan/bulanan, dan distribusi jenis kendaraan.

3. **Peta Lokasi CCTV (Leaflet)**
   Menampilkan titik-titik CCTV pada peta dengan warna sesuai kondisi lalu lintas.

4. **Halaman CCTV**
   Daftar titik CCTV beserta metrik jumlah kendaraan, kepadatan, dan kecepatan.

5. **Panel Admin**
   - Login admin dengan Firebase Authentication.
   - CRUD data CCTV.
   - Kalibrasi per kamera (kapasitas, skala piksel/meter, ROI).
   - Pemantauan status AI server: stabilitas, uptime, dan sinyal per CCTV.
   - Stream deteksi AI langsung di dashboard admin.
   - Kelola artikel (CRUD + upload gambar via Supabase).
   - Kelola ulasan masyarakat dengan deteksi sentimen otomatis (Baik/Netral/Buruk).

6. **Artikel dan Komentar**
   Halaman berita lalu lintas serta kolom ulasan dengan klasifikasi sentimen.

7. **Tampilan Responsif**
   Website dapat diakses dengan nyaman melalui HP, tablet, maupun desktop.

### F. Permasalahan dan Solusi

| No | Permasalahan | Solusi yang Diterapkan |
|----|--------------|------------------------|
| 1 | AI server tidak terhubung pada versi yang sudah di-deploy (tampil "OFFLINE • TIDAK TERHUBUNG") karena alamat server masih `localhost` yang tidak dapat dijangkau dari perangkat lain. | Membuat pengaturan **URL Server AI** yang dapat diisi langsung dari panel admin dan tersimpan di browser, dengan pencarian otomatis: URL manual → URL dari build → otomatis dari hostname halaman. |
| 2 | Sinyal seluruh CCTV muncul sekaligus padahal baru satu CCTV yang dideteksi. | Sinyal hanya ditampilkan untuk CCTV yang **sedang di-streaming deteksinya**, sehingga muncul satu per satu mengikuti kamera yang dipilih pengguna. |
| 3 | Tampilan admin di HP terlihat "zoom" dan tidak rapi. | Memperbaiki CSS responsif: input berukuran minimal 16px agar tidak auto-zoom di iOS, memadatkan kartu dan jarak antar seksi, serta mencegah overflow horizontal. |
| 4 | Model AI (yolo11n.pt) tidak tersedia atau belum dimuat sehingga stream deteksi gagal. | Menyediakan file model `yolo11n.pt` dan menampilkan pesan yang jelas saat server AI tidak berjalan. |
| 5 | Stream CCTV YouTube kadang gagal diambil (masalah teknis dari platform). | AI server diberi penanganan kesalahan (try/except) agar satu kamera yang gagal tidak mengganggu kamera lainnya. |
| 6 | Menampilkan data yang stabil dan tidak berkedip. | Menggunakan **EMA smoothing** dan **voting mayoritas status** sehingga status lalu lintas lebih stabil. |
| 7 | Grafik sumbu X yang berdesakan di layar sempit. | Merapikan label grafik, menyesuaikan ukuran font, dan membuat area grafik dapat di-scroll horizontal di HP. |

### G. Dampak dan Implementasi

**Dampak terhadap masyarakat dan lingkungan:**
1. Masyarakat mendapatkan informasi lalu lintas real-time sehingga dapat menghindari
   kemacetan, menghemat waktu, dan mengurangi konsumsi bahan bakar.
2. Pengurangan waktu berhenti di kemacetan berdampak pada **penurunan emisi gas
   buang** dan mendukung kualitas udara yang lebih baik.
3. Pengelola jalan dapat mengambil keputusan berbasis data, misalnya penempatan
   lampu lalu lintas, pengaturan jalur, dan penanganan titik rawan macet.

**Implementasi sesuai SDG 11:**
- **Target 11.2**: menyediakan akses sistem transportasi yang aman, terjangkau, dan
  berkelanjutan — melalui pemantauan dan informasi lalu lintas real-time.
- **Target 11.3**: meningkatkan penataan kota yang partisipatif dan terintegrasi —
  dengan menghadirkan data terbuka bagi masyarakat dan pengelola.

**Rencana pengembangan ke depan:**
- Menambahkan deteksi pelanggaran lalu lintas (misalnya melawan arus).
- Memberikan notifikasi otomatis (peringatan kemacetan) kepada masyarakat.
- Memperluas cakupan titik CCTV ke lebih banyak lokasi.
- Menerapkan integrasi dengan sistem manajemen lampu lalu lintas cerdas.

---

<!-- ================== BAB III ================== -->

## BAB III. PENUTUP

### A. Kesimpulan

Smart Traffic Vision berhasil dibangun sebagai website pemantauan lalu lintas
real-time berbasis AI (YOLO11). Sistem ini mampu mendeteksi dan menghitung kendaraan
dari stream CCTV, menampilkan kondisi lalu lintas (kepadatan, kecepatan, antrean,
status) secara otomatis, serta menyediakan dashboard, peta, dan panel admin yang
lengkap. Proyek ini merupakan bentuk penerapan teknologi untuk mendukung **SDG 11 —
Kota dan Permukiman yang Berkelanjutan**, karena membantu pengelolaan lalu lintas
menjadi lebih aman, efisien, dan ramah lingkungan.

### B. Saran

1. Data deteksi sebaiknya divalidasi dengan pengamatan lapangan agar akurasi semakin
   tinggi.
2. Perlu penambahan fitur peringatan dini kemacetan dan pelaporan pelanggaran.
3. Kalibrasi kamera (kapasitas, skala, ROI) perlu dilakukan dengan baik di setiap
   titik agar metrik lalu lintas lebih akurat.
4. Cakupan kamera dapat diperluas ke lebih banyak lokasi untuk pemantauan kota yang
   lebih menyeluruh.

---

<!-- ================== DAFTAR PUSTAKA ================== -->

## DAFTAR PUSTAKA

1. Redmon, J., & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767.

2. Ultralytics. (2024). *YOLO11 Documentation*. https://docs.ultralytics.com/models/yolo11/

3. United Nations. (2015). *Transforming our World: The 2030 Agenda for Sustainable Development*.

4. United Nations. (n.d.). *Goal 11: Make cities inclusive, safe, resilient and sustainable*. https://sdgs.un.org/goals/goal11

5. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.

6. React. (n.d.). *React – A JavaScript library for building user interfaces*. https://react.dev/

7. Vite. (n.d.). *Vite – Next Generation Frontend Tooling*. https://vitejs.dev/

8. Firebase. (n.d.). *Firebase Realtime Database & Authentication Documentation*. https://firebase.google.com/docs

9. Chart.js. (n.d.). *Chart.js Documentation*. https://www.chartjs.org/

10. Supabase. (n.d.). *Supabase Documentation*. https://supabase.com/docs

11. Leaflet. (n.d.). *Leaflet – An open-source JavaScript library for mobile-friendly interactive maps*. https://leafletjs.com/

12. Flask. (n.d.). *Flask Documentation*. https://flask.palletsprojects.com/

---

<!-- ================== LAMPIRAN ================== -->

## LAMPIRAN

**Lampiran 1 — Struktur Proyek**

```
SmartTrafficVisionWeb/
├── ai_server.py                 # AI Server (Flask + YOLO11)
├── app.py                       # Backend Flask (template lama)
├── cap_from_youtube.py          # Pustaka pengambilan stream YouTube
├── yolo11n.pt                   # Model deteksi YOLO11
├── serviceAccountKey.json       # Kredensial Firebase (service account)
├── templates/                   # Template HTML (versi lama)
├── static/                      # File statis
└── frontend/
    ├── index.html
    ├── vercel.json
    ├── .env                     # Konfigurasi Firebase, Supabase, AI URL
    └── src/
        ├── App.jsx              # Konfigurasi rute website
        ├── index.css            # Gaya / CSS
        ├── components/          # Komponen (Navbar, Charts, CctvMap, dll.)
        ├── lib/                 # Pustaka (firebase, supabase, traffic, aiUrl)
        └── pages/               # Halaman (Home, Dashboard, AdminDashboard, dll.)
```

**Lampiran 2 — Alamat URL / Endpoint Utama**

| Endpoint | Fungsi |
|----------|--------|
| `/` | Halaman beranda |
| `/dashboard` | Dashboard statistik lalu lintas |
| `/cctv-page` | Peta dan daftar titik CCTV |
| `/admin` | Dashboard admin |
| `/login` | Halaman login admin |
| `/video_feed?cctv_id=<id>` | Stream deteksi AI (MJPEG) |
| `/api/server_status` | Status AI server dan sinyal CCTV |
| `/api/cctv_list` | Daftar CCTV |
| `/api/analyze_cctv` | Deteksi satu frame + bounding box |

**Lampiran 3 — Kode Inti Deteksi Kendaraan (AI Server)**

```python
from ultralytics import YOLO

# Muat model
model = YOLO("yolo11n.pt")

# Kelas kendaraan: 2=mobil, 3=motor, 5=bus, 7=truk
VEHICLE_CLASSES = [2, 3, 5, 7]

# Deteksi pada satu frame
results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
counts, total, occ, boxes = analyze_frame(frame, results, roi)
```

**Lampiran 4 — Konfigurasi Lingkungan (frontend/.env)**

```
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_DATABASE_URL=...
VITE_FIREBASE_PROJECT_ID=...
VITE_AI_SERVER_URL=http://localhost:5000
VITE_SUPABASE_URL=...
VITE_SUPABASE_ANON_KEY=...
```

**Lampiran 5 — Dokumentasi Foto / Tangkapan Layar**

*(Tempelkan tangkapan layar halaman beranda, dashboard, peta CCTV, panel admin,
stream deteksi AI, dan status server AI di bagian ini.)*
