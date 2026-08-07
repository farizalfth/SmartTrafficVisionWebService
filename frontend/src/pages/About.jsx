import Reveal from "../components/Reveal";

export default function About() {
  return (
    <div className="container text-center" style={{ paddingBottom: 50 }}>
      <Reveal>
        <div style={{ padding: "50px 0 20px" }}>
          <h1 className="display-4 gradient-text fw-bold">Tentang Kami</h1>
          <p className="text-muted mx-auto" style={{ maxWidth: 700, lineHeight: 1.7 }}>
            Sistem pemantauan lalu lintas berbasis Artificial Intelligence yang memanfaatkan kamera CCTV
            YouTube untuk mendeteksi dan menghitung kendaraan secara otomatis menggunakan model YOLO11.
          </p>
        </div>
      </Reveal>

      <div className="row g-4 mb-5 mt-2">
        {[
          { icon: "bi-camera-video", chip: "#3B82F6", t: "Real-time Monitoring", d: "Pemantauan langsung tanpa jeda" },
          { icon: "bi-qr-code-scan", chip: "#B388FF", t: "AI Detection", d: "Deteksi otomatis dengan YOLO11" },
          { icon: "bi-bar-chart", chip: "#00E676", t: "Data Analitik", d: "Grafik tren & distribusi" },
          { icon: "bi-geo-alt", chip: "#FFD600", t: "Peta CCTV", d: "Lokasi kamera interaktif" },
        ].map((f) => (
          <div className="col-lg-3 col-md-6" key={f.t}>
            <Reveal>
              <div className="kpi-card" style={{ height: "100%" }}>
                <div className="icon-chip" style={{ "--chip": f.chip, margin: "0 auto 15px" }}>
                  <i className={`bi ${f.icon}`} style={{ fontSize: 28 }}></i>
                </div>
                <h5 className="fw-bold" style={{ color: "#fff" }}>{f.t}</h5>
                <p className="text-muted mb-0" style={{ fontSize: "0.88rem" }}>{f.d}</p>
              </div>
            </Reveal>
          </div>
        ))}
      </div>

      <div className="row g-4 mb-5">
        <div className="col-md-6">
          <Reveal variant="reveal-left" className="h-100">
            <div className="dashboard-card h-100 d-flex flex-column align-items-center text-center">
              <div className="icon-chip" style={{ "--chip": "#3B82F6", margin: "0 auto 16px" }}>
                <i className="bi bi-eye" style={{ fontSize: 26 }}></i>
              </div>
              <h4 className="fw-bold mb-3" style={{ color: "#fff" }}>Visi Kami</h4>
              <p className="text-muted mb-0" style={{ lineHeight: 1.8, maxWidth: 420 }}>
                Menjadi sistem pemantauan lalu lintas cerdas yang membantu terciptanya kota yang lebih
                teratur, aman, dan nyaman melalui pemanfaatan teknologi Artificial Intelligence.
              </p>
            </div>
          </Reveal>
        </div>
        <div className="col-md-6">
          <Reveal variant="reveal-right" className="h-100">
            <div className="dashboard-card h-100 d-flex flex-column align-items-center text-center">
              <div className="icon-chip" style={{ "--chip": "#00E676", margin: "0 auto 16px" }}>
                <i className="bi bi-bullseye" style={{ fontSize: 26 }}></i>
              </div>
              <h4 className="fw-bold mb-3" style={{ color: "#fff" }}>Misi Kami</h4>
              <ul className="text-muted mb-0" style={{ lineHeight: 2.1, paddingLeft: 0, listStyle: "none", maxWidth: 420 }}>
                <li><i className="bi bi-check-circle me-2" style={{ color: "#00E676" }}></i>Menyediakan data lalu lintas real-time yang akurat.</li>
                <li><i className="bi bi-check-circle me-2" style={{ color: "#00E676" }}></i>Membantu masyarakat memilih rute yang efisien.</li>
                <li><i className="bi bi-check-circle me-2" style={{ color: "#00E676" }}></i>Memberikan insight bagi pengambil kebijakan.</li>
              </ul>
            </div>
          </Reveal>
        </div>
      </div>

      <h3 className="gradient-text fw-bold mb-4">Teknologi Kami</h3>
      <div className="row g-4 mb-5 justify-content-center">
        {[
          { icon: "bi-code-slash", t: "Backend & API", d: "Flask (Python) RESTful API" },
          { icon: "bi-qr-code-scan", t: "AI Detection", d: "Computer Vision & YOLO11" },
          { icon: "bi-lightning", t: "Realtime Database", d: "Firebase Realtime Database" },
          { icon: "bi-database", t: "Database", d: "Data disinkronkan ke Firebase" },
          { icon: "bi-window", t: "Frontend Web", d: "React, Chart.js & Leaflet" },
          { icon: "bi-camera-video", t: "Streaming Video", d: "Flask Video Feed CCTV" },
        ].map((t) => (
          <div className="col-lg-4 col-md-6" key={t.t}>
            <div className="kpi-card" style={{ height: "100%" }}>
              <div className="icon-chip" style={{ "--chip": "#3B82F6", margin: "0 auto 15px", width: 52, height: 52 }}>
                <i className={`bi ${t.icon}`} style={{ fontSize: 24 }}></i>
              </div>
              <h6 className="fw-bold" style={{ color: "#fff" }}>{t.t}</h6>
              <p className="text-muted mb-0" style={{ fontSize: "0.85rem" }}>{t.d}</p>
            </div>
          </div>
        ))}
      </div>

      <h3 className="gradient-text fw-bold mb-4">Tim Pengembang</h3>
      <div className="row g-4 justify-content-center">
        {[
          { initials: "MF", name: "M. Fariz Alfattah" },
          { initials: "NN", name: "Nadhif Nur Fathin" },
          { initials: "BC", name: "Bening Cahya Aura" },
          { initials: "FN", name: "Fajrina Nurhaliza" },
        ].map((m) => (
          <div className="col-lg-3 col-sm-6 col-6" key={m.initials}>
            <div className="kpi-card" style={{ height: "100%" }}>
              <div
                style={{
                  width: 72,
                  height: 72,
                  borderRadius: "50%",
                  margin: "0 auto 15px",
                  background: "linear-gradient(135deg, rgba(59,130,246,0.25), transparent)",
                  border: "2px solid rgba(59,130,246,0.4)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 700,
                  fontSize: 24,
                  color: "#3B82F6",
                }}
              >
                {m.initials}
              </div>
              <h6 className="fw-bold" style={{ color: "#fff" }}>{m.name}</h6>
            </div>
          </div>
        ))}
      </div>
      <p className="text-muted mt-5 pt-3 pb-2">Kelompok 3 - Kelas 5B Teknik Informatika</p>
    </div>
  );
}
