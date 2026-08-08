import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/Reveal";
import CctvMap from "../components/CctvMap";
import CountUp from "../components/CountUp";
import { listenCctv, listenTrafficStats } from "../lib/firebase";
import { statusColor, effectiveStatus, isLiveFresh } from "../lib/traffic";

const FEATURES = [
  {
    icon: "bi-camera-video",
    chip: "#3B82F6",
    title: "Real-time Monitoring",
    desc: "Akses tayangan langsung dari berbagai titik CCTV di kota Anda tanpa jeda waktu, memastikan pemantauan kondisi jalan yang akurat setiap saat.",
  },
  {
    icon: "bi-cpu",
    chip: "#B388FF",
    title: "AI Detection",
    desc: "Sistem cerdas mendeteksi dan menghitung jenis kendaraan (mobil, motor, bus, truk) secara otomatis untuk mengukur tingkat kepadatan lalu lintas.",
  },
  {
    icon: "bi-bar-chart",
    chip: "#00E676",
    title: "Data Analitik",
    desc: "Sajikan data statistik visual berupa grafik tren kepadatan dan distribusi kendaraan untuk membantu pengambilan keputusan yang lebih baik.",
  },
];

const STATUS_LEGEND = [
  { s: "Lancar", c: "#00C853" },
  { s: "Padat", c: "#FFD600" },
  { s: "Macet", c: "#FF5252" },
];

export default function Home() {
  const [cctv, setCctv] = useState([]);
  const [liveMap, setLiveMap] = useState({});
  const [focusId, setFocusId] = useState(null);
  const [focusTick, setFocusTick] = useState(0);
  const [activeCam, setActiveCam] = useState(null);

  useEffect(() => {
    const off = listenCctv(setCctv);
    return off;
  }, []);

  useEffect(() => {
    const off = listenTrafficStats((stats) => {
      const m = {};
      Object.entries(stats || {}).forEach(([k, node]) => {
        if (node && node.live && typeof node.live === "object") m[k] = node.live;
      });
      setLiveMap(m);
    });
    return off;
  }, []);

  const cameras = cctv.filter((c) => c.lat && c.lon);
  const cameraLive = cameras.map((c) => liveMap[c.id] || null);
  const onlineCount = cameraLive.filter((l) => effectiveStatus(l) !== "Tidak Ada Data").length;
  const freshDens = cameraLive
    .filter((l) => isLiveFresh(l) && (l.occupancy_persen ?? l.kepadatan_persen) != null)
    .map((l) => l.occupancy_persen ?? l.kepadatan_persen);
  const totalKendaraan = cameraLive.reduce((a, l) => a + (l?.total || 0), 0);
  const avgKepadatan = freshDens.length
    ? Math.round(freshDens.reduce((a, b) => a + b, 0) / freshDens.length)
    : 0;

  const focusCamera = (id) => {
    setFocusId(id);
    setFocusTick((t) => t + 1);
    setActiveCam(id);
  };

  return (
    <>
      {/* HERO */}
      <section
        className="hero-section"
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          backgroundImage: "linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.85)), url('/hero.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          position: "relative",
        }}
      >
        <HeroParticles />
        <div className="hero-content" style={{ maxWidth: 820, padding: "0 20px", position: "relative", zIndex: 1 }}>
          <span className="hero-badge">
            <span className="pulse-dot"></span>
            AI-Powered · Real-time Monitoring
          </span>
          <h1 className="hero-title gradient-text" style={{ fontSize: "3.5rem", fontWeight: 800, lineHeight: 1.2, marginTop: 24 }}>
            Smart Traffic Vision
          </h1>
          <p
            className="hero-subtitle"
            style={{ fontSize: "1.2rem", color: "#bfdbfe", marginBottom: 40, lineHeight: 1.6 }}
          >
            Solusi cerdas pemantauan lalu lintas berbasis Artificial Intelligence. Dapatkan data kepadatan,
            kecepatan, dan pantauan visual secara real-time untuk kota yang lebih baik.
          </p>
          <div className="d-flex justify-content-center gap-3 flex-wrap hero-cta">
            <Link to="/dashboard" className="btn-cta">
              <i className="bi bi-speedometer2 me-2"></i>Lihat Dashboard
            </Link>
            <Link to="/about" className="btn-ghost">
              <i className="bi bi-info-circle me-2"></i>Tentang Kami
            </Link>
          </div>
        </div>
        <div className="scroll-indicator">
          <span className="scroll-mouse"><span className="scroll-wheel"></span></span>
          <small>Gulir ke bawah</small>
        </div>
      </section>

      {/* FEATURES */}
      <section className="features-section" style={{ padding: "80px 0", backgroundColor: "#080808" }}>
        <div className="container">
          <div className="row g-4">
            {FEATURES.map((f, i) => (
              <div className="col-md-4" key={f.title}>
                <Reveal delay={i * 140}>
                  <div
                    className="feature-card"
                    style={{
                      backgroundColor: "#161616",
                      padding: 30,
                      borderRadius: 16,
                      textAlign: "center",
                      height: "100%",
                      border: "1px solid #2a2a2a",
                      position: "relative",
                      overflow: "hidden",
                    }}
                  >
                    <div className="feature-icon" style={{ "--chip": f.chip }}>
                      <i className={`bi ${f.icon}`} style={{ fontSize: 32 }}></i>
                    </div>
                    <h4 className="fw-bold" style={{ color: "#fff" }}>
                      {f.title}
                    </h4>
                    <p style={{ color: "#c9d4e6", fontSize: "0.95rem", lineHeight: 1.5, marginTop: 10 }}>
                      {f.desc}
                    </p>
                  </div>
                </Reveal>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* MAP */}
      <section className="map-section" style={{ padding: "80px 0" }}>
        <div className="container">
          <Reveal className="text-center mb-2">
            <span className="section-badge">
              <span className="pulse-dot"></span>
              Live CCTV Network
            </span>
            <h2 className="section-title">Sebaran Lokasi CCTV</h2>
            <p style={{ color: "#c9d4e6", fontSize: "1.1rem" }}>
              Pantau titik lokasi kamera pengawas di seluruh kota secara real-time.
            </p>
          </Reveal>

          <Reveal delay={90}>
            <div className="map-stats-row">
              <div className="map-stat">
                <div className="value"><CountUp end={cameras.length} /></div>
                <div className="label">Total CCTV</div>
              </div>
              <div className="map-stat">
                <div className="value"><CountUp end={onlineCount} /></div>
                <div className="label">Kamera Online</div>
              </div>
              <div className="map-stat">
                <div className="value"><CountUp end={totalKendaraan} /></div>
                <div className="label">Kendaraan Terdeteksi</div>
              </div>
              <div className="map-stat">
                <div className="value"><CountUp end={avgKepadatan} suffix="%" /></div>
                <div className="label">Kepadatan Rata-rata</div>
              </div>
            </div>
          </Reveal>

          <div className="map-legend">
            {STATUS_LEGEND.map((x) => (
              <span className="map-legend-item" key={x.s}>
                <span className="map-legend-dot" style={{ background: x.c, boxShadow: `0 0 8px ${x.c}` }}></span>
                {x.s}
              </span>
            ))}
            <span className="map-legend-item">
              <span className="map-legend-dot" style={{ background: "#9aa3b2", boxShadow: "0 0 8px #9aa3b2" }}></span>
              Tidak Ada Data
            </span>
            <span className="map-legend-item" style={{ color: "#9aa3b2" }}>
              <i className="bi bi-cursor me-1" style={{ fontSize: "0.9rem" }}></i>Klik titik CCTV untuk fokus
            </span>
          </div>

          <div className="row g-4 align-items-stretch">
            <div className="col-lg-8">
              <Reveal>
                <div className="map-container" style={{ height: "100%" }}>
                  <CctvMap cctv={cameras} liveMap={liveMap} focusId={focusId} focusTick={focusTick} />
                </div>
              </Reveal>
            </div>
            <div className="col-lg-4">
              <Reveal variant="reveal-right" delay={140}>
                <div className="camera-panel">
                  <div className="camera-panel-header">
                    <span className="camera-panel-title">
                      <i className="bi bi-camera-video me-2"></i>Daftar CCTV
                    </span>
                    <span className="camera-panel-count">
                      {onlineCount}/{cameras.length} Online
                    </span>
                  </div>
                  {cameras.length ? (
                    cameras.map((c) => {
                      const live = liveMap[c.id] || {};
                      const status = effectiveStatus(live);
                      const fresh = isLiveFresh(live);
                      const dens = fresh ? live.occupancy_persen ?? live.kepadatan_persen ?? 0 : 0;
                      const total = live.total ?? c.current_total ?? 0;
                      const color = statusColor(status);
                      const active = activeCam != null && String(activeCam) === String(c.id);
                      return (
                        <div
                          key={c.id}
                          className={`camera-list-item${active ? " active" : ""}`}
                          onClick={() => focusCamera(c.id)}
                        >
                          <span className="camera-item-dot" style={{ background: color, boxShadow: `0 0 8px ${color}` }}></span>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div className="d-flex justify-content-between align-items-center gap-2">
                              <span className="camera-item-name text-truncate">{c.name}</span>
                              <span className="camera-item-count" style={{ color }}>{total}</span>
                            </div>
                            <div className="camera-item-meta">
                              <span style={{ color }}>● {status}</span>
                              <span>{fresh ? `${dens}% kepadatan` : "Tidak ada data deteksi"}</span>
                            </div>
                            <div className="camera-dens-track">
                              <div
                                className="camera-dens-fill"
                                style={{
                                  width: `${Math.min(100, dens)}%`,
                                  background: color,
                                  boxShadow: `0 0 6px ${color}`,
                                }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <p className="text-muted text-center mb-0 py-4">Belum ada CCTV dengan koordinat.</p>
                  )}
                </div>
              </Reveal>
            </div>
          </div>

          <Reveal className="text-center mt-5">
            <Link to="/dashboard" className="btn-cta">
              <i className="bi bi-speedometer2 me-2"></i>Buka Dashboard Lengkap
            </Link>
          </Reveal>
        </div>
      </section>
    </>
  );
}

function HeroParticles() {
  const ref = useRef(null);
  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    const count = 25;
    for (let i = 0; i < count; i++) {
      const p = document.createElement("div");
      p.style.cssText = `position:absolute;bottom:-10px;width:${Math.random() * 5 + 3}px;height:${
        Math.random() * 5 + 3
      }px;border-radius:50%;background:rgba(59,130,246,0.4);left:${Math.random() * 100}%;animation:floatUp ${
        Math.random() * 10 + 8
      }s linear ${Math.random() * 10}s infinite;opacity:${Math.random() * 0.6 + 0.2}`;
      container.appendChild(p);
    }
    return () => (container.innerHTML = "");
  }, []);
  return (
    <div
      ref={ref}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        overflow: "hidden",
        pointerEvents: "none",
      }}
    />
  );
}
