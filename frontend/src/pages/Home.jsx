import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/Reveal";
import CctvMap from "../components/CctvMap";
import { listenCctv } from "../lib/firebase";

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

export default function Home() {
  const [cctv, setCctv] = useState([]);

  useEffect(() => {
    const off = listenCctv(setCctv);
    return off;
  }, []);

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
          background: "radial-gradient(circle at center, #0d0d0d 0%, #000000 100%)",
          position: "relative",
        }}
      >
        <HeroParticles />
        <div className="hero-content" style={{ maxWidth: 800, padding: "0 20px", position: "relative", zIndex: 1 }}>
          <h1 className="hero-title gradient-text" style={{ fontSize: "3.5rem", fontWeight: 800, lineHeight: 1.2 }}>
            Smart Traffic Vision
          </h1>
          <p
            className="hero-subtitle"
            style={{ fontSize: "1.2rem", color: "#bfdbfe", marginBottom: 40, lineHeight: 1.6 }}
          >
            Solusi cerdas pemantauan lalu lintas berbasis Artificial Intelligence. Dapatkan data kepadatan,
            kecepatan, dan pantauan visual secara real-time untuk kota yang lebih baik.
          </p>
          <div className="d-flex justify-content-center hero-cta">
            <Link to="/dashboard" className="btn-cta">
              Lihat Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="features-section" style={{ padding: "80px 0", backgroundColor: "#080808" }}>
        <div className="container">
          <div className="row g-4">
            {FEATURES.map((f) => (
              <div className="col-md-4" key={f.title}>
                <Reveal>
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
          <Reveal className="text-center mb-5">
            <h2 className="fw-bold mb-2">Sebaran Lokasi CCTV</h2>
            <p style={{ color: "#c9d4e6", fontSize: "1.1rem" }}>Pantau titik lokasi kamera pengawas di seluruh kota.</p>
          </Reveal>
          <Reveal>
            <div className="map-container">
              <CctvMap cctv={cctv} />
            </div>
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
