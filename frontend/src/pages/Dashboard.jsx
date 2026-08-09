import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import SummaryCard from "../components/SummaryCard";
import Reveal from "../components/Reveal";
import CountUp from "../components/CountUp";
import { TrafficBarChart, VehicleDoughnut } from "../components/Charts";
import { getArticles, getCctvList, listenLive, pushComment, imageUrl } from "../lib/firebase";
import { buildTrafficData, getVehicleDistribution, getSummary, classifySentiment, sentimentColor, statusColor, effectiveStatus, isLiveFresh } from "../lib/traffic";

const VEHICLE_META = [
  { name: "Mobil", color: "linear-gradient(135deg, #7CB9FF, #2563EB)" },
  { name: "Motor", color: "linear-gradient(135deg, #FFE45E, #FFC400)" },
  { name: "Bus", color: "linear-gradient(135deg, #5CF3B2, #00A86B)" },
  { name: "Truk", color: "linear-gradient(135deg, #FF7A70, #E53935)" },
];

function toEmbedUrl(url) {
  if (!url) return "";
  if (url.includes("watch?v=")) return `https://www.youtube.com/embed/${url.split("watch?v=")[1]}`;
  return url;
}

function trafficPillClass(status) {
  if (status === "Lancar") return "lancar";
  if (status === "Padat") return "padat";
  if (status === "Macet") return "macet";
  return "none";
}

export default function Dashboard() {
  const [cctv, setCctv] = useState([]);
  const [articles, setArticles] = useState([]);
  const [feedId, setFeedId] = useState("");
  const [chartId, setChartId] = useState("all");
  const [period, setPeriod] = useState("harian");
  const [summary, setSummary] = useState({
    kendaraan_hari_ini: 0,
    kepadatan_tertinggi: 0,
    rata_rata_kecepatan: "-",
    kamera_aktif: 0,
  });
  const [traffic, setTraffic] = useState({ labels: [], datasets: {} });
  const [vehicle, setVehicle] = useState({ data: [0, 0, 0, 0], percentages: ["0%", "0%", "0%", "0%"] });
  const [nama, setNama] = useState("");
  const [pesan, setPesan] = useState("");
  const [notice, setNotice] = useState("");
  const [live, setLive] = useState({});

  useEffect(() => {
    getCctvList().then(setCctv);
    getArticles({ published: 1 }).then((a) => setArticles(a.slice(0, 5)));
  }, []);

  useEffect(() => {
    if (!feedId) {
      setLive({});
      return;
    }
    const off = listenLive(feedId, setLive);
    return off;
  }, [feedId]);

  const refreshAll = useCallback(async () => {
    try {
      const [sum, t, v] = await Promise.all([
        getSummary(chartId === "all" ? null : chartId, cctv.length),
        buildTrafficData(chartId === "all" ? null : chartId, period),
        getVehicleDistribution(chartId === "all" ? null : chartId),
      ]);
      setSummary(sum);
      setTraffic(t);
      setVehicle(v);
    } catch (e) {
      console.error("Refresh dashboard error:", e);
    }
  }, [chartId, period, cctv.length]);

  useEffect(() => {
    refreshAll();
    const iv = setInterval(refreshAll, 5000);
    return () => clearInterval(iv);
  }, [refreshAll]);

  const selectedCctv = cctv.find((c) => String(c.id) === String(feedId));

  const sendComment = async (e) => {
    e.preventDefault();
    if (!pesan.trim()) {
      setNotice({ type: "danger", text: "Isi pesan dulu!" });
      return;
    }
    const now = new Date();
    const dayIndo = ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"][now.getDay()];
    await pushComment({
      nama: nama.trim() || "Warga Anonim",
      komentar: pesan.trim(),
      sentimen: classifySentiment(pesan),
      tanggal: now.toISOString().slice(0, 10),
      jam: now.toTimeString().slice(0, 8),
      hari: dayIndo,
      timestamp: Date.now() / 1000,
    });
    setPesan("");
    setNotice({ type: "success", text: "Laporan terkirim ke Firebase & Admin!" });
    setTimeout(() => setNotice(""), 3000);
  };

  return (
    <div className="container mt-4" style={{ paddingBottom: 50 }}>
      <Reveal>
        <div className="dash-hero">
          <span className="dash-hero-badge"><span className="pulse-dot"></span>Live Monitoring</span>
          <h1 className="dash-hero-title">Ringkasan Lalu Lintas Real-time</h1>
          <p className="dash-hero-sub">
            Pantau kondisi jalan dari setiap CCTV, tren kepadatan, kecepatan rata-rata, dan distribusi kendaraan secara langsung.
          </p>
          <div className="dash-hero-meta">
            <span className="refresh-chip"><span className="pulse-dot"></span>Auto-refresh 5 detik</span>
            <span className="refresh-chip"><i className="bi bi-database"></i>Data Firebase</span>
            {selectedCctv && (
              <span className="refresh-chip"><i className="bi bi-camera-video"></i>{selectedCctv.name}</span>
            )}
          </div>
        </div>
      </Reveal>
      <div className="row g-4 mb-5 justify-content-center">
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="car" chip="#3B82F6" value={summary.kendaraan_hari_ini.toLocaleString("id-ID")} label="Total Kendaraan" sub="Hari ini" /></div>
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="activity" chip="#FFD600" value={`${summary.kepadatan_tertinggi}%`} label="Kepadatan" sub="Saat ini" /></div>
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="gauge" chip="#00E676" value={summary.rata_rata_kecepatan} label="Rata-rata Kecepatan" sub="km/j" /></div>
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="camera" chip="#FF5252" value={summary.kamera_aktif} label="Kamera Aktif" sub="Online" /></div>
      </div>

      {/* LIVE FEED & ARTIKEL */}
      <div className="row g-4 mb-5">
        <div className="col-lg-8">
          <div className="dashboard-card">
            <div className="card-header-custom">
              <div className="video-panel-head">
                <span className="live-flag"><i className="bi bi-broadcast"></i>LIVE</span>
                <h4 className="mb-0">Streaming CCTV</h4>
              </div>
              <div className="d-flex gap-2 flex-wrap stream-controls">
                <select
                  className="form-select form-select-sm bg-dark text-white border-secondary select-live"
                  style={{ width: 250 }}
                  value={feedId}
                  onChange={(e) => setFeedId(e.target.value)}
                >
                  <option value="">-- Pilih CCTV untuk Menonton --</option>
                  {cctv.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="video-container position-relative">
              {selectedCctv ? (
                <iframe src={toEmbedUrl(selectedCctv.url)} title="Live CCTV" allowFullScreen />
              ) : (
                <div className="text-center text-muted">
                  <i className="bi bi-camera-video-off" style={{ fontSize: 48 }}></i>
                  <br />
                  Silakan pilih CCTV dari menu di atas untuk melihat tayangan.
                </div>
              )}
            </div>
            {selectedCctv && (
              <div className="live-status-strip">
                <div className="live-status-left">
                  <div className={`traffic-pill ${trafficPillClass(effectiveStatus(live))}`}>
                    <span className="t-dot"></span>{effectiveStatus(live)}
                  </div>
                  <div>
                    <div className="live-status-name" style={{ fontWeight: 700, color: "#e6edf7", fontSize: "0.82rem" }}>{selectedCctv.name}</div>
                    <div className="live-status-name">
                      {live.last_update ? `Update ${live.last_update}` : "Menunggu data deteksi..."}
                      {isLiveFresh(live) && live.queue != null ? ` • Antrean ${Math.round(live.queue * 100)}%` : ""}
                    </div>
                  </div>
                </div>
                <div className="live-status-right">
                  <div className="live-status-kpi">
                    <span className="kpi-ico"><i className="bi bi-car-front"></i></span>
                    <div className="live-status-value"><CountUp end={live.total ?? 0} /></div>
                    <div className="live-status-label">Kendaraan</div>
                  </div>
                  <div className="live-status-kpi">
                    <span className="kpi-ico"><i className="bi bi-activity"></i></span>
                    <div className="live-status-value"><CountUp end={isLiveFresh(live) ? live.occupancy_persen ?? live.kepadatan_persen ?? 0 : 0} suffix="%" /></div>
                    <div className="live-status-label">Kepadatan</div>
                  </div>
                  <div className="live-status-kpi">
                    <span className="kpi-ico"><i className="bi bi-speedometer2"></i></span>
                    {isLiveFresh(live) && live.kecepatan_kmh != null ? (
                      <div className="live-status-value"><CountUp end={live.kecepatan_kmh} suffix=" km/j" decimals={live.kecepatan_kmh % 1 !== 0 ? 1 : 0} /></div>
                    ) : (
                      <div className="live-status-value">— km/j</div>
                    )}
                    <div className="live-status-label">Kecepatan</div>
                  </div>
                </div>
                <div className="camera-dens-track" style={{ position: "absolute", bottom: 0, left: 0, right: 0, margin: 0, borderRadius: 0, height: 4 }}>
                  <div className="camera-dens-fill" style={{ width: `${Math.min(100, isLiveFresh(live) ? live.occupancy_persen ?? live.kepadatan_persen ?? 0 : 0)}%`, background: statusColor(effectiveStatus(live)) }}></div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="col-lg-4">
          <div className="dashboard-card">
            <div className="card-header-custom">
              <div className="sec-head-title">
                <span className="sec-head-ico"><i className="bi bi-newspaper"></i></span>
                <h4 className="mb-0 fw-bold">Artikel Terbaru</h4>
              </div>
              <Link to="/read_artikel" className="btn btn-sm btn-info rounded-pill">
                Lihat Semua
              </Link>
            </div>
            <div>
              {articles.map((a) => (
                <Link key={a.key} to={`/artikel/${a.id}`} className="article-item">
                  <img src={imageUrl(a.gambar)} alt="" />
                  <div>
                    <h6 className="mb-1 fw-bold small" style={{ color: "#fff" }}>{a.judul}</h6>
                    <small className="text-muted" style={{ fontSize: "0.7rem" }}>
                      {formatTanggal(a.tanggal)}
                    </small>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* STATISTIK */}
      <div className="sec-head">
        <div className="sec-head-title">
          <span className="sec-head-ico"><i className="bi bi-graph-up-arrow"></i></span>
          <div>
            <h4 className="fw-bold mb-0 gradient-text" style={{ color: "#fff" }}>Statistik & Analitik</h4>
            <small className="text-muted">Tren kepadatan & distribusi kendaraan per CCTV</small>
          </div>
        </div>
        <select
          className="form-select form-select-sm bg-dark text-white border-secondary select-live"
          style={{ width: 250 }}
          value={chartId}
          onChange={(e) => setChartId(e.target.value)}
        >
          <option value="all">Semua Data CCTV</option>
          {cctv.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </div>

      <div className="row g-4 mb-5">
        <div className="col-lg-7">
          <div className="dashboard-card" style={{ height: 850, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div className="card-header-custom">
              <h4 className="fw-bold">Laporan Data Kendaraan</h4>
              <div className="period-toggle">
                {["harian", "mingguan", "bulanan"].map((p) => (
                  <button
                    key={p}
                    className={period === p ? "active" : ""}
                    onClick={() => setPeriod(p)}
                  >
                    {p === "harian" ? "Hari" : p === "mingguan" ? "Minggu" : "Bulan"}
                  </button>
                ))}
              </div>
            </div>
            <TrafficBarChart labels={traffic.labels} datasets={traffic.datasets} period={period} height={650} />
          </div>
        </div>

        <div className="col-lg-5">
          <div className="dashboard-card" style={{ height: 850, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div className="card-header-custom">
              <h4>Distribusi Kendaraan</h4>
              <span className="badge rounded-pill text-dark" style={{ backgroundColor: "#3B82F6", padding: "6px 15px" }}>
                REAL TIME
              </span>
            </div>
            <div className="d-flex flex-column align-items-center justify-content-center" style={{ height: "100%" }}>
              <div style={{ marginBottom: 40 }}>
                <VehicleDoughnut data={vehicle.data} size={280} />
              </div>
              <div className="chart-legend-custom w-100 px-4">
                {vehicle.percentages.map((pct, i) => (
                  <div className="legend-item mb-3 p-2" key={i}>
                    <div>
                      <span className="legend-color" style={{ background: VEHICLE_META[i].color }}></span>
                      {VEHICLE_META[i].name}
                    </div>
                    <span className="fw-bold fs-5" style={{ color: "#3b82f6" }}>{pct}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* KOMENTAR */}
      <div className="row mb-5">
        <div className="col-12">
          <div className="dashboard-card" style={{ borderRadius: 16, padding: 25 }}>
            <div className="sec-head mb-4">
              <div className="sec-head-title">
                <span className="sec-head-ico"><i className="bi bi-chat-square-text"></i></span>
                <div>
                  <h4 className="mb-0 fw-bold">Kirim Ulasan / Komentar Masyarakat</h4>
                  <small className="text-muted">Komentar Anda dianalisis sentimennya secara otomatis</small>
                </div>
              </div>
            </div>
            {notice && (
              <div className={`alert alert-${notice.type} alert-dismissible`} role="alert">
                {notice.text}
                <button type="button" className="btn-close" onClick={() => setNotice("")}></button>
              </div>
            )}
            <form onSubmit={sendComment}>
              <div className="row g-3">
                <div className="col-md-4">
                  <label className="comment-label"><i className="bi bi-person me-1"></i>Nama Lengkap</label>
                  <input className="form-control form-dark" value={nama} onChange={(e) => setNama(e.target.value)} placeholder="Masukkan nama Anda" />
                </div>
                <div className="col-md-8">
                  <label className="comment-label"><i className="bi bi-chat-dots me-1"></i>Tulis Ulasan / Saran</label>
                  <div className="d-flex gap-2 comment-row">
                    <input className="form-control form-dark flex-grow-1" value={pesan} onChange={(e) => setPesan(e.target.value)} placeholder="Tulis komentar Anda di sini..." />
                    <button type="submit" className="btn-kirim px-4">
                      <i className="bi bi-send me-1"></i> Kirim
                    </button>
                  </div>
                  {pesan.trim() && (
                    <div className="d-flex align-items-center gap-2 mt-2">
                      <span className="text-muted small">
                        <i className="bi bi-magic me-1"></i>Sentimen terdeteksi otomatis:
                      </span>
                      <span className="badge rounded-pill px-3 py-2" style={{ background: sentimentColor(classifySentiment(pesan)), color: classifySentiment(pesan) === "Netral" ? "#000" : "#fff" }}>
                        {classifySentiment(pesan)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

// Logika sentimen pindah ke lib/traffic.js (classifySentiment, sentimentColor)
function formatTanggal(t) {
  if (!t) return "-";
  const d = new Date(String(t).replace(" ", "T"));
  if (isNaN(d)) return String(t).slice(0, 10);
  return d.toLocaleDateString("id-ID", { day: "2-digit", month: "short", year: "numeric" });
}
