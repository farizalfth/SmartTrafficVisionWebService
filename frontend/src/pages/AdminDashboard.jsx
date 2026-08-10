import { useEffect, useState, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import SummaryCard from "../components/SummaryCard";
import CountUp from "../components/CountUp";
import { TrafficBarChart, VehicleDoughnut, CommentBarChart } from "../components/Charts";
import {
  getCctvList,
  getArticles,
  saveCctv,
  deleteCctv,
  listenComments,
  listenLive,
  pushComment,
  updateComment,
  deleteComment,
  imageUrl,
} from "../lib/firebase";
import { buildTrafficData, buildTextSummaries, getVehicleDistribution, getSummary, classifySentiment, sentimentColor, effectiveStatus, isLiveFresh, statusColor } from "../lib/traffic";
import { getAiUrl, setAiServerUrl, clearAiServerUrl, getAiUrlSource } from "../lib/aiUrl";
const VEHICLE_META = [
  { name: "Mobil", color: "linear-gradient(135deg, #7CB9FF, #2563EB)" },
  { name: "Motor", color: "linear-gradient(135deg, #FFE45E, #FFC400)" },
  { name: "Bus", color: "linear-gradient(135deg, #5CF3B2, #00A86B)" },
  { name: "Truk", color: "linear-gradient(135deg, #FF7A70, #E53935)" },
];

const DAYS = ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"];
const MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"];
const ZONES = {
  WIB: { utc: 7, label: "WIB" },
  WITA: { utc: 8, label: "WITA" },
  WIT: { utc: 9, label: "WIT" },
};

function toEmbedUrl(url) {
  if (!url) return "";
  if (url.includes("watch?v=")) return `https://www.youtube.com/embed/${url.split("watch?v=")[1]}`;
  return url;
}

// Warna sinyal: hijau (bagus), kuning (cukup), merah (buruk/off)
function signalColor(v) {
  const n = Number(v) || 0;
  if (n >= 70) return "#00C853";
  if (n >= 40) return "#FFC400";
  return "#FF5252";
}

function trafficPillClass(status) {
  if (status === "Lancar") return "lancar";
  if (status === "Padat") return "padat";
  if (status === "Macet") return "macet";
  return "none";
}

export default function AdminDashboard() {
  const [cctv, setCctv] = useState([]);
  const [articles, setArticles] = useState([]);
  const [feedId, setFeedId] = useState("");
  const [chartId, setChartId] = useState("all");
  const [period, setPeriod] = useState("harian");
  const [summary, setSummary] = useState({});
  const [traffic, setTraffic] = useState({ labels: [], datasets: {} });
  const [textData, setTextData] = useState([]);
  const [vehicle, setVehicle] = useState({ data: [0, 0, 0, 0], percentages: ["0%", "0%", "0%", "0%"] });
  const [comments, setComments] = useState([]);
  const [server, setServer] = useState({});
  const [clock, setClock] = useState(Date.now());
  const [zone, setZone] = useState(() => localStorage.getItem("adminClockZone") || "WIB");
  const [cameraSearch, setCameraSearch] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ name: "", url: "", lat: "", lon: "", status: "Aktif", kapasitas: "", px_per_m: "", roi: "" });
  const [saving, setSaving] = useState(false);
  const [commentModal, setCommentModal] = useState(false);
  const [commentForm, setCommentForm] = useState({ key: null, nama: "", komentar: "", tanggal: "", jam: "" });
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [live, setLive] = useState({});
  const [aiUrlInput, setAiUrlInput] = useState(getAiUrl());
  const streamImgRef = useRef(null);
  const offsetRef = useRef(0);

  useEffect(() => {
    getCctvList().then(setCctv);
    getArticles({ published: 1 }).then((a) => setArticles(a.slice(0, 5)));
    const off = listenComments(setComments);
    return off;
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
      const [sum, t, v, tx] = await Promise.all([
        getSummary(chartId === "all" ? null : chartId, cctv.length),
        buildTrafficData(chartId === "all" ? null : chartId, period),
        getVehicleDistribution(chartId === "all" ? null : chartId),
        buildTextSummaries(chartId === "all" ? null : chartId, period),
      ]);
      setSummary(sum);
      setTraffic(t);
      setVehicle(v);
      setTextData(tx);
    } catch (e) {
      console.error("Refresh error:", e);
    }
  }, [chartId, period, cctv.length]);

  useEffect(() => {
    refreshAll();
    const iv = setInterval(refreshAll, 5000);
    return () => clearInterval(iv);
  }, [refreshAll]);

  // Server status polling
  useEffect(() => {
    const fetchStatus = async () => {
      const AI_URL = getAiUrl();
      if (!AI_URL) return;
      try {
        const res = await fetch(`${AI_URL}/api/server_status`);
        setServer(await res.json());
      } catch {
        setServer({ status: "OFFLINE", status_label: "TIDAK TERHUBUNG", stability: 0, cctv_total: cctv.length, cctv_online: 0, cctv_signals: [] });
      }
    };
    fetchStatus();
    const iv = setInterval(fetchStatus, 3000);
    return () => clearInterval(iv);
  }, [cctv.length]);

  useEffect(() => {
    if (server.server_time) {
      const e = wibToEpoch(server.server_time);
      if (e != null) offsetRef.current = e - Date.now();
    }
  }, [server.server_time]);

  useEffect(() => {
    const iv = setInterval(() => setClock(Date.now() + offsetRef.current), 1000);
    return () => clearInterval(iv);
  }, []);

  const pickZone = (z) => {
    setZone(z);
    localStorage.setItem("adminClockZone", z);
  };

  const selectedCctv = cctv.find((c) => String(c.id) === String(feedId));

  const performAnalysis = () => {
    if (!feedId) {
      alert("Pilih CCTV terlebih dahulu!");
      return;
    }
    const AI_URL = getAiUrl();
    if (!AI_URL) {
      alert("VITE_AI_SERVER_URL belum diisi. Arahkan ke AI server (contoh http://localhost:5000).");
      return;
    }
    setAnalyzing(true);
    setAnalysis(`${AI_URL}/video_feed?cctv_id=${feedId}&t=${Date.now()}`);
  };

  const stopAnalysis = () => {
    setAnalysis(null);
    setAnalyzing(false);
  };

  const saveAiUrl = () => {
    setAiServerUrl(aiUrlInput);
    setAiUrlInput(getAiUrl());
    setServer((prev) => (prev && prev.status === "ONLINE" ? { ...prev, status: "OFFLINE", status_label: "TIDAK TERHUBUNG", cctv_signals: [] } : prev));
  };

  const resetAiUrl = () => {
    clearAiServerUrl();
    setAiUrlInput(getAiUrl());
    setServer((prev) => (prev && prev.status === "ONLINE" ? { ...prev, status: "OFFLINE", status_label: "TIDAK TERHUBUNG", cctv_signals: [] } : prev));
  };

  const openAdd = () => {
    setEditing(null);
    setForm({ name: "", url: "", lat: "", lon: "", status: "Aktif", kapasitas: "", px_per_m: "", roi: "" });
    setModalOpen(true);
  };

  const openEdit = (c) => {
    setEditing(c);
    setForm({
      name: c.name,
      url: c.url || c.youtube_link || "",
      lat: c.lat == null ? "" : c.lat,
      lon: c.lon == null ? "" : c.lon,
      status: c.status || "Aktif",
      kapasitas: c.kapasitas ?? "",
      px_per_m: c.px_per_m ?? "",
      roi: Array.isArray(c.roi) ? c.roi.join(", ") : "",
    });
    setModalOpen(true);
  };

  const submitCamera = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      let roi = null;
      const roiRaw = String(form.roi || "").trim();
      if (roiRaw !== "") {
        const parts = roiRaw.split(",").map((v) => Number(v.trim()));
        if (parts.length === 4 && parts.every((v) => !Number.isNaN(v))) {
          roi = parts;
        } else {
          alert("ROI harus 4 angka dipisah koma: left, top, right, bottom (0-1)");
          return;
        }
      }
      const payload = {
        name: form.name.trim(),
        url: form.url.trim(),
        lat: form.lat === "" ? null : Number(form.lat),
        lon: form.lon === "" ? null : Number(form.lon),
        status: form.status,
      };
      if (form.kapasitas !== "") payload.kapasitas = Number(form.kapasitas);
      if (form.px_per_m !== "") payload.px_per_m = Number(form.px_per_m);
      if (roi) payload.roi = roi;
      if (!payload.name || !payload.url) {
        alert("Nama dan URL wajib diisi");
        return;
      }
      await saveCctv(editing ? { ...payload, id: editing.id } : payload);
      setModalOpen(false);
      getCctvList().then(setCctv);
    } catch (err) {
      alert(`Gagal menyimpan kamera: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const confirmDelete = async (c) => {
    if (confirm(`Apakah Anda yakin ingin menghapus kamera "${c.name}"?`)) {
      await deleteCctv(c.id);
      getCctvList().then(setCctv);
    }
  };

  const openCommentAdd = () => {
    setCommentForm({ key: null, nama: "", komentar: "", tanggal: "", jam: "" });
    setCommentModal(true);
  };

  const openCommentEdit = (c) => {
    setCommentForm({
      key: c.key,
      nama: c.nama || "",
      komentar: c.komentar || "",
      tanggal: c.tanggal || "",
      jam: c.jam || "",
    });
    setCommentModal(true);
  };

  const submitComment = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      const now = new Date();
      const payload = {
        nama: commentForm.nama.trim() || "Warga Anonim",
        komentar: commentForm.komentar.trim(),
        sentimen: classifySentiment(commentForm.komentar),
        tanggal: commentForm.tanggal || now.toISOString().slice(0, 10),
        jam: commentForm.jam || now.toTimeString().slice(0, 8),
        timestamp: Date.now() / 1000,
      };
      if (commentForm.key) {
        await updateComment(commentForm.key, payload);
      } else {
        await pushComment(payload);
      }
      setCommentModal(false);
    } catch (err) {
      alert(`Gagal menyimpan ulasan: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const confirmDeleteComment = async (c) => {
    if (confirm(`Hapus ulasan dari "${c.nama}"?`)) {
      await deleteComment(c.key);
    }
  };

  const commentData = useMemoComments(comments);
  const filteredCameras = cctv.filter(
    (c) =>
      c.name.toLowerCase().includes(cameraSearch.toLowerCase()) ||
      String(c.id).includes(cameraSearch)
  );

  const isOnline = server.status === "ONLINE";
  const stability = server.stability ?? 99.5;
  const serverColor = isOnline ? signalColor(stability) : "#FF5252";

  const renderSignalBar = (s) => {
    const val = s.online ? Number(s.signal) || 0 : 0;
    const color = signalColor(val);
    const width = Math.max(s.online ? 6 : 2, Math.min(100, val));
    return (
      <div className="cctv-signal-item" key={s.id}>
        <span className="signal-dot" style={{ background: color }}></span>
        <span className="text-truncate" style={{ maxWidth: 130 }}>{s.name}</span>
        <div className="signal-bar-track">
          <div className="signal-bar-fill" style={{ width: `${width}%`, background: color }}></div>
        </div>
        <span className="small" style={{ color, fontWeight: 600, minWidth: 44, textAlign: "right" }}>
          {s.online ? `${Math.round(val)}%` : "OFF"}
        </span>
      </div>
    );
  };

  return (
    <div className="container mt-4" style={{ paddingBottom: 50 }}>
      <h4 className="mb-3 fw-bold text-center gradient-text">Ringkasan Lalu Lintas</h4>
      <div className="row g-4 mb-5 justify-content-center">
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="car" chip="#3B82F6" value={(summary.kendaraan_hari_ini || 0).toLocaleString("id-ID")} label="Total Kendaraan" /></div>
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="activity" chip="#FFD600" value={`${summary.kepadatan_tertinggi || 0}%`} label="Kepadatan" /></div>
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="gauge" chip="#3B82F6" value={summary.rata_rata_kecepatan || "-"} label="Rata-rata Kecepatan" /></div>
        <div className="col-md-3 col-sm-6 col-6"><SummaryCard icon="camera" chip="#FF5252" value={summary.kamera_aktif || 0} label="Kamera Aktif" /></div>
      </div>

      {/* LIVE FEED & ARTIKEL */}
      <div className="row g-4 mb-5">
        <div className="col-lg-8">
          <div className="dashboard-card position-relative">
            <div className="card-header-custom">
              <h4>Live Feed CCTV Utama</h4>
              <div className="admin-stream-controls">
                <select className="form-select form-select-sm bg-dark text-white border-secondary" value={feedId} onChange={(e) => setFeedId(e.target.value)}>
                  <option value="">-- Pilih CCTV --</option>
                  {cctv.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
                <button className="btn btn-sm fw-bold" style={{ background: "linear-gradient(45deg,#FFD600,#FF6D00)", color: "#000" }} onClick={performAnalysis}>
                  <i className="bi bi-qr-code-scan me-1"></i>Deteksi Kendaraan
                </button>
              </div>
            </div>
            <div className="video-container position-relative">
              {analysis ? (
                <>
                  <img ref={streamImgRef} src={analysis} alt="Deteksi" style={{ width: "100%", height: 400, objectFit: "contain", background: "#000" }} onLoad={() => setAnalyzing(false)} onError={() => { setAnalyzing(false); alert("Gagal memuat stream deteksi. Pastikan AI server berjalan dan yolo11n.pt tersedia."); }} />
                  {analyzing && (
                    <div className="analysis-overlay">
                      <div className="spinner mb-3"></div>
                      <p className="text-warning">Menganalisis Frame Real-time...</p>
                    </div>
                  )}
                  <button className="btn btn-sm btn-danger position-absolute" style={{ top: 10, right: 10, zIndex: 10 }} onClick={stopAnalysis}>
                    <i className="bi bi-x-lg me-1"></i>Kembali ke Live
                  </button>
                </>
              ) : selectedCctv ? (
                <iframe src={toEmbedUrl(selectedCctv.url)} title="Live CCTV" allowFullScreen />
              ) : (
                <div className="text-center text-muted">
                  <i className="bi bi-camera-video-off" style={{ fontSize: 48 }}></i><br />
                  Silakan pilih CCTV...
                </div>
              )}
            </div>
            {analysis && !analyzing && (
              <div className="text-center mt-3 mb-1">
                <span className="badge rounded-pill text-dark fw-bold px-3 py-2" style={{ background: "linear-gradient(45deg,#FFD600,#FF6D00)" }}>
                  <i className="bi bi-qr-code-scan me-1"></i>HASIL DETEKSI YOLO11
                </span>
                <span className="badge rounded-pill text-white fw-bold px-3 py-2 ms-2" style={{ background: "#e11d48" }}>
                  <i className="bi bi-broadcast-pin me-1"></i>Live Streaming Aktif
                </span>
                <div className="d-flex justify-content-center align-items-center gap-2 mt-2 small text-muted">
                  <span className="live-badge"><i className="bi bi-camera-video me-1"></i>LIVE</span>
                  <span><i className="bi bi-camera me-1"></i>{selectedCctv?.name}</span>
                </div>
              </div>
            )}
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
              <h4>Artikel Terbaru</h4>
              <Link to="/kelola_artikel" className="btn btn-sm btn-info rounded-pill">Kelola</Link>
            </div>
            <div>
              {articles.map((a) => (
                <Link key={a.key} to={`/artikel/${a.id}`} className="article-item">
                  <img src={imageUrl(a.gambar)} alt="" />
                  <div>
                    <h6 className="mb-1 fw-bold small" style={{ color: "#fff" }}>{a.judul}</h6>
                    <small className="text-muted" style={{ fontSize: "0.7rem" }}>{fmtDate(a.tanggal)}</small>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* STATISTIK */}
      <div className="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-3">
        <h4 className="fw-bold mb-0 gradient-text">Statistik & Analitik</h4>
        <select className="form-select form-select-sm bg-dark text-white border-secondary admin-chart-select" value={chartId} onChange={(e) => setChartId(e.target.value)}>
          <option value="all">Semua Data CCTV</option>
          {cctv.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
      </div>

      <div className="row g-4 mb-5">
        <div className="col-lg-7">
          <div className="dashboard-card admin-chart-card">
            <div className="card-header-custom">
              <h4 className="fw-bold">Laporan Data Kendaraan</h4>
              <div className="period-toggle">
                {["harian", "mingguan", "bulanan"].map((p) => (
                  <button key={p} className={period === p ? "active" : ""} onClick={() => setPeriod(p)}>
                    {p === "harian" ? "Hari" : p === "mingguan" ? "Minggu" : "Bulan"}
                  </button>
                ))}
              </div>
            </div>
            <TrafficBarChart labels={traffic.labels} datasets={traffic.datasets} period={period} height={650} />
          </div>
        </div>
        <div className="col-lg-5">
          <div className="dashboard-card admin-chart-card">
            <div className="card-header-custom">
              <h4>Distribusi Kendaraan</h4>
              <span className="badge rounded-pill text-dark" style={{ backgroundColor: "#3B82F6", padding: "6px 15px" }}>REAL TIME</span>
            </div>
            <div className="d-flex flex-column align-items-center justify-content-center" style={{ height: "100%" }}>
              <div style={{ marginBottom: 40 }}><VehicleDoughnut data={vehicle.data} size={280} /></div>
              <div className="chart-legend-custom w-100 px-4">
                {vehicle.percentages.map((pct, i) => (
                  <div className="legend-item mb-3 p-2" key={i}>
                    <div><span className="legend-color" style={{ background: VEHICLE_META[i].color }}></span>{VEHICLE_META[i].name}</div>
                    <span className="fw-bold fs-5" style={{ color: "#3b82f6" }}>{pct}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* RINGKASAN DATA TEKS */}
      <div className="dashboard-card text-summary-card mb-4">
        <div className="card-header-custom">
          <h4>Ringkasan Data Teks</h4>
          <span className="badge rounded-pill text-dark fw-bold px-3 py-2" style={{ background: "linear-gradient(45deg,#FFD600,#FF6D00)" }}>
            <i className="bi bi-file-text me-1"></i>
            {period === "harian" ? "PER HARI" : period === "mingguan" ? "PER MINGGU" : "PER BULAN"}
          </span>
        </div>

        {/* Tabel desktop */}
        <div className="text-summary-table-wrap d-none d-md-block">
          <table className="table table-dark table-hover text-summary-table mb-0">
            <thead>
              <tr>
                <th>Periode</th>
                <th className="text-center">Total</th>
                <th className="text-center">Mobil</th>
                <th className="text-center">Motor</th>
                <th className="text-center">Bus</th>
                <th className="text-center">Truk</th>
                <th className="text-center">Kepadatan</th>
                <th className="text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {textData.map((r) => (
                <tr key={r.label}>
                  <td className="fw-bold" style={{ color: "#e6edf7" }}>{r.label}</td>
                  <td className="text-center fw-bold" style={{ color: "#3b82f6" }}>{r.total.toLocaleString("id-ID")}</td>
                  <td className="text-center text-muted">{r.mobil.toLocaleString("id-ID")}</td>
                  <td className="text-center text-muted">{r.motor.toLocaleString("id-ID")}</td>
                  <td className="text-center text-muted">{r.bus.toLocaleString("id-ID")}</td>
                  <td className="text-center text-muted">{r.truk.toLocaleString("id-ID")}</td>
                  <td className="text-center">{r.kepadatan != null ? `${r.kepadatan}%` : "-"}</td>
                  <td className="text-center">
                    {r.status ? (
                      <span className={`traffic-pill ${trafficPillClass(r.status)}`}><span className="t-dot"></span>{r.status}</span>
                    ) : "-"}
                  </td>
                </tr>
              ))}
              {!textData.length && (
                <tr><td colSpan={8} className="text-center text-muted py-3">Belum ada data pada periode ini.</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Kartu khusus mobile */}
        <div className="text-summary-mobile d-md-none">
          {textData.map((r) => (
            <div className="text-summary-mobile-card" key={r.label}>
              <div className="text-summary-mobile-head">
                <span className="text-summary-mobile-label">{r.label}</span>
                {r.status ? (
                  <span className={`traffic-pill ${trafficPillClass(r.status)}`}><span className="t-dot"></span>{r.status}</span>
                ) : (
                  <span className="text-muted small fw-bold">-</span>
                )}
              </div>
              <div className="text-summary-mobile-grid">
                <div className="text-summary-mobile-cell">
                  <span className="tsm-value" style={{ color: "#3b82f6" }}>{r.total.toLocaleString("id-ID")}</span>
                  <span className="tsm-label">Total</span>
                </div>
                <div className="text-summary-mobile-cell">
                  <span className="tsm-value">{r.mobil.toLocaleString("id-ID")}</span>
                  <span className="tsm-label">Mobil</span>
                </div>
                <div className="text-summary-mobile-cell">
                  <span className="tsm-value">{r.motor.toLocaleString("id-ID")}</span>
                  <span className="tsm-label">Motor</span>
                </div>
                <div className="text-summary-mobile-cell">
                  <span className="tsm-value">{r.bus.toLocaleString("id-ID")}</span>
                  <span className="tsm-label">Bus</span>
                </div>
                <div className="text-summary-mobile-cell">
                  <span className="tsm-value">{r.truk.toLocaleString("id-ID")}</span>
                  <span className="tsm-label">Truk</span>
                </div>
                <div className="text-summary-mobile-cell">
                  <span className="tsm-value">{r.kepadatan != null ? `${r.kepadatan}%` : "-"}</span>
                  <span className="tsm-label">Kepadatan</span>
                </div>
              </div>
            </div>
          ))}
          {!textData.length && <p className="text-muted text-center py-3">Belum ada data pada periode ini.</p>}
        </div>
      </div>

      {/* TABEL DETAIL KENDARAAN */}
      <div className="dashboard-card text-summary-card mb-4">
        <div className="card-header-custom">
          <h4>Tabel Detail Kendaraan</h4>
          <span className="badge rounded-pill text-dark fw-bold" style={{ backgroundColor: "#0dcaf0", padding: "6px 15px" }}>REAL TIME</span>
        </div>

        {/* Tabel desktop */}
        <div className="text-summary-table-wrap d-none d-md-block">
          <table className="table table-dark table-striped align-middle mb-0">
            <thead>
              <tr>
                <th>Periode</th>
                <th className="text-end">Mobil</th>
                <th className="text-end">Motor</th>
                <th className="text-end">Bus</th>
                <th className="text-end">Truk</th>
                <th className="text-end">Total</th>
              </tr>
            </thead>
            <tbody>
              {traffic.labels.map((lbl, i) => {
                const label = Array.isArray(lbl) ? lbl.join(" · ") : lbl;
                const vals = ["mobil", "motor", "bus", "truk"].map((k) => traffic.datasets[k]?.[i] || 0);
                const total = vals.reduce((a, b) => a + b, 0);
                return (
                  <tr key={i}>
                    <td className="fw-bold" style={{ color: "#e6edf7" }}>{label}</td>
                    {vals.map((v, j) => <td key={j} className="text-end text-muted">{v.toLocaleString("id-ID")}</td>)}
                    <td className="text-end fw-bold" style={{ color: "#3b82f6" }}>{total.toLocaleString("id-ID")}</td>
                  </tr>
                );
              })}
              {!traffic.labels.length && (
                <tr><td colSpan={6} className="text-center text-muted">Belum ada data.</td></tr>
              )}
            </tbody>
            {traffic.labels.length > 0 && (
              <tfoot>
                <tr style={{ borderTop: "2px solid #0dcaf0" }}>
                  <td className="fw-bold">Total</td>
                  {["mobil", "motor", "bus", "truk"].map((k) => {
                    const v = (traffic.datasets[k] || []).reduce((a, b) => a + (b || 0), 0);
                    return <td key={k} className="text-end">{v.toLocaleString("id-ID")}</td>;
                  })}
                  <td className="text-end fw-bold" style={{ color: "#0dcaf0" }}>
                    {traffic.labels.reduce((a, _, i) => a + ["mobil", "motor", "bus", "truk"].reduce((s, k) => s + (traffic.datasets[k]?.[i] || 0), 0), 0).toLocaleString("id-ID")}
                  </td>
                </tr>
              </tfoot>
            )}
          </table>
        </div>

        {/* Kartu khusus mobile */}
        <div className="text-summary-mobile d-md-none">
          {traffic.labels.map((lbl, i) => {
            const label = Array.isArray(lbl) ? lbl.join(" · ") : lbl;
            const vals = ["mobil", "motor", "bus", "truk"].map((k) => traffic.datasets[k]?.[i] || 0);
            const total = vals.reduce((a, b) => a + b, 0);
            return (
              <div className="text-summary-mobile-card" key={i}>
                <div className="text-summary-mobile-head">
                  <span className="text-summary-mobile-label">{label}</span>
                </div>
                <div className="text-summary-mobile-grid">
                  <div className="text-summary-mobile-cell">
                    <span className="tsm-value">{vals[0].toLocaleString("id-ID")}</span>
                    <span className="tsm-label">Mobil</span>
                  </div>
                  <div className="text-summary-mobile-cell">
                    <span className="tsm-value">{vals[1].toLocaleString("id-ID")}</span>
                    <span className="tsm-label">Motor</span>
                  </div>
                  <div className="text-summary-mobile-cell">
                    <span className="tsm-value">{vals[2].toLocaleString("id-ID")}</span>
                    <span className="tsm-label">Bus</span>
                  </div>
                  <div className="text-summary-mobile-cell">
                    <span className="tsm-value">{vals[3].toLocaleString("id-ID")}</span>
                    <span className="tsm-label">Truk</span>
                  </div>
                  <div className="text-summary-mobile-cell tsm-wide" style={{ background: "rgba(13,202,240,0.12)", borderColor: "rgba(13,202,240,0.35)" }}>
                    <span className="tsm-value" style={{ color: "#0dcaf0" }}>{total.toLocaleString("id-ID")}</span>
                    <span className="tsm-label" style={{ color: "#7dd3fc" }}>Total</span>
                  </div>
                </div>
              </div>
            );
          })}
          {!traffic.labels.length && <p className="text-muted text-center py-3">Belum ada data.</p>}
        </div>
      </div>

      {/* KOMENTAR ANALITIK */}
      <div className="row g-4 mb-5">
        <div className="col-lg-6">
          <div className="dashboard-card">
            <div className="card-header-custom">
              <h4>Tren Ulasan Masuk</h4>
              <i className="bi bi-graph-up-arrow gradient-icon" style={{ color: "#ffd600" }}></i>
            </div>
            <CommentBarChart labels={commentData.labels} baik={commentData.baik} netral={commentData.netral} buruk={commentData.buruk} />
          </div>
        </div>
        <div className="col-lg-6">
          <div className="dashboard-card">
            <div className="text-center mb-3">
              <h4 className="fw-bold mb-1" style={{ color: "#fff" }}>Daftar Ulasan Terbaru</h4>
              <span className="badge rounded-pill text-dark px-3 py-2" style={{ background: "#3B82F6" }}>{comments.length} Ulasan</span>
            </div>
            <div className="d-flex justify-content-end mb-2">
              <button className="btn btn-sm btn-info rounded-pill" onClick={openCommentAdd}>
                <i className="bi bi-plus-lg me-1"></i>Tambah
              </button>
            </div>
            <div style={{ maxHeight: 400, overflowY: "auto" }}>
              {comments.slice(0, 20).map((c) => (
                <div key={c.key} className="comment-item">
                  <div className="d-flex justify-content-between align-items-center gap-2">
                    <div className="d-flex align-items-center gap-2" style={{ minWidth: 0 }}>
                      <span className="badge" style={{ background: sentimentColor(c.sentimen), color: c.sentimen === "Netral" ? "#000" : "#fff" }}>{c.sentimen}</span>
                      <strong className="small text-truncate" style={{ color: "#fff" }}>{c.nama}</strong>
                    </div>
                    <div className="comment-actions">
                      <button className="btn btn-sm btn-warning" title="Edit" onClick={() => openCommentEdit(c)}><i className="bi bi-pencil"></i></button>
                      <button className="btn btn-sm btn-danger" title="Hapus" onClick={() => confirmDeleteComment(c)}><i className="bi bi-trash"></i></button>
                    </div>
                  </div>
                  <p className="text-muted mb-1 small" style={{ marginTop: 6 }}>{c.komentar}</p>
                  <small className="text-muted" style={{ fontSize: "0.7rem" }}>{c.tanggal} {c.jam}</small>
                </div>
              ))}
              {!comments.length && <p className="text-muted text-center py-3">Belum ada ulasan.</p>}
            </div>
          </div>
        </div>
      </div>

      {/* KAMERA & STATUS SISTEM */}
      <div className="row g-4">
        <div className="col-lg-8">
          <div className="dashboard-card">
            <div className="card-header-custom">
              <h4>Manajemen Kamera</h4>
              <button className="btn btn-sm btn-info rounded-pill" onClick={openAdd}>
                <i className="bi bi-plus-lg me-1"></i>Tambah
              </button>
            </div>
            <input className="form-control rounded-pill mb-3" style={{ color: "#fff", caretColor: "#3b82f6" }} placeholder="Cari kamera berdasarkan nama..." value={cameraSearch} onChange={(e) => setCameraSearch(e.target.value)} />
            <div>
              {filteredCameras.map((c) => (
                <div className="camera-item" key={c.id}>
                  <div className="camera-thumb"><i className="bi bi-camera-video" style={{ color: "#3B82F6" }}></i></div>
                  <div className="camera-item-info">
                    <div className="camera-name">{c.name}</div>
                    <div className="camera-loc">ID: {c.id} • {c.status} • {c.lat != null ? `${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}` : "-"}</div>
                    <div className="camera-loc">Kap: {c.kapasitas ?? 15} • Skala: {c.px_per_m ?? 30} px/m{c.roi ? ` • ROI: ${Array.isArray(c.roi) ? c.roi.join(", ") : c.roi}` : ""}</div>
                  </div>
                  <div className="camera-item-actions">
                    <button className="btn btn-sm btn-warning" onClick={() => openEdit(c)}><i className="bi bi-pencil"></i></button>
                    <button className="btn btn-sm btn-danger" onClick={() => confirmDelete(c)}><i className="bi bi-trash"></i></button>
                  </div>
                </div>
              ))}
              {!filteredCameras.length && <p className="text-muted text-center py-3">Tidak ada kamera.</p>}
            </div>
          </div>
        </div>

        <div className="col-lg-4">
          <div className="server-status-card">
            <div className="text-center mb-3">
              <span className="status-indicator" style={{ background: serverColor, marginBottom: 8 }}></span>
              <div>
                <strong style={{ color: serverColor, fontSize: "1.05rem" }}>
                  {server.status === "OFFLINE" || !server.status ? "OFFLINE • TIDAK TERHUBUNG" : `ONLINE • ${server.status_label}`}
                </strong>
              </div>
            </div>
            <div className="d-flex align-items-center gap-2 mb-2">
              <span className="text-muted small">Stabilitas</span>
              <span className="ms-auto fw-bold" style={{ color: serverColor }}>{isOnline ? `${stability}%` : "-"}</span>
            </div>
            <div className="status-bar-track">
              <div className="status-bar-fill" style={{ width: `${isOnline ? Math.min(100, stability) : 0}%`, background: serverColor }}></div>
            </div>
            <div style={{ maxHeight: 220, overflowY: "auto" }}>
              {(server.cctv_signals || []).map((s) => renderSignalBar(s))}
              {!server.cctv_signals?.length && (
                <p className="text-muted small text-center py-2">
                  {server.status ? "Belum ada data sinyal CCTV dari server AI." : "Menunggu koneksi ke server AI..."}
                </p>
              )}
            </div>
            <div className="d-flex justify-content-between mt-3 pt-3" style={{ borderTop: "1px solid #2a2a2a" }}>
              <span className="text-muted small">Uptime: <b style={{ color: "#fff" }}>{server.uptime || "-"}</b></span>
            </div>
            <div className="server-clock admin-clock">
              <div className="server-clock-chip">
                <span className="pulse-dot"></span>JAM SERVER
              </div>
              <div className="server-clock-time">{formatClock(clock, zone)}</div>
              <div className="server-clock-date">{formatClockDate(clock, zone)}</div>
              <div className="zone-toggle">
                {Object.keys(ZONES).map((z) => (
                  <button
                    key={z}
                    className={zone === z ? "active" : ""}
                    onClick={() => pickZone(z)}
                    title={`UTC+${ZONES[z].utc}`}
                  >
                    {z}
                  </button>
                ))}
              </div>
            </div>
            <div className="text-center mt-2">
              <span className="badge" style={{ background: "rgba(59,130,246,0.15)", color: "#3B82F6" }}>
                <i className="bi bi-arrow-repeat me-1"></i>Update Deteksi: {fmtDetTime(server.last_update)}
              </span>
            </div>
            <div className="ai-url-box mt-3">
              <label className="small text-muted mb-1 d-block">
                <i className="bi bi-gear me-1"></i>URL Server AI
              </label>
              <div className="d-flex gap-1">
                <input
                  className="form-control form-control-sm bg-dark text-white border-secondary"
                  value={aiUrlInput}
                  onChange={(e) => setAiUrlInput(e.target.value)}
                  placeholder="http://ip-server:5000"
                />
                <button className="btn btn-sm btn-info" title="Simpan URL" onClick={saveAiUrl}><i className="bi bi-check-lg"></i></button>
                <button className="btn btn-sm btn-outline-light" title="Kembalikan otomatis" onClick={resetAiUrl}><i className="bi bi-arrow-counterclockwise"></i></button>
              </div>
              <small className="text-muted" style={{ fontSize: "0.68rem" }}>Saat ini: {getAiUrlSource().source === "override" ? "manual" : getAiUrlSource().source === "env" ? "dari build" : "otomatis"} • {getAiUrl()}</small>
            </div>
          </div>
        </div>
      </div>

      {/* MODAL KAMERA */}
      {modalOpen && (
        <div className="modal show d-block" tabIndex="-1" role="dialog">
          <div className="modal-dialog modal-dialog-centered">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">{editing ? "Edit Kamera" : "Tambah Kamera"}</h5>
                <button type="button" className="btn-close" onClick={() => setModalOpen(false)} style={{ filter: "invert(1)" }}></button>
              </div>
              <form onSubmit={submitCamera}>
                <div className="modal-body">
                  <div className="mb-3">
                    <label className="form-label">Nama Kamera</label>
                    <input className="form-control" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Nama kamera" required />
                  </div>
                  <div className="mb-3">
                    <label className="form-label">URL Stream YouTube</label>
                    <input className="form-control" value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} placeholder="https://www.youtube.com/watch?v=..." required />
                  </div>
                  <div className="row">
                    <div className="col-6 mb-3">
                      <label className="form-label">Latitude</label>
                      <input className="form-control" value={form.lat} onChange={(e) => setForm({ ...form, lat: e.target.value })} placeholder="-6.8797" />
                    </div>
                    <div className="col-6 mb-3">
                      <label className="form-label">Longitude</label>
                      <input className="form-control" value={form.lon} onChange={(e) => setForm({ ...form, lon: e.target.value })} placeholder="109.1256" />
                    </div>
                  </div>
                  <div className="mb-3">
                    <label className="form-label">Status</label>
                    <select className="form-select" value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value })}>
                      <option value="Aktif">Aktif</option>
                      <option value="Nonaktif">Nonaktif</option>
                    </select>
                  </div>
                  <hr />
                  <div className="row">
                    <div className="col-6 mb-3">
                      <label className="form-label">Kapasitas (max kendaraan)</label>
                      <input type="number" min="1" max="200" className="form-control" value={form.kapasitas} onChange={(e) => setForm({ ...form, kapasitas: e.target.value })} placeholder="15" />
                      <small className="text-muted">Referensi untuk kalibrasi kepadatan (opsional)</small>
                    </div>
                    <div className="col-6 mb-3">
                      <label className="form-label">Skala px/meter</label>
                      <input type="number" min="5" max="300" className="form-control" value={form.px_per_m} onChange={(e) => setForm({ ...form, px_per_m: e.target.value })} placeholder="30" />
                      <small className="text-muted">Kalibrasi skala jalan utk estimasi kecepatan (opsional)</small>
                    </div>
                  </div>
                  <div className="mb-3">
                    <label className="form-label">ROI (left, top, right, bottom)</label>
                    <input className="form-control" value={form.roi} onChange={(e) => setForm({ ...form, roi: e.target.value })} placeholder="0, 0.2, 1, 1" />
                    <small className="text-muted">Area deteksi ternormalisasi 0-1. Kosongkan = seluruh frame</small>
                  </div>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-secondary" onClick={() => setModalOpen(false)}>Batal</button>
                  <button type="submit" className="btn btn-primary" disabled={saving}>
                    {saving ? "Menyimpan..." : "Simpan Data"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
      {/* MODAL ULASAN */}
      {commentModal && (
        <div className="modal show d-block" tabIndex="-1" role="dialog">
          <div className="modal-dialog modal-dialog-centered">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">{commentForm.key ? "Edit Ulasan" : "Tambah Ulasan"}</h5>
                <button type="button" className="btn-close" onClick={() => setCommentModal(false)} style={{ filter: "invert(1)" }}></button>
              </div>
              <form onSubmit={submitComment}>
                <div className="modal-body">
                  <div className="mb-3">
                    <label className="form-label">Nama</label>
                    <input className="form-control" value={commentForm.nama} onChange={(e) => setCommentForm({ ...commentForm, nama: e.target.value })} placeholder="Nama pengirim" required />
                  </div>
                  <div className="mb-3">
                    <label className="form-label">Komentar</label>
                    <textarea className="form-control" rows={3} value={commentForm.komentar} onChange={(e) => setCommentForm({ ...commentForm, komentar: e.target.value })} placeholder="Isi ulasan" required />
                  </div>
                  <div className="mb-3">
                    <label className="form-label">Sentimen (otomatis)</label>
                    <div className="d-flex align-items-center gap-2">
                      <span className="badge rounded-pill px-3 py-2" style={{ background: sentimentColor(classifySentiment(commentForm.komentar)), color: classifySentiment(commentForm.komentar) === "Netral" ? "#000" : "#fff" }}>
                        {classifySentiment(commentForm.komentar)}
                      </span>
                      <small className="text-muted">terdeteksi otomatis dari isi komentar</small>
                    </div>
                  </div>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-secondary" onClick={() => setCommentModal(false)}>Batal</button>
                  <button type="submit" className="btn btn-primary" disabled={saving}>
                    {saving ? "Menyimpan..." : "Simpan Data"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function fmtDetTime(s) {
  if (!s) return "-";
  const m = String(s).match(/(\d{2}):(\d{2}):(\d{2})/);
  return m ? m[0] : String(s);
}

function wibToEpoch(s) {
  if (!s) return null;
  const m = s.match(/(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2}):(\d{2})/);
  if (!m) return null;
  return Date.UTC(+m[3], +m[2] - 1, +m[1], +m[4] - 7, +m[5], +m[6]);
}

function zoneParts(epochMs, zone) {
  const d = new Date(epochMs + ZONES[zone].utc * 3600 * 1000);
  return {
    h: d.getUTCHours(),
    m: d.getUTCMinutes(),
    s: d.getUTCSeconds(),
    day: d.getUTCDay(),
    date: d.getUTCDate(),
    month: d.getUTCMonth(),
    year: d.getUTCFullYear(),
  };
}

function formatClock(epochMs, zone) {
  const p = zoneParts(epochMs, zone);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(p.h)}:${pad(p.m)}:${pad(p.s)}`;
}

function formatClockDate(epochMs, zone) {
  const p = zoneParts(epochMs, zone);
  return `${DAYS[p.day]}, ${p.date} ${MONTHS[p.month]} ${p.year} (UTC+${ZONES[zone].utc})`;
}

function useMemoComments(comments) {
  const byDate = {};
  comments.forEach((c) => {
    const tgl = c.tanggal || "Data Lama";
    if (!byDate[tgl]) byDate[tgl] = { Baik: 0, Buruk: 0, Netral: 0 };
    if (c.sentimen === "Baik") byDate[tgl].Baik += 1;
    else if (c.sentimen === "Buruk") byDate[tgl].Buruk += 1;
    else byDate[tgl].Netral += 1;
  });
  const labels = Object.keys(byDate).sort();
  return {
    labels,
    baik: labels.map((d) => byDate[d].Baik),
    buruk: labels.map((d) => byDate[d].Buruk),
    netral: labels.map((d) => byDate[d].Netral),
  };
}

function fmtDate(t) {
  if (!t) return "-";
  const d = new Date(String(t).replace(" ", "T"));
  if (isNaN(d)) return String(t).slice(0, 10);
  return d.toLocaleDateString("id-ID", { day: "2-digit", month: "short", year: "numeric" });
}
