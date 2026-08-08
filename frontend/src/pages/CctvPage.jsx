import { useEffect, useState, useCallback } from "react";
import Reveal from "../components/Reveal";
import CctvMap from "../components/CctvMap";
import { getCctvList, getTrafficStats } from "../lib/firebase";
import { statusColor, effectiveStatus } from "../lib/traffic";

const AI_URL = import.meta.env.VITE_AI_SERVER_URL || "";
const DAYS = ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"];
const MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"];
const ZONES = {
  WIB: { utc: 7, label: "WIB" },
  WITA: { utc: 8, label: "WITA" },
  WIT: { utc: 9, label: "WIT" },
};

export default function CctvPage() {
  const [cctv, setCctv] = useState([]);
  const [liveMap, setLiveMap] = useState({});
  const [stats, setStats] = useState({ total: 0, aktif: 0, padat: 0, macet: 0 });
  const [selected, setSelected] = useState(null);
  const [clock, setClock] = useState(Date.now());
  const [clockSynced, setClockSynced] = useState(false);
  const [zone, setZone] = useState(() => localStorage.getItem("clockZone") || "WIB");

  useEffect(() => {
    let offset = 0;
    const tick = () => setClock(Date.now() + offset);
    const sync = async () => {
      try {
        const res = await fetch(`${AI_URL}/api/server_status`, { cache: "no-store" });
        const data = await res.json();
        const epoch = wibToEpoch(data.server_time);
        if (epoch != null) {
          offset = epoch - Date.now();
          setClockSynced(true);
        }
      } catch {
        setClockSynced(false);
      }
    };
    sync();
    const iv = setInterval(tick, 1000);
    const syncIv = setInterval(sync, 30000);
    return () => { clearInterval(iv); clearInterval(syncIv); };
  }, []);

  const pickZone = (z) => {
    setZone(z);
    localStorage.setItem("clockZone", z);
  };

  const refresh = useCallback(async () => {
    try {
      const statsData = await getTrafficStats();
      const live = {};
      Object.entries(statsData || {}).forEach(([id, node]) => {
        const l = node?.live || {};
        live[id] = l;
      });
      setLiveMap(live);

      let padat = 0, macet = 0, aktif = 0;
      Object.values(live).forEach((l) => {
        const s = effectiveStatus(l);
        if (s === "Padat") padat += 1;
        else if (s === "Macet") macet += 1;
        else if (s === "Lancar") aktif += 1;
      });
      setStats({ total: cctv.length, aktif, padat, macet });
    } catch (e) {
      console.error("Gagal sinkronisasi peta:", e);
    }
  }, [cctv.length]);

  useEffect(() => {
    getCctvList().then(setCctv);
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, [refresh]);

  const flyTo = (c) => {
    setSelected(c);
  };

  return (
    <div className="container" style={{ paddingBottom: 50 }}>
      <Reveal>
        <div className="page-header">
          <h2 className="page-title gradient-text">Peta Lokasi CCTV</h2>
          <p className="page-subtitle">Pantau titik kamera pengawas lalu lintas secara real-time.</p>
        </div>
      </Reveal>

      <div className="row g-3 mb-4">
        <StatCard icon="bi-camera-reels" chip="#3B82F6" value={stats.total} label="Total Titik CCTV" />
        <StatCard icon="bi-broadcast" chip="#00E676" value={stats.aktif} label="Aktif" />
        <StatCard icon="bi-cone-striped" chip="#FFD600" value={stats.padat} label="Kondisi Padat" />
        <StatCard icon="bi-exclamation-triangle" chip="#FF5252" value={stats.macet} label="Kondisi Macet" />
      </div>

      <div className="d-flex justify-content-center flex-wrap gap-2 mb-3">
        <span className="status-badge" style={{ background: "#00C853" }}>● Lancar</span>
        <span className="status-badge" style={{ background: "#FFD600", color: "#000" }}>● Padat</span>
        <span className="status-badge" style={{ background: "#FF5252" }}>● Macet</span>
        <span className="status-badge" style={{ background: "#9aa3b2", color: "#000" }}>● Tidak Ada Data</span>
        <span className="status-badge" style={{ background: "#3B82F6", color: "#000" }}><i className="bi bi-arrow-repeat me-1"></i>Auto-refresh 5 detik</span>
      </div>

      <Reveal>
        <div className="server-clock-wrap">
          <div className="server-clock">
            <div className="server-clock-chip">
              <span className="pulse-dot"></span>
              {clockSynced ? "JAM SERVER" : "JAM LOKAL"}
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
        </div>
      </Reveal>

      <Reveal>
        <div className="map-container" style={{ height: "50vh", minHeight: 380, maxHeight: 620 }}>
          <CctvMap cctv={cctv} liveMap={liveMap} height="100%" zoom={5} />
        </div>
      </Reveal>

      <p className="text-center text-muted mt-3">
        <i className="bi bi-broadcast me-1"></i>Status & jumlah kendaraan diperbarui otomatis dari Firebase setiap 5 detik
      </p>

      <div className="row g-4 mb-4">
        {["Klik Titik Kamera", "Warna Menunjukkan Kondisi", "Klik Kartu CCTV"].map((t, i) => {
          const icons = ["bi-mouse", "bi-cone-striped", "bi-crosshair"];
          const descs = [
            "Klik titik pada peta untuk melihat detail kamera dan status lalu lintas terbaru.",
            "Hijau = Lancar, Kuning = Padat, Merah = Macet, Abu-abu = Tidak Ada Data. Warna mengikuti data real-time.",
            "Pilih kartu CCTV di bawah untuk memindahkan peta ke lokasi kamera tersebut.",
          ];
          return (
            <div className="col-lg-4" key={t}>
              <div className="dashboard-card text-center" style={{ height: "100%" }}>
                <div className="icon-chip" style={{ "--chip": ["#3B82F6", "#00E676", "#FFD600"][i], margin: "0 auto 15px" }}>
                  <i className={`bi ${icons[i]}`} style={{ fontSize: 26 }}></i>
                </div>
                <h5 className="fw-bold" style={{ color: "#fff" }}>{t}</h5>
                <p className="text-muted mb-0" style={{ fontSize: "0.9rem" }}>{descs[i]}</p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="text-center mb-3">
        <h4 className="fw-bold mb-1" style={{ color: "#fff" }}>
          <i className="bi bi-collection-play me-2 gradient-icon"></i>Daftar Titik CCTV
        </h4>
        <span className="live-badge">LIVE</span>
      </div>

      <div className="row g-3 justify-content-center">
        {cctv.map((c) => {
          const live = liveMap[c.id] || {};
          const status = live.status || "Tidak Ada Data";
          const color = statusColor(status);
          return (
            <div className="col-md-6 col-lg-4" key={c.id}>
              <div className="cctv-card" onClick={() => flyTo(c)}>
                <div className="d-flex align-items-center justify-content-between mb-2">
                  <div className="d-flex align-items-center gap-2">
                    <span className="signal-dot online" style={{ background: color }}></span>
                    <span className="camera-name">{c.name}</span>
                  </div>
                  <span className="status-badge" style={{ background: color, color: status === "Padat" ? "#000" : "#fff" }}>
                    {status}
                  </span>
                </div>
                <div className="camera-loc">
                  <b>{live.total ?? 0}</b> kendaraan terdeteksi
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
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

function StatCard({ icon, chip, value, label }) {
  return (
    <div className="col-md-3 col-6">
      <div className="kpi-card">
        <div className="icon-chip" style={{ "--chip": chip, margin: "0 auto 12px", width: 52, height: 52 }}>
          <i className={`bi ${icon}`} style={{ fontSize: 24 }}></i>
        </div>
        <div className="kpi-value">{value}</div>
        <div className="kpi-label">{label}</div>
      </div>
    </div>
  );
}
