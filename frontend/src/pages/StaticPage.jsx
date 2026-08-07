import { useEffect, useState, useCallback } from "react";
import Reveal from "../components/Reveal";
import { TrafficBarChart } from "../components/Charts";
import { getCctvList } from "../lib/firebase";
import { buildTrafficData } from "../lib/traffic";

const PERIOD_INFO = {
  harian: "7 Hari Terakhir",
  mingguan: "Mingguan (Bulan Berjalan)",
  bulanan: "12 Bulan",
};

export default function StaticPage() {
  const [cctv, setCctv] = useState([]);
  const [cctvId, setCctvId] = useState("");
  const [period, setPeriod] = useState("harian");
  const [traffic, setTraffic] = useState({ labels: [], datasets: {} });
  const [kpi, setKpi] = useState({ total: 0, avg: 0, peak: "-", topCat: "-" });
  const [time, setTime] = useState("");

  useEffect(() => {
    getCctvList().then(setCctv);
  }, []);

  const refresh = useCallback(async () => {
    if (!cctvId) return;
    const data = await buildTrafficData(cctvId === "all" ? null : cctvId, period);
    setTraffic(data);
    setTime(new Date().toLocaleTimeString("id-ID", { hour12: false }));

    // Hitung KPI
    const keys = ["mobil", "motor", "bus", "truk"];
    let grandTotal = 0;
    const perLabel = data.labels.map((_, i) => {
      const total = keys.reduce((s, k) => s + (data.datasets[k]?.[i] || 0), 0);
      grandTotal += total;
      return total;
    });
    const avg = data.labels.length ? Math.round(grandTotal / data.labels.length) : 0;
    const peakIdx = perLabel.indexOf(Math.max(...perLabel));
    const peakLabel = peakIdx >= 0 ? String(data.labels[peakIdx]).split(",")[0] : "-";
    let topCatName = "-", topCatVal = 0;
    keys.forEach((k) => {
      const v = (data.datasets[k] || []).reduce((a, b) => a + (b || 0), 0);
      if (v > topCatVal) { topCatVal = v; topCatName = k.charAt(0).toUpperCase() + k.slice(1); }
    });
    setKpi({
      total: grandTotal,
      avg,
      peak: grandTotal > 0 ? peakLabel : "-",
      topCat: grandTotal > 0 ? `${topCatName} (${topCatVal.toLocaleString("id-ID")})` : "-",
    });
  }, [cctvId, period]);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, [refresh]);

  const selText = cctvId === "all" ? "Semua Data CCTV" : cctv.find((c) => String(c.id) === String(cctvId))?.name || "";

  return (
    <div className="container" style={{ paddingBottom: 50 }}>
      <Reveal>
        <div className="page-header">
          <h2 className="page-title gradient-text">Analitik Lalu Lintas</h2>
          <p className="page-subtitle">Pilih lokasi CCTV untuk melihat tren kepadatan lalu lintas.</p>
        </div>
      </Reveal>

      <div className="row justify-content-center mb-4">
        <div className="col-md-6">
          <select
            className="form-select text-center py-2"
            value={cctvId}
            onChange={(e) => setCctvId(e.target.value)}
          >
            <option value="" disabled>Pilih CCTV</option>
            <option value="all">Semua Data CCTV</option>
            {cctv.map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="d-flex justify-content-center mb-4">
        <div className="period-toggle">
          {["harian", "mingguan", "bulanan"].map((p) => (
            <button key={p} className={period === p ? "active" : ""} onClick={() => setPeriod(p)}>
              {p === "harian" ? "Harian" : p === "mingguan" ? "Mingguan" : "Bulanan"}
            </button>
          ))}
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-3 col-6"><KpiCard icon="bi-car-front" chip="#3B82F6" value={kpi.total.toLocaleString("id-ID")} label="Total Kendaraan" large={kpi.total >= 100000} /></div>
        <div className="col-md-3 col-6"><KpiCard icon="bi-speedometer2" chip="#FFD600" value={kpi.avg.toLocaleString("id-ID")} label="Rata-rata / Titik" /></div>
        <div className="col-md-3 col-6"><KpiCard icon="bi-fire" chip="#FF5252" value={kpi.peak} label="Periode Terpadat" /></div>
        <div className="col-md-3 col-6"><KpiCard icon="bi-bar-chart" chip="#00E676" value={kpi.topCat} label="Kendaraan Terbanyak" /></div>
      </div>

      <div className="row justify-content-center">
        <div className="col-lg-10">
          <div className="chart-card">
            <h4 className="text-center" style={{ color: "#fff" }}>{titleFor(period)}</h4>
            <p className="text-secondary text-center">
              {selText} • {PERIOD_INFO[period]} • Update {time}
            </p>
            <TrafficBarChart labels={traffic.labels} datasets={traffic.datasets} period={period} height={460} />
          </div>
        </div>
      </div>

      <div className="row justify-content-center">
        <div className="col-lg-10">
          <div className="chart-card position-relative">
            <span className="badge rounded-pill text-dark position-absolute" style={{ top: 30, right: 30, background: "#0dcaf0" }}>
              REAL TIME
            </span>
            <h4 className="mb-4 text-center" style={{ color: "#fff" }}>Tabel Detail Kendaraan</h4>
            <div className="table-responsive" style={{ maxHeight: 420, overflowY: "auto" }}>
              <table className="table table-dark table-striped align-middle mb-0">
                <thead>
                  <tr>
                    <th>Periode</th><th className="text-end">Mobil</th><th className="text-end">Motor</th>
                    <th className="text-end">Bus</th><th className="text-end">Truk</th><th className="text-end">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {traffic.labels.map((lbl, i) => {
                    const label = Array.isArray(lbl) ? lbl.join(" · ") : lbl;
                    const vals = ["mobil", "motor", "bus", "truk"].map((k) => traffic.datasets[k]?.[i] || 0);
                    const total = vals.reduce((a, b) => a + b, 0);
                    return (
                      <tr key={i}>
                        <td>{label}</td>
                        {vals.map((v, j) => <td key={j} className="text-end">{v.toLocaleString("id-ID")}</td>)}
                        <td className="text-end fw-bold">{total.toLocaleString("id-ID")}</td>
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
                      <td className="text-end fw-bold text-info">
                        {(traffic.datasets.mobil || []).reduce((a, b, i) => a + (b || 0) + (traffic.datasets.motor?.[i] || 0) + (traffic.datasets.bus?.[i] || 0) + (traffic.datasets.truk?.[i] || 0), 0).toLocaleString("id-ID")}
                      </td>
                    </tr>
                  </tfoot>
                )}
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function titleFor(period) {
  if (period === "harian") return "Laporan Kendaraan 7 Hari Terakhir";
  if (period === "mingguan") return "Laporan Kendaraan Mingguan";
  return "Laporan Kendaraan 12 Bulan";
}

function KpiCard({ icon, chip, value, label, large }) {
  return (
    <div className="kpi-card">
      <div className="icon-chip" style={{ "--chip": chip, margin: "0 auto 12px", width: 52, height: 52 }}>
        <i className={`bi ${icon}`} style={{ fontSize: 24 }}></i>
      </div>
      <div className={`kpi-value ${large ? "is-large" : ""}`}>{value}</div>
      <div className="kpi-label">{label}</div>
    </div>
  );
}
