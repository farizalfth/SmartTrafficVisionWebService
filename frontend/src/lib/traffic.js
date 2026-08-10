// Logika agregasi data lalu lintas dari Firebase (port dari app.py)
import { getTrafficStats } from "./firebase";

const pad = (n) => String(n).padStart(2, "0");
const dateKey = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

function nodesMap(stats) {
  // Normalisasi: dict {cctv_id: {...}} atau array
  if (Array.isArray(stats)) {
    const out = {};
    stats.forEach((item, idx) => {
      if (item && typeof item === "object") out[String(idx)] = item;
    });
    return out;
  }
  return stats || {};
}

export async function buildTrafficData(cctvId, period = "harian") {
  const stats = await getTrafficStats();
  const nodes = cctvId ? { [String(cctvId)]: stats[cctvId] || {} } : nodesMap(stats);
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;

  const labels = [];
  const dataMobil = [], dataMotor = [], dataBus = [], dataTruk = [];

  const add = (label, det) => {
    labels.push(label);
    dataMobil.push(det?.mobil || 0);
    dataMotor.push(det?.motor || 0);
    dataBus.push(det?.bus || 0);
    dataTruk.push(det?.truk || 0);
  };

  const sumDetail = (det) => ({ mobil: det?.mobil || 0, motor: det?.motor || 0, bus: det?.bus || 0, truk: det?.truk || 0 });

  if (period === "harian") {
    for (let i = 6; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
      const key = dateKey(d);
      labels.push(d.toLocaleDateString("id-ID", { day: "2-digit", month: "short" }));
      let sum = { mobil: 0, motor: 0, bus: 0, truk: 0 };
      Object.values(nodes).forEach((node) => {
        const det = node?.daily_reports?.[key]?.detail;
        const s = sumDetail(det);
        sum.mobil += s.mobil; sum.motor += s.motor; sum.bus += s.bus; sum.truk += s.truk;
      });
      dataMobil.push(sum.mobil); dataMotor.push(sum.motor); dataBus.push(sum.bus); dataTruk.push(sum.truk);
    }
  } else if (period === "mingguan") {
    const lastDay = new Date(year, month, 0).getDate();
    const ranges = [[1, 7], [8, 14], [15, 21], [22, lastDay]];
    const monthShort = now.toLocaleDateString("id-ID", { month: "short" });
    ranges.forEach(([start, end], i) => {
      let sum = { mobil: 0, motor: 0, bus: 0, truk: 0 };
      Object.values(nodes).forEach((node) => {
        const daily = node?.daily_reports || {};
        for (let day = start; day <= end; day++) {
          const det = daily[`${year}-${pad(month)}-${pad(day)}`]?.detail;
          const s = sumDetail(det);
          sum.mobil += s.mobil; sum.motor += s.motor; sum.bus += s.bus; sum.truk += s.truk;
        }
      });
      labels.push([`Minggu ${i + 1}`, `${start}-${end} ${monthShort}`]);
      dataMobil.push(sum.mobil); dataMotor.push(sum.motor); dataBus.push(sum.bus); dataTruk.push(sum.truk);
    });
  } else if (period === "bulanan") {
    const monthList = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"];
    monthList.forEach((m, i) => {
      const prefix = `${year}-${pad(i + 1)}`;
      let sum = { mobil: 0, motor: 0, bus: 0, truk: 0 };
      Object.values(nodes).forEach((node) => {
        Object.entries(node?.daily_reports || {}).forEach(([dk, val]) => {
          if (dk.startsWith(prefix)) {
            const s = sumDetail(val?.detail);
            sum.mobil += s.mobil; sum.motor += s.motor; sum.bus += s.bus; sum.truk += s.truk;
          }
        });
      });
      labels.push(m);
      dataMobil.push(sum.mobil); dataMotor.push(sum.motor); dataBus.push(sum.bus); dataTruk.push(sum.truk);
    });
  }

  return { labels, datasets: { mobil: dataMobil, motor: dataMotor, bus: dataBus, truk: dataTruk } };
}

// Ringkasan data teks per periode (harian / mingguan / bulanan).
// Menghasilkan baris: { label, total, mobil, motor, bus, truk, kepadatan, status }.
export async function buildTextSummaries(cctvId, period = "harian") {
  const stats = await getTrafficStats();
  const nodes = cctvId ? { [String(cctvId)]: stats[cctvId] || {} } : nodesMap(stats);
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;

  const summarize = (filter) => {
    let total = 0;
    const sum = { mobil: 0, motor: 0, bus: 0, truk: 0 };
    let kepSum = 0, kepN = 0;
    const statusCount = {};
    Object.values(nodes).forEach((node) => {
      Object.entries(node?.daily_reports || {}).forEach(([dk, rep]) => {
        if (rep && typeof rep === "object" && filter(dk, rep)) {
          const d = rep?.detail || {};
          sum.mobil += d.mobil || 0;
          sum.motor += d.motor || 0;
          sum.bus += d.bus || 0;
          sum.truk += d.truk || 0;
          total += rep?.total_hari_ini || 0;
          if (rep?.kepadatan_terakhir_persen != null) {
            kepSum += Number(rep.kepadatan_terakhir_persen);
            kepN += 1;
          }
          if (rep?.status_terakhir) statusCount[rep.status_terakhir] = (statusCount[rep.status_terakhir] || 0) + 1;
        }
      });
    });
    const kepadatan = kepN ? Math.round(kepSum / kepN) : null;
    let status = null;
    const statusKeys = Object.keys(statusCount);
    if (statusKeys.length) {
      status = statusKeys.sort((a, b) => statusCount[b] - statusCount[a])[0];
    } else if (kepadatan != null) {
      status = kepadatan < 30 ? "Lancar" : kepadatan < 55 ? "Padat" : "Macet";
    }
    return { total, ...sum, kepadatan, status };
  };

  const rows = [];
  if (period === "harian") {
    for (let i = 6; i >= 0; i--) {
      const d = new Date(year, month - 1, now.getDate() - i);
      const key = dateKey(d);
      rows.push({
        label: d.toLocaleDateString("id-ID", { weekday: "long", day: "2-digit", month: "short" }),
        ...summarize((dk) => dk === key),
      });
    }
  } else if (period === "mingguan") {
    const lastDay = new Date(year, month, 0).getDate();
    const ranges = [[1, 7], [8, 14], [15, 21], [22, lastDay]];
    const monthShort = now.toLocaleDateString("id-ID", { month: "short" });
    ranges.forEach(([start, end], i) => {
      rows.push({
        label: `Minggu ${i + 1} (${start}-${end} ${monthShort})`,
        ...summarize((dk) => dk.startsWith(`${year}-${pad(month)}-`) && Number(dk.slice(8, 10)) >= start && Number(dk.slice(8, 10)) <= end),
      });
    });
  } else {
    const monthList = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"];
    monthList.forEach((m, i) => {
      const prefix = `${year}-${pad(i + 1)}`;
      rows.push({ label: m, ...summarize((dk) => dk.startsWith(prefix)) });
    });
  }
  return rows;
}

export async function getVehicleDistribution(cctvId) {
  const stats = await getTrafficStats();
  const nodes = cctvId ? { [String(cctvId)]: stats[cctvId] || {} } : nodesMap(stats);
  const today = dateKey(new Date());
  const counts = { mobil: 0, motor: 0, bus: 0, truk: 0 };

  Object.values(nodes).forEach((node) => {
    const det = node?.daily_reports?.[today]?.detail || {};
    counts.mobil += det.mobil || 0;
    counts.motor += det.motor || 0;
    counts.bus += det.bus || 0;
    counts.truk += det.truk || 0;
  });

  const values = [counts.mobil, counts.motor, counts.bus, counts.truk];
  const total = values.reduce((a, b) => a + b, 0);
  const percentages = values.map((v) => (total > 0 ? `${((v / total) * 100).toFixed(1)}%` : "0%"));
  return { labels: ["Mobil", "Motor", "Bus", "Truk"], data: values, percentages };
}

export async function getSummary(cctvId, cctvCount) {
  const stats = await getTrafficStats();
  const today = dateKey(new Date());
  const empty = { kendaraan_hari_ini: 0, kepadatan_tertinggi: 0, rata_rata_kecepatan: "—", status: "Lancar", kamera_aktif: cctvCount };

  if (!cctvId) {
    // Agregasi semua CCTV
    const nodes = nodesMap(stats);
    let total = 0, sumKep = 0, camCount = 0, speedSum = 0, speedN = 0;
    Object.values(nodes).forEach((node) => {
      const live = node?.live || {};
      const daily = node?.daily_reports?.[today] || {};
      total += daily?.total_hari_ini || live?.total_akumulasi_hari_ini || 0;
      sumKep += live?.occupancy_persen || live?.kepadatan_persen || 0;
      if (live?.kecepatan_kmh != null) {
        speedSum += live.kecepatan_kmh;
        speedN += 1;
      }
      camCount += 1;
    });
    const avg = camCount ? Math.round(sumKep / camCount) : 0;
    const status = avg < 30 ? "Lancar" : avg < 55 ? "Padat" : "Macet";
    const kecepatan = speedN ? Math.round(speedSum / speedN) : null;
    return { kendaraan_hari_ini: total, kepadatan_tertinggi: avg, rata_rata_kecepatan: kecepatan != null ? `${kecepatan} km/j` : "—", status, kamera_aktif: cctvCount };
  }

  const node = stats[cctvId] || {};
  const live = node?.live || {};
  const daily = node?.daily_reports?.[today] || {};
  const totalHariIni = daily?.total_hari_ini || live?.total_akumulasi_hari_ini || 0;
  const kepadatan = Math.round(live?.occupancy_persen ?? live?.kepadatan_persen ?? 0);
  const status = live?.status || (kepadatan < 30 ? "Lancar" : kepadatan < 55 ? "Padat" : "Macet");
  const kecepatan = live?.kecepatan_kmh ?? null;
  return { kendaraan_hari_ini: totalHariIni, kepadatan_tertinggi: kepadatan, rata_rata_kecepatan: kecepatan != null ? `${kecepatan} km/j` : "—", status, kamera_aktif: cctvCount };
}

// Berapa lama data live dianggap masih segar (ms).
// AI server menulis ulang setiap ~10 detik, jadi 90 detik tanpa update = tidak ada data.
const STALE_MS = 90 * 1000;

export function isLiveFresh(live) {
  if (!live || typeof live !== "object") return false;
  if (typeof live.last_update_ts === "number") {
    return Date.now() - live.last_update_ts * 1000 < STALE_MS;
  }
  if (!live.last_update) return false;
  const d = new Date(String(live.last_update).replace(" ", "T"));
  if (isNaN(d.getTime())) return false;
  return Date.now() - d.getTime() < STALE_MS;
}

// Status yang benar-benar terukur. Jika tidak ada deteksi segar,
// jangan mengarang "Lancar" — kembalikan "Tidak Ada Data".
export function effectiveStatus(live) {
  if (!isLiveFresh(live) || !live.status) return "Tidak Ada Data";
  return live.status;
}

export function statusColor(status) {
  if (status === "Lancar") return "#00C853";
  if (status === "Padat") return "#FFD600";
  if (status === "Macet") return "#FF5252";
  return "#9aa3b2";
}

// ===== Analisis sentimen komentar (Baik / Netral / Buruk) =====
const KATA_POSITIF = ["baik", "bagus", "mantap", "keren", "membantu", "lancar", "puas", "terima kasih", "terimakasih", "hebat", "luar biasa", "bermanfaat", "akurat", "informatif", "canggih", "oke", "ok", "sip", "jos", "top", "solusi", "cepat", "responsif", "terbantu", "senang", "update", "recommended", "sangat baik", "sangat membantu"];
const KATA_NEGATIF = ["buruk", "jelek", "macet", "parah", "lambat", "kecewa", "salah", "error", "rusak", "lemot", "payah", "ribet", "sulit", "susah", "lag", "gagal", "tidak membantu", "tidak berguna", "kurang", "kacau", "berantakan", "penipu", "bohong", "sampah", "patah", "tidak akurat", "bukan solusi", "mati", "down", "bug", "parah banget", "benci", "jengkel", "menyesal", "lambat sekali"];

export function classifySentiment(text) {
  const t = String(text || "").toLowerCase();
  let positif = 0, negatif = 0;
  KATA_NEGATIF.forEach((k) => { if (t.includes(k)) negatif += 1; });
  KATA_POSITIF.forEach((k) => {
    if (t.includes(k)) {
      if (/tidak |kurang |bukan |gak|ga |jangan |engga/.test(t)) negatif += 1.5;
      else positif += 1;
    }
  });
  if (positif > negatif) return "Baik";
  if (negatif > positif) return "Buruk";
  return "Netral";
}

export function sentimentColor(s) {
  if (s === "Baik") return "#00C853";
  if (s === "Buruk") return "#FF5252";
  return "#94A3B8";
}
