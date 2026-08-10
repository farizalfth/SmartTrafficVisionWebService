import { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  BarElement,
  ArcElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";

ChartJS.register(BarElement, ArcElement, CategoryScale, LinearScale, Legend, Tooltip);

ChartJS.defaults.font.family = "'Poppins', sans-serif";
ChartJS.defaults.color = "#e0e0e0";

const tooltipStyle = {
  backgroundColor: "rgba(15,15,15,0.95)",
  titleColor: "#fff",
  bodyColor: "#e0e0e0",
  borderColor: "rgba(59,130,246,0.35)",
  borderWidth: 1,
  padding: 12,
  cornerRadius: 8,
  boxPadding: 4,
  usePointStyle: true,
};

const legendStyle = {
  display: true,
  labels: {
    color: "#fff",
    font: { size: 12, weight: "600", family: "'Poppins', sans-serif" },
    usePointStyle: true,
    pointStyle: "circle",
    boxWidth: 8,
    padding: 20,
  },
};

function barGradient(context, topColor, bottomColor) {
  const { ctx, chartArea } = context.chart;
  if (!chartArea) return topColor;
  const g = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
  g.addColorStop(0, bottomColor);
  g.addColorStop(1, topColor);
  return g;
}

const DATASET_COLORS = {
  mobil: { top: "#7CB9FF", bottom: "rgba(37,99,235,0.15)", border: "#2563EB", hover: "#3B82F6" },
  motor: { top: "#FFE45E", bottom: "rgba(255,196,0,0.15)", border: "#FFC400", hover: "#FFD600" },
  bus: { top: "#5CF3B2", bottom: "rgba(0,168,107,0.15)", border: "#00A86B", hover: "#00E676" },
  truk: { top: "#FF7A70", bottom: "rgba(229,57,53,0.15)", border: "#E53935", hover: "#FF5252" },
};

function makeLabels(labels) {
  // Label sumbu X dibuat bersih satu baris; rincian kendaraan pindah ke tooltip.
  return labels.map((label) =>
    Array.isArray(label) ? label[0] : label
  );
}

function computeMax(labels, datasets) {
  let maxVal = 0;
  labels.forEach((_, i) => {
    const total =
      (datasets.mobil?.[i] || 0) +
      (datasets.motor?.[i] || 0) +
      (datasets.bus?.[i] || 0) +
      (datasets.truk?.[i] || 0);
    if (total > maxVal) maxVal = total;
  });
  return maxVal > 0 ? Math.ceil((maxVal * 1.25) / 100) * 100 : 500;
}

export function TrafficBarChart({ labels = [], datasets = {}, _period = "harian", height = 650 }) {
  const [vw, setVw] = useState(typeof window !== "undefined" ? window.innerWidth : 1280);
  useEffect(() => {
    const onResize = () => setVw(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const isSmall = vw <= 576;
  const chartHeight = isSmall ? Math.max(height, 420) : height;
  const xFont = isSmall
    ? labels.length > 6 ? 9 : 10
    : labels.length > 10 ? 10 : labels.length > 6 ? 11 : 12;

  const data = {
    labels: makeLabels(labels),
    datasets: ["mobil", "motor", "bus", "truk"].map((k) => {
      const c = DATASET_COLORS[k];
      return {
        label: k.charAt(0).toUpperCase() + k.slice(1),
        data: datasets[k] || [],
        barThickness: labels.length > 10 ? 16 : labels.length > 6 ? 22 : 35,
        backgroundColor: (ctx) => barGradient(ctx, c.top, c.bottom),
        borderColor: c.border,
        borderWidth: 1.5,
        borderRadius: 8,
        hoverBackgroundColor: c.hover,
      };
    }),
  };

  return (
    <div style={{ height: `${chartHeight}px`, position: "relative", width: "100%" }}>
      <Bar
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { top: 12, right: 10, bottom: 8, left: 6 } },
          scales: {
            x: {
              stacked: true,
              ticks: {
                color: "#fff",
                font: { size: xFont },
                maxRotation: 0,
                minRotation: 0,
                autoSkip: false,
                padding: 8,
              },
              grid: { display: false },
              border: { display: false },
            },
            y: {
              stacked: true,
              beginAtZero: true,
              min: 0,
              max: computeMax(labels, datasets),
              ticks: {
                color: "#94a3b8",
                callback: (v) => Number(v).toLocaleString("id-ID"),
                maxTicksLimit: 5,
                padding: 10,
              },
              grid: { color: "rgba(255,255,255,0.06)", drawTicks: false },
              border: { display: false },
            },
          },
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                color: "#fff",
                font: { size: 11, weight: "600", family: "'Poppins', sans-serif" },
                usePointStyle: true,
                pointStyle: "circle",
                boxWidth: 8,
                padding: 14,
              },
            },
            tooltip: {
              ...tooltipStyle,
              callbacks: {
                title: (items) => {
                  const label = items[0]?.label || "";
                  const total = items.reduce((s, i) => s + (Number(i.parsed.y) || 0), 0);
                  return `${label} — Total ${total.toLocaleString("id-ID")}`;
                },
                label: (ctx) => ` ${ctx.dataset.label} : ${Number(ctx.parsed.y).toLocaleString("id-ID")}`,
                footer: () => undefined,
              },
            },
          },
        }}
      />
    </div>
  );
}

// Plugin teks total di tengah doughnut
const doughnutCenterText = {
  id: "doughnutCenterText",
  afterDraw(chart) {
    if (chart.config.type !== "doughnut") return;
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    if (!meta.data.length) return;
    const raw = chart.data.datasets[0].data;
    const total = raw.reduce((a, b) => (Number(a) || 0) + (Number(b) || 0), 0);
    const x = (chart.chartArea.left + chart.chartArea.right) / 2;
    const y = (chart.chartArea.top + chart.chartArea.bottom) / 2;
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = total > 0 ? "#fff" : "#555";
    ctx.font = "700 26px Poppins, sans-serif";
    ctx.fillText(total > 0 ? total.toLocaleString("id-ID") : "0", x, y - 8);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px Poppins, sans-serif";
    ctx.fillText("KENDARAAN", x, y + 16);
    ctx.restore();
  },
};

const DOUGHNUT_COLORS = ["#7CB9FF", "#FFE45E", "#5CF3B2", "#FF7A70"];

export function VehicleDoughnut({ data = [], size = 280 }) {
  const hasData = data.some((v) => v > 0);
  const chartData = {
    labels: ["Mobil", "Motor", "Bus", "Truk"],
    datasets: [
      {
        data: hasData ? data : [1],
        backgroundColor: hasData ? DOUGHNUT_COLORS : ["#333"],
        borderColor: "#101010",
        borderWidth: 3,
        hoverOffset: 10,
      },
    ],
  };

  return (
    <div style={{ height: `${size}px`, width: `${size}px`, position: "relative" }}>
      <Doughnut
        data={chartData}
        plugins={[doughnutCenterText]}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          cutout: "72%",
          plugins: { legend: { display: false }, tooltip: { ...tooltipStyle, enabled: hasData } },
          animation: {
            animateRotate: true,
            animateScale: true,
            duration: 1200,
            easing: "easeOutQuart",
          },
        }}
      />
    </div>
  );
}

const COMMENT_COLORS = {
  baik: { top: "#5CF3B2", bottom: "rgba(0,168,107,0.16)", border: "#00C853", hover: "#00E676" },
  netral: { top: "#E2E8F0", bottom: "rgba(148,163,184,0.16)", border: "#94A3B8", hover: "#CBD5E1" },
  buruk: { top: "#FF8A80", bottom: "rgba(229,57,53,0.16)", border: "#FF5252", hover: "#FF6E6E" },
};

function commentDataset(label, data, color, barPercentage, categoryPercentage) {
  return {
    label,
    data,
    backgroundColor: (ctx) => barGradient(ctx, color.top, color.bottom),
    borderColor: color.border,
    borderWidth: 1.5,
    borderRadius: { topLeft: 7, topRight: 7, bottomLeft: 7, bottomRight: 7 },
    borderSkipped: false,
    hoverBackgroundColor: color.hover,
    hoverBorderColor: color.hover,
    maxBarThickness: 32,
    barPercentage,
    categoryPercentage,
  };
}

export function CommentBarChart({ labels = [], baik = [], netral = [], buruk = [], total = 0 }) {
  const [vw, setVw] = useState(typeof window !== "undefined" ? window.innerWidth : 1280);
  useEffect(() => {
    const onResize = () => setVw(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const isSmall = vw <= 576;
  const hasData = labels.length > 0 && total > 0;

  const multiLabels = labels.map((label, i) => [
    label,
    "──────",
    `Baik : ${baik[i] || 0}`,
    `Netral : ${netral[i] || 0}`,
    `Buruk : ${buruk[i] || 0}`,
    `Total : ${(baik[i] || 0) + (netral[i] || 0) + (buruk[i] || 0)}`,
  ]);

  const data = {
    labels: multiLabels,
    datasets: [
      commentDataset("Puas (Baik)", baik, COMMENT_COLORS.baik, isSmall ? 0.55 : 0.6, isSmall ? 0.45 : 0.5),
      commentDataset("Netral", netral, COMMENT_COLORS.netral, isSmall ? 0.55 : 0.6, isSmall ? 0.45 : 0.5),
      commentDataset("Laporan (Buruk)", buruk, COMMENT_COLORS.buruk, isSmall ? 0.55 : 0.6, isSmall ? 0.45 : 0.5),
    ],
  };

  return (
    <div style={{ height: isSmall ? 420 : 300, position: "relative", width: "100%" }}>
      <Bar
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { top: 10, right: 6, bottom: 6, left: 4 } },
          animation: {
            duration: 800,
            easing: "easeOutQuart",
            animateScale: true,
          },
          scales: {
            x: {
              stacked: false,
              ticks: {
                color: "#fff",
                font: { size: isSmall ? 8 : 11, weight: "600" },
                lineHeight: isSmall ? 1.4 : 1.5,
                padding: 6,
                maxRotation: 0,
                minRotation: 0,
                autoSkip: false,
              },
              grid: { display: false },
              border: { display: false },
            },
            y: {
              beginAtZero: true,
              grace: "5%",
              ticks: {
                color: "#94a3b8",
                precision: 0,
                maxTicksLimit: 6,
                font: { size: isSmall ? 10 : 12 },
                padding: 8,
                callback: (v) => Number(v).toLocaleString("id-ID"),
              },
              grid: { color: "rgba(255,255,255,0.05)", drawTicks: false },
              border: { display: false },
            },
          },
          plugins: {
            legend: isSmall
              ? {
                  position: "bottom",
                  labels: {
                    color: "#fff",
                    font: { size: 9, weight: "600", family: "'Poppins', sans-serif" },
                    usePointStyle: true,
                    pointStyle: "circle",
                    boxWidth: 6,
                    padding: 8,
                  },
                }
              : {
                  ...legendStyle,
                  position: "bottom",
                },
            tooltip: {
              ...tooltipStyle,
              callbacks: {
                title: (items) => {
                  const i = items[0]?.dataIndex;
                  const sum = (baik[i] || 0) + (netral[i] || 0) + (buruk[i] || 0);
                  return `${labels[i] || ""} — ${sum} ulasan`;
                },
                label: (ctx) => ` ${ctx.dataset.label} : ${ctx.parsed.y}`,
                footer: (items) => {
                  const i = items[0]?.dataIndex;
                  const sum = (baik[i] || 0) + (netral[i] || 0) + (buruk[i] || 0);
                  return `Total ${sum} ulasan masuk`;
                },
                labelTextColor: () => "#e0e0e0",
              },
            },
          },
        }}
      />
      {!hasData && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(15,15,15,0.4)",
            borderRadius: 14,
          }}
        >
          <i className="bi bi-inbox" style={{ fontSize: 34, color: "#94a3b8", opacity: 0.7 }}></i>
          <p className="mb-0 mt-2 small text-muted">Belum ada ulasan pada periode ini.</p>
        </div>
      )}
    </div>
  );
}
