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
  return maxVal > 0 ? Math.ceil((maxVal * 1.2) / 100) * 100 : 500;
}

export function TrafficBarChart({ labels = [], datasets = {}, _period = "harian", height = 650 }) {
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

  const xFont = labels.length > 10 ? 10 : labels.length > 6 ? 11 : 12;

  return (
    <div style={{ height: `${height}px`, position: "relative", width: "100%" }}>
      <Bar
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { top: 10, right: 8, bottom: 4, left: 4 } },
          scales: {
            x: {
              stacked: true,
              ticks: {
                color: "#fff",
                font: { size: xFont },
                maxRotation: 0,
                minRotation: 0,
                autoSkip: false,
                padding: 6,
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
                maxTicksLimit: 6,
                padding: 6,
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

export function CommentBarChart({ labels = [], baik = [], netral = [], buruk = [] }) {
  const multiLabels = labels.map((label, i) => [
    label,
    "──────────",
    `Baik : ${baik[i] || 0}`,
    `Netral : ${netral[i] || 0}`,
    `Buruk : ${buruk[i] || 0}`,
    `Total : ${(baik[i] || 0) + (netral[i] || 0) + (buruk[i] || 0)}`,
  ]);

  const data = {
    labels: multiLabels,
    datasets: [
      {
        label: "Puas (Baik)",
        data: baik,
        backgroundColor: "rgba(59,130,246,0.7)",
        borderColor: "#3B82F6",
        borderWidth: 1.5,
        borderRadius: 6,
        barPercentage: 0.6,
        categoryPercentage: 0.5,
      },
      {
        label: "Netral",
        data: netral,
        backgroundColor: "rgba(148,163,184,0.7)",
        borderColor: "#94A3B8",
        borderWidth: 1.5,
        borderRadius: 6,
        barPercentage: 0.6,
        categoryPercentage: 0.5,
      },
      {
        label: "Laporan (Buruk)",
        data: buruk,
        backgroundColor: "rgba(255,82,82,0.7)",
        borderColor: "#FF5252",
        borderWidth: 1.5,
        borderRadius: 6,
        barPercentage: 0.6,
        categoryPercentage: 0.5,
      },
    ],
  };

  return (
    <div style={{ height: "300px", position: "relative", width: "100%" }}>
      <Bar
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              stacked: false,
              ticks: { color: "#fff", font: { size: 11 }, lineHeight: 1.5 },
              grid: { display: false },
            },
            y: {
              beginAtZero: true,
              ticks: { color: "#94a3b8", stepSize: 1 },
              grid: { color: "rgba(255,255,255,0.05)" },
            },
          },
          plugins: {
            legend: legendStyle,
            tooltip: {
              ...tooltipStyle,
              callbacks: {
                label: (ctx) => ` ${ctx.dataset.label} : ${ctx.parsed.y}`,
              },
            },
          },
        }}
      />
    </div>
  );
}
