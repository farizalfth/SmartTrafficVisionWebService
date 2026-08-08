const ICONS = {
  car: "bi-car-front",
  activity: "bi-activity",
  gauge: "bi-speedometer2",
  camera: "bi-camera-video",
  cctv: "bi-camera-reels",
  radio: "bi-broadcast",
  cone: "bi-cone-striped",
  alert: "bi-exclamation-triangle",
  flame: "bi-fire",
  chart: "bi-bar-chart",
};

// Card ringkasan KPI dengan ikon (bootstrap-icons)
export default function SummaryCard({ icon = "activity", chip = "#3B82F6", value = "-", label = "", sub = "" }) {
  return (
    <div className="summary-card">
      <div className="icon-chip" style={{ "--chip": chip }}>
        <i className={`bi ${ICONS[icon] || ICONS.activity}`} style={{ fontSize: 28 }}></i>
      </div>
      <div className="value">{value}</div>
      <div className="label">{label}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}
