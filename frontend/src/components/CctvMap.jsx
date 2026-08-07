import { useEffect, useRef } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { Link } from "react-router-dom";
import { statusColor } from "../lib/traffic";

function makeIcon(status) {
  const color = statusColor(status);
  return L.divIcon({
    className: "",
    html: `<div class="cctv-marker"><span class="cctv-dot" style="background:${color};box-shadow:0 0 12px ${color}"></span></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -14],
  });
}

function FitBounds({ markers }) {
  const map = useMap();
  const done = useRef(false);
  useEffect(() => {
    if (!markers.length || done.current) return;
    const valid = markers.filter((m) => m.lat && m.lon);
    if (!valid.length) return;
    const bounds = L.featureGroup(
      valid.map((m) => L.marker([m.lat, m.lon]))
    ).getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50] });
      done.current = true;
    }
  }, [markers, map]);
  return null;
}

export default function CctvMap({ cctv = [], liveMap = {}, height = "550px", zoom = 9 }) {
  return (
    <MapContainer
      center={[-2.5489, 118.0149]}
      zoom={zoom}
      scrollWheelZoom
      style={{ width: "100%", height }}
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {cctv
        .filter((c) => c.lat && c.lon)
        .map((c) => {
          const live = liveMap[c.id] || {};
          const status = live.status || "Lancar";
          const total = live.total ?? c.current_total ?? 0;
          return (
            <Marker key={c.id} position={[c.lat, c.lon]} icon={makeIcon(status)}>
              <Popup>
                <div style={{ textAlign: "center", fontFamily: "Poppins, sans-serif" }}>
                  <b style={{ color: "#fff" }}>{c.name}</b>
                  <br />
                  <span style={{ color: statusColor(status), fontWeight: 600 }}>● {status}</span>
                  <span style={{ color: "#bfdbfe" }}> · {total} kendaraan</span>
                  <br />
                  <Link to="/dashboard" style={{ color: "#2563EB", fontWeight: 600 }}>
                    Lihat Live →
                  </Link>
                </div>
              </Popup>
            </Marker>
          );
        })}
      <FitBounds markers={cctv} />
    </MapContainer>
  );
}
