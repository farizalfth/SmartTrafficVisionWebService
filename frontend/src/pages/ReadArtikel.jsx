import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/Reveal";
import { getArticles, imageUrl } from "../lib/firebase";

export default function ReadArtikel() {
  const [artikel, setArtikel] = useState([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    getArticles({ published: 1 }).then(setArtikel);
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return artikel;
    return artikel.filter((a) => (a.judul || "").toLowerCase().includes(q));
  }, [artikel, query]);

  const totalViews = useMemo(
    () => artikel.reduce((s, a) => s + Number(a.views || 0), 0),
    [artikel]
  );
  const latestDate = useMemo(() => {
    if (!artikel.length) return "-";
    return formatDate([...artikel].sort((a, b) => String(b.tanggal).localeCompare(String(a.tanggal)))[0].tanggal);
  }, [artikel]);

  const featured = filtered[0];
  const rest = filtered.slice(1);

  return (
    <div className="container" style={{ paddingBottom: 50 }}>
      <Reveal>
        <div className="page-header">
          <h2 className="page-title gradient-text">Berita & Informasi</h2>
          <p className="page-subtitle">Update terbaru seputar lalu lintas dan teknologi Smart City</p>
        </div>
      </Reveal>

      {/* STATS */}
      <Reveal>
        <div className="row g-3 mb-4">
          <ReadStat icon="bi-journal-richtext" chip="#3B82F6" value={artikel.length} label="Total Artikel" />
          <ReadStat icon="bi-calendar-event" chip="#00E676" value={latestDate} label="Update Terbaru" />
          <ReadStat icon="bi-eye" chip="#FFD600" value={totalViews.toLocaleString("id-ID")} label="Total Dilihat" />
          <ReadStat icon="bi-newspaper" chip="#FF5252" value={artikel.length ? "Aktif" : "Kosong"} label="Status Publikasi" />
        </div>
      </Reveal>

      {/* SEARCH */}
      <Reveal>
        <div className="d-flex justify-content-center mb-4">
          <div className="read-search">
            <i className="bi bi-search"></i>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Cari judul artikel..."
            />
            {query && (
              <button className="read-search-clear" onClick={() => setQuery("")}>
                <i className="bi bi-x-lg"></i>
              </button>
            )}
          </div>
        </div>
      </Reveal>

      {filtered.length === 0 ? (
        <div className="text-center text-muted py-5">
          <i className="bi bi-file-earmark-x" style={{ fontSize: 48 }}></i>
          <p className="mt-3">{artikel.length ? "Artikel tidak ditemukan. Coba kata kunci lain." : "Belum ada artikel yang dipublikasikan."}</p>
        </div>
      ) : (
        <>
          {/* FEATURED */}
          <Reveal>
            <div className="article-featured mb-4">
              <Link to={`/artikel/${featured.id}`} className="article-featured-inner">
                <div className="article-featured-img-wrap">
                  <img src={imageUrl(featured.gambar)} alt="" />
                  <span className="article-featured-badge">
                    <i className="bi bi-stars me-1"></i>Terbaru
                  </span>
                </div>
                <div className="article-featured-content">
                  <span className="badge rounded-pill mb-2" style={{ background: "rgba(59,130,246,0.15)", color: "#3B82F6" }}>
                    <i className="bi bi-newspaper me-1"></i>Artikel Unggulan
                  </span>
                  <h3 className="article-featured-title">{featured.judul}</h3>
                  <p className="text-muted" style={{ fontSize: "0.95rem", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                    {stripHtml(featured.isi).slice(0, 220)}
                  </p>
                  <div className="d-flex flex-wrap gap-2 mt-2">
                    <Chip icon="bi-calendar3" text={formatDate(featured.tanggal)} />
                    <Chip icon="bi-clock" text={readTimeOf(featured.isi)} />
                    <Chip icon="bi-eye" text={`${Number(featured.views || 0).toLocaleString("id-ID")} dilihat`} />
                  </div>
                  <span className="article-featured-cta">
                    Baca Selengkapnya <i className="bi bi-arrow-right"></i>
                  </span>
                </div>
              </Link>
            </div>
          </Reveal>

          {/* GRID */}
          {rest.length > 0 && (
            <div className="row g-4">
              {rest.map((item, i) => (
                <div className="col-lg-4 col-md-6 d-flex align-items-stretch" key={item.key}>
                  <Reveal className="w-100" threshold={0.1}>
                    <Link to={`/artikel/${item.id}`} className="article-card">
                      <div style={{ overflow: "hidden", height: 200 }}>
                        <img src={imageUrl(item.gambar)} alt="" className="article-thumb" style={{ transition: "transform 0.4s" }} />
                      </div>
                      <div style={{ padding: "18px 20px" }}>
                        <div className="article-date mb-2 d-flex align-items-center justify-content-between">
                          <span>
                            <i className="bi bi-calendar3 me-1" style={{ fontSize: 14 }}></i>
                            {formatDate(item.tanggal)}
                          </span>
                          <span style={{ color: "#2563EB", fontSize: 13 }}>
                            <i className="bi bi-eye me-1"></i>{Number(item.views || 0).toLocaleString("id-ID")}
                          </span>
                        </div>
                        <h5 className="article-title" style={{
                          display: "-webkit-box",
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                        }}>{item.judul}</h5>
                        <p className="text-muted mt-2 mb-2" style={{ fontSize: "0.88rem", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                          {stripHtml(item.isi).slice(0, 120)}
                        </p>
                        <div className="d-flex align-items-center justify-content-between">
                          <span style={{ color: "#2563EB", fontWeight: 600, fontSize: "0.9rem" }}>
                            Baca Selengkapnya <i className="bi bi-arrow-right"></i>
                          </span>
                          <span className="read-time-chip">
                            <i className="bi bi-clock"></i> {readTimeOf(item.isi)}
                          </span>
                        </div>
                      </div>
                    </Link>
                  </Reveal>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ReadStat({ icon, chip, value, label }) {
  return (
    <div className="col-md-3 col-6">
      <div className="kpi-card">
        <div className="icon-chip" style={{ "--chip": chip, margin: "0 auto 12px", width: 52, height: 52 }}>
          <i className={`bi ${icon}`} style={{ fontSize: 24 }}></i>
        </div>
        <div className="kpi-value" style={{ fontSize: "1.35rem" }}>{value}</div>
        <div className="kpi-label">{label}</div>
      </div>
    </div>
  );
}

function Chip({ icon, text }) {
  return (
    <span className="badge rounded-pill" style={{ background: "rgba(59,130,246,0.12)", color: "#bfdbfe", padding: "7px 14px", fontWeight: 500, fontSize: "0.8rem" }}>
      <i className={`bi ${icon} me-1`} style={{ color: "#3B82F6" }}></i>{text}
    </span>
  );
}

function stripHtml(html = "") {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent || div.innerText || "";
}

function readTimeOf(isi = "") {
  const words = isi.split(/\s+/).filter(Boolean).length;
  return `${Math.max(1, Math.ceil(words / 200))} mnt`;
}

function formatDate(t) {
  if (!t) return "-";
  const d = new Date(String(t).replace(" ", "T"));
  if (isNaN(d)) return String(t).slice(0, 10);
  return d.toLocaleDateString("id-ID", { day: "2-digit", month: "long", year: "numeric" });
}
