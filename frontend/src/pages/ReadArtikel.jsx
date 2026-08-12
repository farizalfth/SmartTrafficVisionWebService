import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/Reveal";
import { getArticles, imageUrl } from "../lib/firebase";

const WEB_PAGE1 = 7;       // halaman 1: 1 banner + 6 artikel
const WEB_PER_PAGE = 6;    // halaman 2 dst: 6 artikel
const MOBILE_PER_SLIDE = 3;

export default function ReadArtikel() {
  const [artikel, setArtikel] = useState([]);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [slideIdx, setSlideIdx] = useState(0);
  const sliderRef = useRef(null);

  useEffect(() => {
    getArticles({ published: 1 }).then(setArtikel);
  }, []);

  useEffect(() => {
    setPage(1);
    setSlideIdx(0);
  }, [query]);

  // Urutan: tanggal terbaru di depan. Artikel pertama otomatis menjadi banner/artikel
  // unggulan saat ada inputan artikel baru (tidak dikunci ID tertentu).
  const ordered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? artikel.filter((a) => (a.judul || "").toLowerCase().includes(q))
      : artikel;
    return [...list].sort((a, b) => String(b.tanggal).localeCompare(String(a.tanggal)));
  }, [artikel, query]);

  const totalViews = useMemo(
    () => artikel.reduce((s, a) => s + Number(a.views || 0), 0),
    [artikel]
  );
  const latestDate = useMemo(() => {
    if (!artikel.length) return "-";
    return formatDate([...artikel].sort((a, b) => String(b.tanggal).localeCompare(String(a.tanggal)))[0].tanggal);
  }, [artikel]);

  // ---- Web: pagination (halaman 1 = 7 artikel, halaman 2 dst = 6) ----
  const webTotalPages = useMemo(() => {
    const len = ordered.length;
    return len === 0 ? 0 : 1 + Math.max(0, Math.ceil((len - WEB_PAGE1) / WEB_PER_PAGE));
  }, [ordered]);

  const webPage = useMemo(() => {
    if (page === 1) {
      return { banner: ordered[0] || null, grid: ordered.slice(1, WEB_PAGE1) };
    }
    const start = WEB_PAGE1 + (page - 2) * WEB_PER_PAGE;
    return { banner: null, grid: ordered.slice(start, start + WEB_PER_PAGE) };
  }, [ordered, page]);

  const safePage = Math.min(page, webTotalPages || 1);

  // ---- Mobile: slider (slide 1 = 1 banner + 3 artikel, slide 2 dst = 3) ----
  const mobileSlides = useMemo(() => {
    const out = [];
    if (ordered.length) out.push({ banner: ordered[0], items: ordered.slice(1, 1 + MOBILE_PER_SLIDE) });
    for (let i = 1 + MOBILE_PER_SLIDE; i < ordered.length; i += MOBILE_PER_SLIDE) {
      out.push({ banner: null, items: ordered.slice(i, i + MOBILE_PER_SLIDE) });
    }
    return out;
  }, [ordered]);

  const goSlide = (i) => {
    const el = sliderRef.current;
    if (!el) return;
    const n = Math.max(0, Math.min(mobileSlides.length - 1, i));
    el.scrollTo({ left: n * el.clientWidth, behavior: "smooth" });
  };

  const onSliderScroll = () => {
    const el = sliderRef.current;
    if (!el) return;
    const i = Math.round(el.scrollLeft / el.clientWidth);
    if (i !== slideIdx) setSlideIdx(i);
  };

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

      {ordered.length === 0 ? (
        <div className="text-center text-muted py-5">
          <i className="bi bi-file-earmark-x" style={{ fontSize: 48 }}></i>
          <p className="mt-3">{artikel.length ? "Artikel tidak ditemukan. Coba kata kunci lain." : "Belum ada artikel yang dipublikasikan."}</p>
        </div>
      ) : (
        <>
          {/* WEB / TABLET (≥768px): 1 banner + 6 di halaman 1, 6 per halaman berikutnya */}
          <div className="d-none d-md-block">
            {webPage.banner && (
              <Reveal>
                <BannerCard item={webPage.banner} first />
              </Reveal>
            )}

            <div className={`row g-4 ${webPage.banner ? "mt-1" : ""}`}>
              {webPage.grid.map((item) => (
                <div className="col-lg-4 col-md-6 d-flex align-items-stretch" key={item.key}>
                  <Reveal className="w-100" threshold={0.1}>
                    <ArticleCard item={item} />
                  </Reveal>
                </div>
              ))}
            </div>

            {webTotalPages > 1 && (
              <div className="read-pagination">
                <button
                  className="read-page-btn"
                  disabled={safePage === 1}
                  onClick={() => setPage(safePage - 1)}
                >
                  <i className="bi bi-chevron-left"></i> Sebelumnya
                </button>
                <div className="read-page-nums">
                  {Array.from({ length: webTotalPages }, (_, i) => i + 1).map((p) => (
                    <button
                      key={p}
                      className={`read-page-num${p === safePage ? " active" : ""}`}
                      onClick={() => setPage(p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
                <button
                  className="read-page-btn"
                  disabled={safePage === webTotalPages}
                  onClick={() => setPage(safePage + 1)}
                >
                  Selanjutnya <i className="bi bi-chevron-right"></i>
                </button>
              </div>
            )}
          </div>

          {/* MOBILE (≤767px): slide 1 = 1 banner + 3, slide 2 dst = 3 */}
          <div className="d-md-none">
            <div className="read-slider" ref={sliderRef} onScroll={onSliderScroll}>
              {mobileSlides.map((slide, gi) => (
                <div className="read-slide-item" key={gi}>
                  {slide.banner && <BannerCard item={slide.banner} first />}
                  {slide.items.length > 0 && (
                    <div className="read-slide-grid">
                      {slide.items.map((item) => (
                        <SmallCard key={item.key} item={item} />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {mobileSlides.length > 1 && (
              <div className="read-slide-nav">
                <button
                  className="read-slide-btn"
                  onClick={() => goSlide(slideIdx - 1)}
                  disabled={slideIdx === 0}
                  aria-label="Slide sebelumnya"
                >
                  <i className="bi bi-chevron-left"></i>
                </button>
                <div className="read-slide-dots">
                  {mobileSlides.map((_, i) => (
                    <button
                      key={i}
                      className={`read-slide-dot${i === slideIdx ? " active" : ""}`}
                      onClick={() => goSlide(i)}
                      aria-label={`Slide ${i + 1}`}
                    />
                  ))}
                </div>
                <button
                  className="read-slide-btn"
                  onClick={() => goSlide(slideIdx + 1)}
                  disabled={slideIdx === mobileSlides.length - 1}
                  aria-label="Slide berikutnya"
                >
                  <i className="bi bi-chevron-right"></i>
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function BannerCard({ item, first }) {
  return (
    <Link to={`/artikel/${item.id}`} className="article-featured w-100" style={{ display: "block", height: "100%" }}>
      <div className="article-featured-inner">
        <div className="article-featured-img-wrap">
          <img src={imageUrl(item.gambar)} alt="" />
          {first && (
            <span className="article-featured-badge">
              <i className="bi bi-stars me-1"></i>Terbaru
            </span>
          )}
        </div>
        <div className="article-featured-content">
          <span className="badge rounded-pill mb-2" style={{ background: "rgba(59,130,246,0.15)", color: "#3B82F6" }}>
            <i className="bi bi-newspaper me-1"></i>Artikel Unggulan
          </span>
          <h3 className="article-featured-title">{item.judul}</h3>
          <p className="text-muted" style={{ fontSize: "0.95rem", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
            {stripHtml(item.isi).slice(0, 220)}
          </p>
          <div className="d-flex flex-wrap gap-2 mt-2">
            <Chip icon="bi-calendar3" text={formatDate(item.tanggal)} />
            <Chip icon="bi-clock" text={readTimeOf(item.isi)} />
            <Chip icon="bi-eye" text={`${Number(item.views || 0).toLocaleString("id-ID")} dilihat`} />
          </div>
          <span className="article-featured-cta">
            Baca Selengkapnya <i className="bi bi-arrow-right"></i>
          </span>
        </div>
      </div>
    </Link>
  );
}

function ArticleCard({ item }) {
  return (
    <Link to={`/artikel/${item.id}`} className="article-card" style={{ display: "block", height: "100%" }}>
      <div style={{ overflow: "hidden", height: 190 }}>
        <img src={imageUrl(item.gambar)} alt="" className="article-thumb" style={{ height: 190, transition: "transform 0.4s" }} />
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
        <h5 className="article-title" style={{ display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{item.judul}</h5>
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
  );
}

function SmallCard({ item }) {
  return (
    <Link to={`/artikel/${item.id}`} className="article-card" style={{ display: "block", height: "100%" }}>
      <div style={{ overflow: "hidden", height: 100 }}>
        <img src={imageUrl(item.gambar)} alt="" className="article-thumb" style={{ height: 100, transition: "transform 0.4s" }} />
      </div>
      <div style={{ padding: "12px 14px" }}>
        <div className="article-date mb-2 d-flex align-items-center justify-content-between">
          <span>
            <i className="bi bi-calendar3 me-1" style={{ fontSize: 12 }}></i>
            {formatDate(item.tanggal)}
          </span>
          <span style={{ color: "#2563EB", fontSize: 12 }}>
            <i className="bi bi-eye me-1"></i>{Number(item.views || 0).toLocaleString("id-ID")}
          </span>
        </div>
        <h5 className="article-title" style={{ fontSize: "0.95rem", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
          {item.judul}
        </h5>
        <p className="text-muted mt-2 mb-2" style={{ fontSize: "0.78rem", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
          {stripHtml(item.isi).slice(0, 80)}
        </p>
        <div className="d-flex align-items-center justify-content-between">
          <span style={{ color: "#2563EB", fontWeight: 600, fontSize: "0.8rem" }}>
            Baca <i className="bi bi-arrow-right"></i>
          </span>
          <span className="read-time-chip">
            <i className="bi bi-clock"></i> {readTimeOf(item.isi)}
          </span>
        </div>
      </div>
    </Link>
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
