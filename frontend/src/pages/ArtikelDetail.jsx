import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getArticleById, getArticles, imageUrl, bumpArticleViews } from "../lib/firebase";

export default function ArtikelDetail() {
  const { id } = useParams();
  const [artikel, setArtikel] = useState(null);
  const [related, setRelated] = useState([]);
  const [readTime, setReadTime] = useState("1 mnt baca");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    getArticleById(id).then((a) => {
      if (!a) {
        window.location.href = "/read_artikel";
        return;
      }
      setArtikel(a);
      const words = (a.isi || "").split(/\s+/).filter(Boolean).length;
      setReadTime(`${Math.max(1, Math.ceil(words / 200))} mnt baca`);

      // Hit views sekali per sesi
      const key = `read_a${a.id}`;
      if (!sessionStorage.getItem(key)) {
        sessionStorage.setItem(key, "1");
        bumpArticleViews(a.id).catch(() => {});
      }
    });

    getArticles({ published: 1 }).then((list) => {
      setRelated(
        list
          .filter((x) => String(x.id) !== String(id))
          .sort((x, y) => String(y.tanggal).localeCompare(String(x.tanggal)))
          .slice(0, 3)
      );
    });
  }, [id]);

  // Progress baca
  useEffect(() => {
    const onScroll = () => {
      const el = document.getElementById("progressFill");
      if (!el) return;
      const h = document.documentElement;
      const pct = (h.scrollTop / (h.scrollHeight - h.clientHeight)) * 100;
      el.style.width = `${pct}%`;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  if (!artikel) {
    return (
      <div className="container text-center py-5">
        <div className="spinner mx-auto mb-3"></div>
        <p className="text-muted">Memuat artikel...</p>
      </div>
    );
  }

  const shareUrl = encodeURIComponent(window.location.href);
  const shareText = encodeURIComponent(artikel.judul);

  const copyLink = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(window.location.href);
      } else {
        const ta = document.createElement("textarea");
        ta.value = window.location.href;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied("gagal");
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const nativeShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: artikel.judul,
          text: artikel.judul,
          url: window.location.href,
        });
      } catch {}
    } else {
      copyLink();
    }
  };

  const hasCover = !!artikel.gambar && imageUrl(artikel.gambar) !== "https://via.placeholder.com/600x400?text=No+Image";

  return (
    <div>
      {/* Reading progress */}
      <div style={{ position: "fixed", top: 0, left: 0, width: "100%", height: 4, zIndex: 2000, background: "transparent" }}>
        <div id="progressFill" style={{ height: "100%", width: 0, background: "linear-gradient(90deg,#3B82F6,#2563EB)" }}></div>
      </div>

      <div className="container" style={{ maxWidth: 820, paddingBottom: 50 }}>
        {/* HERO */}
        {hasCover ? (
          <div className="article-hero">
            <img src={imageUrl(artikel.gambar)} alt="" className="article-hero-bg" />
            <div className="article-hero-overlay"></div>
            <div className="article-hero-content">
              <span className="badge rounded-pill mb-3" style={{ background: "rgba(59,130,246,0.35)", color: "#fff", backdropFilter: "blur(4px)" }}>
                <i className="bi bi-newspaper me-1"></i>Artikel Smart Traffic
              </span>
              <h1 className="article-hero-title">{artikel.judul}</h1>
              <div className="d-flex flex-wrap gap-2">
                <HeroPill icon="bi-calendar3" text={formatDate(artikel.tanggal)} />
                <HeroPill icon="bi-person" text="Admin" />
                <HeroPill icon="bi-clock" text={readTime} />
                <HeroPill icon="bi-eye" text={`${Number(artikel.views || 0).toLocaleString("id-ID")} dilihat`} />
              </div>
            </div>
          </div>
        ) : (
          <article style={{ background: "linear-gradient(145deg,#151515,#101010)", border: "1px solid #2a2a2a", borderRadius: 22, padding: "42px 48px" }}>
            <span className="badge rounded-pill mb-3" style={{ background: "rgba(59,130,246,0.15)", color: "#3B82F6" }}>
              <i className="bi bi-newspaper me-1"></i>Artikel Smart Traffic
            </span>
            <h1 className="gradient-text" style={{ fontSize: "1.8rem", fontWeight: 700, lineHeight: 1.3 }}>
              {artikel.judul}
            </h1>
            <div className="d-flex flex-wrap gap-2 my-4">
              <MetaPill icon="bi-calendar3" text={formatDate(artikel.tanggal)} />
              <MetaPill icon="bi-person" text="Admin" />
              <MetaPill icon="bi-clock" text={readTime} />
              <MetaPill icon="bi-eye" text={`${Number(artikel.views || 0).toLocaleString("id-ID")} dilihat`} />
            </div>
          </article>
        )}

        {/* KONTEN */}
        <article className="article-body">
          <div style={{ whiteSpace: "pre-wrap", textAlign: "justify", lineHeight: 1.85, color: "#e2e8f0", fontSize: "1.02rem" }}>
            {artikel.isi}
          </div>

          {/* INFO BOX */}
          <div className="article-info-box">
            <div className="article-info-avatar">
              <i className="bi bi-robot"></i>
            </div>
            <div>
              <div className="fw-bold" style={{ color: "#fff" }}>
                Smart Traffic Vision
                <span className="badge rounded-pill ms-2" style={{ background: "rgba(0,230,118,0.15)", color: "#00E676" }}>
                  <i className="bi bi-patch-check-fill me-1"></i>Terverifikasi
                </span>
              </div>
              <div className="text-muted" style={{ fontSize: "0.88rem" }}>
                Ditulis oleh <b style={{ color: "#bfdbfe" }}>Admin</b> — Info & berita lalu lintas terbaru dari sistem pemantauan Smart Traffic.
              </div>
            </div>
          </div>

          {/* BAGIKAN */}
          <div className="d-flex align-items-center flex-wrap share-row mt-5 pt-4" style={{ borderTop: "1px solid #2a2a2a", gap: "8px 10px" }}>
            <span className="me-1 text-muted share-label" style={{ fontSize: "0.9rem" }}>Bagikan:</span>
            <button className="share-btn" onClick={nativeShare} title="Bagikan melalui aplikasi lain">
              <i className="bi bi-share-fill"></i>
              <span className="ms-1" style={{ fontSize: "0.8rem" }}>Bagikan</span>
            </button>
            <ShareBtn href={`https://www.facebook.com/sharer/sharer.php?u=${shareUrl}`} icon="bi-facebook" />
            <ShareBtn href={`https://twitter.com/intent/tweet?url=${shareUrl}&text=${shareText}`} icon="bi-twitter" />
            <ShareBtn href={`https://wa.me/?text=${shareText}%20${shareUrl}`} icon="bi-whatsapp" />
            <button className="share-btn" onClick={copyLink} title="Salin Tautan">
              {copied === true ? (
                <i className="bi bi-check-lg" style={{ color: "#00E676" }}></i>
              ) : copied === "gagal" ? (
                <i className="bi bi-x-lg" style={{ color: "#FF5252" }}></i>
              ) : (
                <i className="bi bi-link-45deg"></i>
              )}
              <span className="ms-1" style={{ fontSize: "0.8rem" }}>
                {copied === true ? "Tersalin!" : copied === "gagal" ? "Gagal Salin" : "Salin Tautan"}
              </span>
            </button>
          </div>
        </article>

        {/* BACA JUGA */}
        {related.length > 0 && (
          <div className="mt-5">
            <h4 className="fw-bold mb-3" style={{ color: "#fff" }}>
              <i className="bi bi-collection me-2 gradient-icon"></i>Baca Juga
            </h4>
            <div className="row g-4">
              {related.map((item) => (
                <div className="col-md-4 d-flex align-items-stretch" key={item.key}>
                  <Link to={`/artikel/${item.id}`} className="article-card w-100">
                    <div style={{ overflow: "hidden", height: 140 }}>
                      <img src={imageUrl(item.gambar)} alt="" className="article-thumb" style={{ transition: "transform 0.4s" }} />
                    </div>
                    <div style={{ padding: "14px 16px" }}>
                      <div className="article-date mb-1" style={{ fontSize: 13 }}>
                        <i className="bi bi-calendar3 me-1" style={{ fontSize: 13 }}></i>
                        {formatDate(item.tanggal)}
                      </div>
                      <h6 className="article-title" style={{
                        display: "-webkit-box",
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                        fontSize: "0.95rem",
                      }}>{item.judul}</h6>
                    </div>
                  </Link>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        .share-btn {
          background: rgba(59,130,246,0.1);
          border: 1px solid rgba(59,130,246,0.25);
          color: #3B82F6;
          border-radius: 50px;
          padding: 8px 16px;
          display: inline-flex;
          align-items: center;
          cursor: pointer;
          transition: 0.3s;
          white-space: nowrap;
        }
        .share-btn:hover { background: rgba(59,130,246,0.25); transform: translateY(-2px); }
        @media (max-width: 576px) {
          .share-label { flex-basis: 100%; }
          .share-btn { padding: 6px 12px; }
          .share-btn span { font-size: 0.75rem; }
        }
      `}</style>
    </div>
  );
}

function HeroPill({ icon, text }) {
  return (
    <span className="badge rounded-pill" style={{ background: "rgba(0,0,0,0.45)", color: "#e5e7eb", padding: "8px 16px", fontWeight: 500, backdropFilter: "blur(4px)" }}>
      <i className={`bi ${icon} me-1`} style={{ color: "#3B82F6" }}></i>{text}
    </span>
  );
}

function MetaPill({ icon, text }) {
  return (
    <span className="badge rounded-pill" style={{ background: "rgba(59,130,246,0.15)", color: "#bfdbfe", padding: "8px 16px", fontWeight: 500 }}>
      <i className={`bi ${icon} me-1`} style={{ color: "#3B82F6" }}></i>{text}
    </span>
  );
}

function ShareBtn({ href, icon }) {
  return (
    <a className="share-btn" href={href} target="_blank" rel="noreferrer">
      <i className={`bi ${icon}`}></i>
    </a>
  );
}

function formatDate(t) {
  if (!t) return "-";
  const d = new Date(String(t).replace(" ", "T"));
  if (isNaN(d)) return String(t).slice(0, 10);
  return d.toLocaleDateString("id-ID", { day: "2-digit", month: "long", year: "numeric" });
}
