import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getArticleById, saveArticle, imageUrl } from "../lib/firebase";

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "https://<project>.supabase.co";

export default function CrudArtikel({ mode }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const [form, setForm] = useState({ judul: "", tanggal: "", isi: "", gambar: "" });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (mode === "edit" && id) {
      getArticleById(id).then((a) => {
        if (!a) return;
        setForm({
          judul: a.judul || "",
          tanggal: (a.tanggal || "").replace(" ", "T").slice(0, 16),
          isi: a.isi || "",
          gambar: a.gambar || "",
        });
      });
    }
  }, [mode, id]);

  const submit = async (e) => {
    e.preventDefault();
    const link = form.gambar.trim();
    if (link && !link.startsWith("http://") && !link.startsWith("https://")) {
      alert("Link gambar harus berupa URL yang diawali http:// atau https://");
      return;
    }
    if (link && !new URL(link).origin.includes("supabase.co")) {
      alert(`Link gambar harus dari Supabase Storage, contoh:\n${SUPABASE_URL}/storage/v1/object/public/Image Artikel/nama-file.jpg`);
      return;
    }
    setSaving(true);
    try {
      const payload = { ...form, gambar: link, published: mode === "edit" ? undefined : 0 };
      if (mode === "edit") {
        const existing = await getArticleById(id);
        payload.published = existing?.published ?? 0;
        payload.id = id;
      }
      await saveArticle(payload);
      navigate("/kelola_artikel");
    } catch (err) {
      alert(`Gagal menyimpan artikel: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="container" style={{ paddingBottom: 50 }}>
      <div className="row justify-content-center">
        <div className="col-lg-8">
          <div className="dashboard-card" style={{ maxWidth: 700, margin: "0 auto" }}>
            <h3 className="gradient-text fw-bold mb-4">{mode === "edit" ? "Edit Artikel" : "Tambah Artikel"}</h3>
            <form onSubmit={submit}>
              <div className="mb-3">
                <label className="form-label">Judul Artikel</label>
                <input className="form-control" value={form.judul} onChange={(e) => setForm({ ...form, judul: e.target.value })} required />
              </div>
              <div className="mb-3">
                <label className="form-label">Tanggal Publish</label>
                <input type="datetime-local" className="form-control" value={form.tanggal} onChange={(e) => setForm({ ...form, tanggal: e.target.value })} required />
                <small className="text-muted">Format: YYYY-MM-DD HH:MM</small>
              </div>
              <div className="mb-3">
                <label className="form-label">Isi Artikel</label>
                <textarea className="form-control" rows={8} value={form.isi} onChange={(e) => setForm({ ...form, isi: e.target.value })} required />
              </div>
              <div className="mb-3">
                <label className="form-label">Link Gambar (Supabase Storage)</label>
                <input
                  type="url"
                  className="form-control"
                  placeholder={`${SUPABASE_URL}/storage/v1/object/public/Image Artikel/nama-file.jpg`}
                  value={form.gambar}
                  onChange={(e) => setForm({ ...form, gambar: e.target.value })}
                />
                <small className="text-muted">
                  Upload gambar ke Supabase Storage (bucket <b>Image Artikel</b>), lalu salin linknya ke sini.
                </small>
              </div>

              {form.gambar && (
                <div className="mb-3">
                  <label className="form-label">Preview Gambar</label>
                  <img
                    src={imageUrl(form.gambar)}
                    alt=""
                    onError={(e) => { e.target.style.display = "none"; }}
                    style={{ maxWidth: "100%", maxHeight: 240, borderRadius: 10, display: "block" }}
                  />
                </div>
              )}

              <div className="d-flex gap-2">
                <Link to="/kelola_artikel" className="btn btn-secondary">Batal</Link>
                <button type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? "Menyimpan..." : mode === "edit" ? "Update Artikel" : "Tambah & Simpan"}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
