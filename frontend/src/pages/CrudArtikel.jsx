import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getArticleById, saveArticle, uploadArticleImage, imageUrl } from "../lib/firebase";

export default function CrudArtikel({ mode }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const [form, setForm] = useState({ judul: "", tanggal: "", isi: "", gambar: "" });
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
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

  const onFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    if (f) {
      const reader = new FileReader();
      reader.onload = (ev) => setPreview(ev.target.result);
      reader.readAsDataURL(f);
    } else {
      setPreview(null);
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      let gambar = form.gambar;
      if (file) {
        const url = await uploadArticleImage(file);
        if (url) gambar = url;
        else gambar = file.name; // fallback: simpan nama file (dilayani AI server /static/uploads)
      }
      const payload = { ...form, gambar, published: mode === "edit" ? undefined : 0 };
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
                <label className="form-label">Gambar Utama</label>
                <input type="file" className="form-control" accept="image/*" onChange={onFile} />
                <small className="text-muted">Ukuran gambar maksimal 2MB.</small>
              </div>

              {(preview || form.gambar) && (
                <div className="mb-3">
                  <label className="form-label">{preview ? "Preview Gambar Baru" : "Preview Gambar Lama"}</label>
                  <img
                    src={preview || imageUrl(form.gambar)}
                    alt=""
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
