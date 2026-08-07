import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getArticles, deleteArticle, saveArticle, imageUrl } from "../lib/firebase";

const PER_PAGE = 8;

export default function KelolaArtikel() {
  const [artikel, setArtikel] = useState([]);
  const [page, setPage] = useState(1);

  const reload = () => getArticles().then(setArtikel);
  useEffect(() => {
    reload();
  }, []);

  const totalPages = Math.max(1, Math.ceil(artikel.length / PER_PAGE));
  const current = artikel.slice((page - 1) * PER_PAGE, page * PER_PAGE);

  const hapus = async (a) => {
    if (confirm(`Apakah Anda yakin ingin menghapus artikel ini?`)) {
      await deleteArticle(a.id);
      reload();
    }
  };

  const togglePublish = async (a) => {
    const { key, ...rest } = a;
    await saveArticle({ ...rest, published: Number(a.published) === 1 ? 0 : 1 });
    reload();
  };

  return (
    <div className="container mt-4" style={{ paddingBottom: 50 }}>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2 className="mb-0 gradient-text fw-bold">Kelola Artikel</h2>
        <Link to="/artikel/tambah" className="btn btn-info rounded-pill">
          <i className="bi bi-plus-lg me-1"></i>Tambah Artikel
        </Link>
      </div>

      <div className="table-responsive">
        <table className="table table-dark table-striped align-middle">
          <thead>
            <tr>
              <th>No</th><th>Judul</th><th>Gambar</th><th>Status</th><th>Tanggal</th><th>Aksi</th>
            </tr>
          </thead>
          <tbody>
            {current.map((a, i) => (
              <tr key={a.key}>
                <td>{(page - 1) * PER_PAGE + i + 1}</td>
                <td style={{ maxWidth: 280 }}>{a.judul}</td>
                <td>
                  {a.gambar ? (
                    <img src={imageUrl(a.gambar)} alt="" width="80" height="60" style={{ objectFit: "cover", borderRadius: 8 }} />
                  ) : (
                    <span className="text-muted">Tidak ada</span>
                  )}
                </td>
                <td>
                  {Number(a.published) === 1 ? (
                    <span className="badge bg-success">Published</span>
                  ) : (
                    <span className="badge bg-secondary">Draft</span>
                  )}
                </td>
                <td>{fmtTanggal(a.tanggal)}</td>
                <td>
                  <div className="d-flex gap-1 flex-wrap">
                    <Link to={`/artikel/edit/${a.id}`} className="btn btn-sm btn-warning"><i className="bi bi-pencil me-1"></i>Edit</Link>
                    <button className="btn btn-sm btn-danger" onClick={() => hapus(a)}><i className="bi bi-trash me-1"></i>Hapus</button>
                    {Number(a.published) === 1 ? (
                      <button className="btn btn-sm btn-secondary" onClick={() => togglePublish(a)}><i className="bi bi-x-circle me-1"></i>Batal Publish</button>
                    ) : (
                      <button className="btn btn-sm btn-success" onClick={() => togglePublish(a)}><i className="bi bi-check-circle me-1"></i>Publish</button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {!current.length && (
              <tr><td colSpan={6} className="text-center text-muted py-4">Belum ada artikel.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <nav>
          <ul className="pagination justify-content-center">
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
              <li key={p} className={`page-item ${p === page ? "active" : ""}`}>
                <button className="page-link bg-dark text-white border-secondary" onClick={() => setPage(p)}>{p}</button>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </div>
  );
}

function fmtTanggal(t) {
  if (!t) return "N/A";
  return String(t).replace("T", " ").slice(0, 16);
}
