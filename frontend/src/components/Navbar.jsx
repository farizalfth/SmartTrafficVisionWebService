import { useEffect, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { logoutAdmin } from "../lib/firebase";

export default function Navbar({ admin = false }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const close = () => setOpen(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const logout = async () => {
    try {
      await logoutAdmin();
    } catch {
      // Tetap lanjut ke halaman login meski sesi sudah tidak valid.
    }
    localStorage.removeItem("stv_admin");
    navigate("/login");
  };

  if (admin) {
    return (
      <nav className="navbar navbar-expand-lg navbar-dark fixed-top admin-nav">
        <div className="container">
          <Link className="navbar-brand" to="/admin">
            <span className="brand-icon"><i className="bi bi-activity"></i></span> STV ADMIN
          </Link>
          <div className="d-flex align-items-center gap-2">
            <Link className="btn btn-sm btn-outline-info rounded-pill" to="/" title="Lihat situs publik (keluar dari area admin)">
              <i className="bi bi-globe me-1"></i>Lihat Situs
            </Link>
            <button className="btn btn-sm btn-outline-light rounded-pill" onClick={logout}>
              Logout
            </button>
          </div>
        </div>
      </nav>
    );
  }

  return (
    <nav className="navbar navbar-expand-lg navbar-dark fixed-top">
      <div className="container">
        <Link className="navbar-brand" to="/" onClick={close}>
          <span className="brand-icon"><i className="bi bi-activity"></i></span> STV
        </Link>
        <button
          className={`navbar-toggler${open ? " open" : ""}`}
          type="button"
          aria-label={open ? "Tutup menu" : "Buka menu"}
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          <span className="hamb-ico" aria-hidden="true">
            <span className="hamb-bar"></span>
            <span className="hamb-bar"></span>
            <span className="hamb-bar"></span>
          </span>
        </button>
        <div className={`collapse navbar-collapse justify-content-end${open ? " show" : ""}`} id="navbarNav">
          <ul className="navbar-nav align-items-center" onClick={close}>
            <li className="nav-item">
              <NavLink className="nav-link" to="/dashboard">
                Dashboard
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink className="nav-link" to="/static-page">
                Statistik & Analitik
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink className="nav-link" to="/cctv-page">
                Peta CCTV
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink className="nav-link" to="/read_artikel">
                Artikel
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink className="nav-link" to="/about">
                Tentang Kami
              </NavLink>
            </li>
            <li className="nav-item ms-lg-3">
              <Link to="/login" className="btn btn-login" onClick={close}>
                Login Admin
              </Link>
            </li>
          </ul>
        </div>
      </div>
      {open && <div className="nav-backdrop" onClick={close}></div>}
    </nav>
  );
}
