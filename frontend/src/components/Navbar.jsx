import { Link, NavLink, useNavigate } from "react-router-dom";

export default function Navbar({ admin = false }) {
  const navigate = useNavigate();

  const logout = () => {
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
        <Link className="navbar-brand" to="/">
          <span className="brand-icon"><i className="bi bi-activity"></i></span> STV
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse justify-content-end" id="navbarNav">
          <ul className="navbar-nav align-items-center">
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
              <Link to="/login" className="btn btn-login">
                Login Admin
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
