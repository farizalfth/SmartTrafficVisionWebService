import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { loginAdmin, subscribeAuth } from "../lib/firebase";

export default function AdminLogin() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Jika sudah login, langsung arahkan ke dashboard admin.
  useEffect(() => {
    const off = subscribeAuth((user) => {
      if (user) navigate("/admin", { replace: true });
    });
    return off;
  }, [navigate]);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await loginAdmin(email.trim(), password);
      localStorage.setItem("stv_admin", "1");
      navigate("/admin");
    } catch (err) {
      const code = err?.code || "";
      if (code === "auth/user-not-found" || code === "auth/wrong-password" || code === "auth/invalid-credential") {
        setError("Email atau password salah!");
      } else if (code === "auth/invalid-email") {
        setError("Format email tidak valid!");
      } else if (code === "auth/too-many-requests") {
        setError("Terlalu banyak percobaan. Coba lagi beberapa saat.");
      } else {
        setError("Gagal masuk. Periksa konfigurasi Firebase Authentication.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-stage d-flex align-items-center justify-content-center" style={{ padding: "40px 16px" }}>
      <div className="login-grid"></div>
      <div className="login-orb orb1"></div>
      <div className="login-orb orb2"></div>
      <div className="login-orb orb3"></div>

      <div className="login-card-shell">
        {error && (
          <div className="login-error mb-4">
            <i className="bi bi-exclamation-octagon"></i>
            <span>{error}</span>
          </div>
        )}

        <div className="text-center mb-4">
          <div className="login-brand-wrap">
            <i className="bi bi-shield-lock"></i>
          </div>
          <h2 className="login-title gradient-text fw-bold">Admin Portal</h2>
          <p className="login-sub mb-0">Masuk untuk mengelola SmartTrafficVision</p>
        </div>

        <form onSubmit={submit}>
          <div className="input-group mb-3">
            <span className="input-group-text" style={{ background: "#1a1c24", border: "1px solid #333", color: "#3B82F6" }}>
              <i className="bi bi-envelope"></i>
            </span>
            <input
              type="email"
              className="form-control login-field"
              placeholder="Masukkan email admin"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
            />
          </div>

          <div className="input-group password-wrap mb-4">
            <span className="input-group-text" style={{ background: "#1a1c24", border: "1px solid #333", color: "#3B82F6" }}>
              <i className="bi bi-lock"></i>
            </span>
            <input
              type={showPass ? "text" : "password"}
              className="form-control login-field"
              style={{ paddingRight: 44 }}
              placeholder="Masukkan password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
            <button
              type="button"
              className="password-toggle"
              onClick={() => setShowPass((s) => !s)}
              tabIndex={-1}
              aria-label={showPass ? "Sembunyikan password" : "Tampilkan password"}
            >
              <i className={`bi ${showPass ? "bi-eye-slash" : "bi-eye"}`}></i>
            </button>
          </div>

          <button type="submit" className="btn btn-cta w-100" disabled={loading}>
            {loading ? <span className="spinner-border spinner-border-sm me-2"></span> : null}
            {loading ? "Memeriksa..." : "MASUK SEKARANG"}
          </button>
        </form>

        <div className="login-meta">
          <span className="login-meta-chip"><i className="bi bi-shield-check"></i>Firebase Auth</span>
          <span className="login-meta-chip"><i className="bi bi-lock"></i>Terenkripsi</span>
          <span className="login-meta-chip"><i className="bi bi-camera-video"></i>SmartTrafficVision</span>
        </div>

        <div className="login-secure-note">
          <i className="bi bi-shield-lock-fill"></i>
          Sesi Anda dikelola Firebase Authentication
        </div>

        <div className="text-center">
          <Link to="/" className="login-back">
            <i className="bi bi-arrow-left"></i>Kembali ke Beranda
          </Link>
        </div>
      </div>
    </div>
  );
}
