import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { checkAdmin } from "../lib/firebase";

export default function AdminLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const ok = await checkAdmin(username, password);
      if (ok) {
        localStorage.setItem("stv_admin", "1");
        navigate("/admin");
      } else {
        setError("Username atau password salah!");
      }
    } catch (err) {
      setError("Gagal terhubung ke database. Periksa konfigurasi Firebase.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "calc(100vh - 150px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "radial-gradient(circle at center, #0d0d0d, #000000)",
        padding: "40px 16px",
      }}
    >
      <div
        style={{
          maxWidth: 400,
          width: "100%",
          backdropFilter: "blur(15px)",
          background: "rgba(30,30,30,0.6)",
          borderRadius: 20,
          padding: 40,
          border: "1px solid rgba(59,130,246,0.25)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        }}
      >
        {error && <div className="alert alert-danger">{error}</div>}
        <div className="text-center mb-4">
          <div
            style={{
              width: 60,
              height: 60,
              borderRadius: "50%",
              border: "2px solid #3B82F6",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 16px",
            }}
          >
            <i className="bi bi-shield-lock" style={{ fontSize: 32, color: "#3B82F6" }}></i>
          </div>
          <h2 className="gradient-text fw-bold">Admin Portal</h2>
          <p className="text-muted mb-0">Masuk untuk mengelola sistem</p>
        </div>

        <form onSubmit={submit}>
          <div className="input-group mb-3">
            <span className="input-group-text" style={{ background: "#2c2c2c", border: "1px solid #444", color: "#3B82F6" }}>
              <i className="bi bi-person"></i>
            </span>
            <input
              className="form-control"
              style={{ background: "#2c2c2c", border: "1px solid #444" }}
              placeholder="Masukkan username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="input-group mb-4">
            <span className="input-group-text" style={{ background: "#2c2c2c", border: "1px solid #444", color: "#3B82F6" }}>
              <i className="bi bi-lock"></i>
            </span>
            <input
              type="password"
              className="form-control"
              style={{ background: "#2c2c2c", border: "1px solid #444" }}
              placeholder="Masukkan password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="btn btn-cta w-100" disabled={loading}>
            {loading ? <span className="spinner-border spinner-border-sm me-2"></span> : null}
            {loading ? "Memeriksa..." : "MASUK SEKARANG"}
          </button>
        </form>

        <div className="text-center mt-4">
          <Link to="/" style={{ color: "#2563EB", fontSize: "0.9rem" }}>
            <i className="bi bi-arrow-left me-1"></i>Kembali ke Beranda
          </Link>
        </div>
      </div>
    </div>
  );
}
