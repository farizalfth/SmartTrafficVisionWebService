import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import { subscribeAuth } from "../lib/firebase";

export default function ProtectedRoute({ children }) {
  const [checking, setChecking] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const off = subscribeAuth((u) => {
      setUser(u);
      setChecking(false);
    });
    return off;
  }, []);

  // Saat sesi belum selesai dipulihkan, tampilkan indikator loading.
  if (checking) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 16,
          background: "radial-gradient(circle at center, #0d0d0d, #000000)",
          color: "#fff",
        }}
      >
        <div className="spinner-border text-primary" role="status"></div>
        <span className="text-muted">Memeriksa sesi...</span>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
}
