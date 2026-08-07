export default function ProtectedRoute({ children }) {
  const isAdmin = localStorage.getItem("stv_admin") === "1";
  if (!isAdmin) {
    window.location.href = "/login";
    return null;
  }
  return children;
}
