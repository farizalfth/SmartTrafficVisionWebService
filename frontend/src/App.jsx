import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import StaticPage from "./pages/StaticPage";
import CctvPage from "./pages/CctvPage";
import ReadArtikel from "./pages/ReadArtikel";
import ArtikelDetail from "./pages/ArtikelDetail";
import About from "./pages/About";
import AdminLogin from "./pages/AdminLogin";
import AdminDashboard from "./pages/AdminDashboard";
import KelolaArtikel from "./pages/KelolaArtikel";
import CrudArtikel from "./pages/CrudArtikel";
import ProtectedRoute from "./components/ProtectedRoute";
import ScrollToTop from "./components/ScrollToTop";

export default function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <Routes>
        <Route
          path="/"
          element={
            <>
              <Navbar />
              <Home />
              <Footer />
            </>
          }
        />
        <Route
          path="/dashboard"
          element={
            <>
              <Navbar />
              <Dashboard />
              <Footer />
            </>
          }
        />
        <Route
          path="/static-page"
          element={
            <>
              <Navbar />
              <StaticPage />
              <Footer />
            </>
          }
        />
        <Route
          path="/cctv-page"
          element={
            <>
              <Navbar />
              <CctvPage />
              <Footer />
            </>
          }
        />
        <Route
          path="/read_artikel"
          element={
            <>
              <Navbar />
              <ReadArtikel />
              <Footer />
            </>
          }
        />
        <Route
          path="/artikel/:id"
          element={
            <>
              <Navbar />
              <ArtikelDetail />
              <Footer />
            </>
          }
        />
        <Route
          path="/about"
          element={
            <>
              <Navbar />
              <About />
              <Footer />
            </>
          }
        />
        <Route
          path="/login"
          element={
            <>
              <Navbar />
              <AdminLogin />
              <Footer />
            </>
          }
        />
        <Route
          path="/admin"
          element={
            <ProtectedRoute>
              <Navbar admin />
              <AdminDashboard />
              <Footer />
            </ProtectedRoute>
          }
        />
        <Route
          path="/kelola_artikel"
          element={
            <ProtectedRoute>
              <Navbar admin />
              <KelolaArtikel />
              <Footer />
            </ProtectedRoute>
          }
        />
        <Route
          path="/artikel/tambah"
          element={
            <ProtectedRoute>
              <Navbar admin />
              <CrudArtikel mode="tambah" />
              <Footer />
            </ProtectedRoute>
          }
        />
        <Route
          path="/artikel/edit/:id"
          element={
            <ProtectedRoute>
              <Navbar admin />
              <CrudArtikel mode="edit" />
              <Footer />
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
