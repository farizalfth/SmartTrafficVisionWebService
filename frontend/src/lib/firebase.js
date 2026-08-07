// Konfigurasi & helper Firebase (Realtime Database)
import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue, get, set, update, push, remove } from "firebase/database";
import {
  listArticles,
  getArticle,
  saveArticle as supabaseSaveArticle,
  deleteArticle as supabaseDeleteArticle,
  bumpArticleViews as supabaseBumpArticleViews,
} from "./supabase";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  databaseURL: import.meta.env.VITE_FIREBASE_DATABASE_URL,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

export const app = initializeApp(firebaseConfig);
export const rtdb = getDatabase(app);

const toArray = (obj) =>
  obj == null ? [] : Array.isArray(obj) ? obj.filter(Boolean) : Object.values(obj).filter(Boolean);

// ---- CCTV ----
export function listenCctv(cb) {
  const r = ref(rtdb, "cctv");
  return onValue(r, (snap) => {
    const raw = snap.val() || {};
    const list = Object.entries(raw)
      .filter(([, v]) => v && typeof v === "object")
      .map(([key, v]) => ({
        id: (v.id ?? Number(String(key).replace(/\D/g, ""))) || key,
        ...v,
      }));
    cb(list.sort((a, b) => a.id - b.id));
  });
}

export async function getCctvList() {
  const snap = await get(ref(rtdb, "cctv"));
  const raw = snap.val() || {};
  return Object.entries(raw)
    .filter(([, v]) => v && typeof v === "object")
    .map(([key, v]) => ({
      id: (v.id ?? Number(String(key).replace(/\D/g, ""))) || key,
      ...v,
    }))
    .sort((a, b) => a.id - b.id);
}

export async function saveCctv(cctv) {
  const list = await getCctvList();
  let id = cctv.id;
  if (id == null) {
    id = list.length ? Math.max(...list.map((c) => Number(c.id) || 0)) + 1 : 1;
  }
  const { id: _drop, ...payload } = { ...cctv, id: Number(id) };
  await set(ref(rtdb, `cctv/c${id}`), payload);
  return id;
}

export async function deleteCctv(id) {
  await remove(ref(rtdb, `cctv/c${id}`));
}

// ---- Artikel (data disimpan di Supabase Postgres, lihat supabase.sql) ----
export async function getArticles({ published } = {}) {
  return listArticles({ published });
}

export async function getArticleById(id) {
  return getArticle(id);
}

export async function saveArticle(artikel) {
  return supabaseSaveArticle(artikel);
}

export async function deleteArticle(id) {
  return supabaseDeleteArticle(id);
}

export async function bumpArticleViews(id) {
  return supabaseBumpArticleViews(id);
}

const isUnreachableHost = (hostname = "") =>
  hostname === "localhost" ||
  hostname === "127.0.0.1" ||
  hostname === "0.0.0.0" ||
  hostname.startsWith("192.168.") ||
  hostname.startsWith("10.") ||
  hostname.startsWith("172.");

export function imageUrl(name) {
  if (!name) return "https://via.placeholder.com/600x400?text=No+Image";
  if (/^https?:\/\//.test(name)) {
    try {
      const u = new URL(name);
      if (isUnreachableHost(u.hostname)) return `${window.location.origin}${u.pathname}`;
    } catch {}
    return name;
  }
  let base =
    import.meta.env.VITE_IMAGE_BASE_URL || import.meta.env.VITE_AI_SERVER_URL || "";
  try {
    if (isUnreachableHost(new URL(base).hostname)) base = window.location.origin;
  } catch {}
  return `${base}/static/uploads/${name}`;
}

// ---- Traffic stats ----
export async function getTrafficStats() {
  const snap = await get(ref(rtdb, "traffic_stats"));
  return snap.val() || {};
}

export async function getTrafficNode(cctvId) {
  const snap = await get(ref(rtdb, `traffic_stats/${cctvId}`));
  return snap.val() || {};
}

export async function getLive(cctvId) {
  const snap = await get(ref(rtdb, `traffic_stats/${cctvId}/live`));
  return snap.val() || {};
}

export function listenLive(cctvId, cb) {
  const r = ref(rtdb, `traffic_stats/${cctvId}/live`);
  return onValue(r, (snap) => cb(snap.val() || {}));
}

export async function getComments() {
  const snap = await get(ref(rtdb, "user_comments"));
  return toArrayWithKey(snap.val());
}

export function listenComments(cb) {
  const r = ref(rtdb, "user_comments");
  return onValue(r, (snap) => cb(toArrayWithKey(snap.val())));
}

const toArrayWithKey = (obj) => {
  if (obj == null) return [];
  if (Array.isArray(obj)) return obj.filter(Boolean).map((v, i) => ({ key: `i${i}`, ...v }));
  return Object.entries(obj)
    .filter(([, v]) => v)
    .map(([key, v]) => ({ key, ...v }));
};

export async function pushComment(payload) {
  await push(ref(rtdb, "user_comments"), payload);
}

export async function updateComment(key, payload) {
  await update(ref(rtdb, `user_comments/${key}`), payload);
}

export async function deleteComment(key) {
  await remove(ref(rtdb, `user_comments/${key}`));
}

// ---- Admin auth (sederhana, dari node `admin`) ----
export async function checkAdmin(username, password) {
  const snap = await get(ref(rtdb, "admin"));
  const a = snap.val();
  return !!a && a.username === username && a.password === password;
}

// ---- Upload gambar artikel (Supabase Storage publik) ----
export async function uploadArticleImage(file) {
  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
  const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
  const bucket = import.meta.env.VITE_SUPABASE_BUCKET || "artikel";
  if (!supabaseUrl || !supabaseAnonKey) return null;

  const base = file.name.replace(/\.[^.]+$/, "");
  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  const safe = base.replace(/[^a-zA-Z0-9._-]/g, "_");
  const path = `${Date.now()}_${safe}${ext}`;
  const objectPath = `${encodeURIComponent(bucket)}/${encodeURIComponent(path)}`;

  const res = await fetch(`${supabaseUrl}/storage/v1/object/${objectPath}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${supabaseAnonKey}`,
      apikey: supabaseAnonKey,
      "Content-Type": file.type || "application/octet-stream",
      "x-upsert": "true",
    },
    body: file,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.message || data.error || "Upload gagal");
  }
  return `${supabaseUrl}/storage/v1/object/public/${objectPath}`;
}
