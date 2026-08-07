// Konfigurasi & helper Firebase (Realtime Database)
import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue, get, set, update, push, remove } from "firebase/database";

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

// ---- Artikel ----
export async function getArticles({ published } = {}) {
  const snap = await get(ref(rtdb, "artikel"));
  const raw = snap.val() || {};
  let list = Object.entries(raw)
    .filter(([, v]) => v && typeof v === "object")
    .map(([key, v]) => ({ key, id: v.id ?? key, ...v }));
  if (published !== undefined) list = list.filter((a) => Number(a.published) === Number(published));
  return list.sort((a, b) => String(b.tanggal).localeCompare(String(a.tanggal)) || (b.id - a.id));
}

export async function getArticleById(id) {
  const list = await getArticles();
  return list.find((a) => String(a.id) === String(id));
}

export async function saveArticle(artikel) {
  let id = artikel.id;
  if (id == null) {
    const list = await getArticles();
    id = list.length ? Math.max(...list.map((a) => Number(a.id) || 0)) + 1 : 1;
  }
  const { key: _drop, id: _idDrop, ...payload } = { ...artikel, id: Number(id) };
  await set(ref(rtdb, `artikel/a${id}`), payload);
  return id;
}

export async function deleteArticle(id) {
  await remove(ref(rtdb, `artikel/a${id}`));
}

export async function bumpArticleViews(id) {
  const snap = await get(ref(rtdb, `artikel/a${id}`));
  const cur = Number(snap.val()?.views || 0);
  await update(ref(rtdb, `artikel/a${id}`), { views: cur + 1 });
}

export function imageUrl(name) {
  if (!name) return "https://via.placeholder.com/600x400?text=No+Image";
  if (/^https?:\/\//.test(name)) return name;
  const base =
    import.meta.env.VITE_IMAGE_BASE_URL || import.meta.env.VITE_AI_SERVER_URL || "";
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

// ---- Upload gambar artikel (ke AI server local /static/uploads) ----
export async function uploadArticleImage(file) {
  const base = import.meta.env.VITE_AI_SERVER_URL;
  if (!base) return null;
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(`${base}/api/upload`, { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.filename) {
    throw new Error(data.error || "Upload gagal");
  }
  return `${base}/static/uploads/${data.filename}`;
}
