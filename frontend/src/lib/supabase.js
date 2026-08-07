// Helper Supabase (Postgres via REST) untuk data artikel.
// Butuh tabel `artikel` (lihat supabase.sql) + RLS policy untuk role anon.
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
const TABLE = "artikel";

function headers(json) {
  return {
    apikey: SUPABASE_KEY,
    Authorization: `Bearer ${SUPABASE_KEY}`,
    ...(json ? { "Content-Type": "application/json" } : {}),
  };
}

async function request(path, { method = "GET", body, prefer } = {}) {
  const h = headers(!!body);
  if (prefer) h.Prefer = prefer;
  const res = await fetch(`${SUPABASE_URL}/rest/v1/${path}`, {
    method,
    headers: h,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.message || data.details || `Supabase error (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

const toRow = (r) => ({ key: String(r.id), id: r.id, ...r });

// ---- Artikel CRUD ----
export async function listArticles({ published } = {}) {
  let q = "select=*&order=tanggal.desc";
  if (published !== undefined) q += `&published=eq.${Number(published)}`;
  const rows = await request(`${TABLE}?${q}`);
  return rows.map(toRow);
}

export async function getArticle(id) {
  const rows = await request(`${TABLE}?select=*&id=eq.${encodeURIComponent(id)}`);
  return rows.length ? toRow(rows[0]) : null;
}

const FIELDS = ["judul", "tanggal", "isi", "gambar", "published", "views"];

export async function saveArticle(artikel) {
  const payload = {};
  for (const f of FIELDS) if (artikel[f] !== undefined) payload[f] = artikel[f];
  if (payload.published === undefined) payload.published = 0;

  if (artikel.id == null || artikel.id === "") {
    const rows = await request(TABLE, {
      method: "POST",
      body: payload,
      prefer: "return=representation",
    });
    return rows[0];
  }
  await request(`${TABLE}?id=eq.${encodeURIComponent(artikel.id)}`, {
    method: "PATCH",
    body: payload,
  });
  return { id: artikel.id };
}

export async function deleteArticle(id) {
  await request(`${TABLE}?id=eq.${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function bumpArticleViews(id) {
  const a = await getArticle(id);
  if (!a) return;
  await request(`${TABLE}?id=eq.${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: { views: Number(a.views || 0) + 1 },
  });
}
