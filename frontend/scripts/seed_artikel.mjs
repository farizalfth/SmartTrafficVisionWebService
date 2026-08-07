// Seed artikel ke Supabase (Postgres, tabel `artikel`).
// Wajib menjalankan frontend/supabase.sql di dashboard Supabase dulu (buat tabel + RLS).
//
// Cara pakai:
//   cd frontend
//   node --env-file=.env scripts/seed_artikel.mjs
//
// Secara default SEMUA artikel yang sudah ada dihapus, lalu artikel baru dimasukkan.
// Tambahkan --keep-existing untuk tidak menghapus artikel lama.

const URL = process.env.VITE_SUPABASE_URL;
const KEY = process.env.VITE_SUPABASE_ANON_KEY;
const TABLE = "artikel";

if (!URL || !KEY) {
  console.error("Isi VITE_SUPABASE_URL & VITE_SUPABASE_ANON_KEY di frontend/.env dulu.");
  process.exit(1);
}

const keepExisting = process.argv.includes("--keep-existing");

async function request(path, { method = "GET", body, prefer } = {}) {
  const headers = {
    apikey: KEY,
    Authorization: `Bearer ${KEY}`,
  };
  if (body) headers["Content-Type"] = "application/json";
  if (prefer) headers.Prefer = prefer;
  const res = await fetch(`${URL}/rest/v1/${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.message || data.details || `Supabase error (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

const artikel = {
  judul: "Pemotor Ngantuk Jatuh di Pantura Demak, Pembonceng Tewas Tertabrak Truk",
  tanggal: "2026-07-17T14:15",
  isi: `Demak - Kecelakaan maut terjadi di jalur Pantura Km 34+900, Desa Sedo, Kecamatan Demak, Kabupaten Demak. Satu orang tewas dalam peristiwa itu.

Kanit Gakkum Sat Lantas Polres Demak, Ipda Muhammad Lutfil Hakim, mengatakan peristiwa ini terjadi pada Jumat (17/7) sekitar pukul 14.15 WIB. Peristiwa bermula saat sepeda motor Honda Revo bernomor polisi AA-4868-ASB berjalan dari arah Semarang ke Kudus.

Setiba di TKP, sepeda motor itu oleng karena pengendaranya diduga mengalami microsleep. Hal itu bahkan sampai membuat kendaraannya menabrak median jalan.

Sumber: detikjateng, "Pemotor Ngantuk Jatuh di Pantura Demak, Pembonceng Tewas Tertabrak Truk" — https://www.detik.com/jateng/berita/d-8578822/pemotor-ngantuk-jatuh-di-pantura-demak-pembonceng-tewas-tertabrak-truk`,
  gambar:
    "https://bgxgcwhakgaxiejeadwq.supabase.co/storage/v1/object/public/Image%20Artikel/Laka%20Demak.jpeg",
  published: 1,
  views: 0,
};

async function main() {
  if (!keepExisting) {
    await request(`${TABLE}?id=gt.0`, { method: "DELETE" });
    console.log("Semua artikel lama dihapus.");
  }

  const rows = await request(TABLE, {
    method: "POST",
    body: artikel,
    prefer: "return=representation",
  });
  console.log(`Artikel disimpan dengan id=${rows[0].id}: ${rows[0].judul}`);
}

main().catch((e) => {
  console.error("Gagal:", e.message);
  process.exit(1);
});
