# -*- coding: utf-8 -*-
"""Upload gambar artikel (static/uploads) ke Firebase Storage
dan perbarui field `gambar` di node artikel dengan URL publik."""
import json
import os

import firebase_admin
from firebase_admin import credentials, db as firebase_db, storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

DATABASE_URL = "https://smart-traffic-vision-app-default-rtdb.asia-southeast1.firebasedatabase.app/"
BUCKET = "smart-traffic-vision-app.firebasestorage.app"


def main():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL,
        "storageBucket": BUCKET,
    })
    bucket = storage.bucket()

    artikel = firebase_db.reference("artikel").get() or {}
    updated = 0

    for key, item in artikel.items():
        if not isinstance(item, dict):
            continue
        gambar = item.get("gambar")
        if not gambar:
            continue

        local = os.path.join(UPLOAD_DIR, os.path.basename(gambar))
        if not os.path.exists(local):
            print(f"  SKIP {gambar}: file tidak ada di static/uploads")
            continue

        remote = f"artikel/{os.path.basename(gambar)}"
        blob = bucket.blob(remote)
        if not blob.exists():
            blob.upload_from_filename(local)
            blob.make_public()
            print(f"  Upload {gambar} -> {remote}")

        item["gambar"] = blob.public_url
        firebase_db.reference(f"artikel/{key}").update({"gambar": blob.public_url})
        updated += 1

    print(f"\nSelesai: {updated} artikel diperbarui dengan URL Storage.")


if __name__ == "__main__":
    main()
