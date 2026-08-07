# -*- coding: utf-8 -*-
"""Migrasi data MySQL (XAMPP) -> Firebase Realtime Database.

Cara pakai:
    python migrate_to_firebase.py

Membuat backup JSON di folder exports/ lalu meng-upload
node cctv, artikel, dan admin ke Firebase Realtime Database.
"""
import json
import os
from datetime import date, datetime

import mysql.connector
import firebase_admin
from firebase_admin import credentials, db as firebase_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "smart_traffic",
}

DATABASE_URL = "https://smart-traffic-vision-app-default-rtdb.asia-southeast1.firebasedatabase.app/"


def _clean(value):
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def export_mysql():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor(dictionary=True)
    data = {}

    for table in ("cctv", "artikel", "admin"):
        cursor.execute(f"SELECT * FROM {table} ORDER BY id")
        rows = cursor.fetchall()
        rows = [{k: _clean(v) for k, v in row.items()} for row in rows]

        if table == "admin":
            data[table] = rows[0] if rows else {}
        else:
            prefix = "c" if table == "cctv" else "a"
            data[table] = {f"{prefix}{row['id']}": row for row in rows}

    cursor.close()
    conn.close()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, "mysql_export.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Backup JSON tersimpan: {path}")
    for table, rows in data.items():
        print(f"  - {table}: {len(rows) if isinstance(rows, dict) else 1} baris")
    return data


def upload_to_firebase(data):
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

    for table in ("cctv", "artikel", "admin"):
        ref = firebase_db.reference(table)
        ref.set(data[table])
        print(f"Upload {table} -> Firebase selesai.")


if __name__ == "__main__":
    data = export_mysql()
    upload_to_firebase(data)
    print("\nMigrasi selesai!")
