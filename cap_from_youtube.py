import os
import shutil

import cv2
import yt_dlp


def _find_node():
    """Cari executable Node.js (dipakai yt-dlp sebagai JS runtime untuk ekstraksi YouTube)."""
    for cand in (os.environ.get("NODE_BIN"), shutil.which("node")):
        if cand and os.path.exists(cand):
            return cand
    return None


def cap_from_youtube(url, resolution='360p'):
    """
    Mengambil stream YouTube dan mengembalikan cv2.VideoCapture
    """

    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]/best',
    }

    node = _find_node()
    if node:
        ydl_opts['js_runtimes'] = {'node': {'path': node}}
    ydl_opts['remote_components'] = {'ejs:github'}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # Ambil stream sesuai resolusi jika ada
        formats = info.get('formats', [])
        stream_url = None

        for f in formats:
            if resolution.replace('p', '') in str(f.get('height')):
                stream_url = f.get('url')
                break

        # fallback
        if not stream_url:
            stream_url = info['url']

    return cv2.VideoCapture(stream_url)
