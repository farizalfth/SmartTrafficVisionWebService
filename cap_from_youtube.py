import cv2
import yt_dlp

def cap_from_youtube(url, resolution='360p'):
    """
    Mengambil stream YouTube dan mengembalikan cv2.VideoCapture
    """

    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]/best',
    }

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
