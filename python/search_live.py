import yt_dlp

with open("results.txt", "w", encoding="utf-8") as f:
    def search_live(query):
        f.write(f"--- {query} ---\n")
        ydl_opts = {'extract_flat': True, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                res = ydl.extract_info(f"ytsearch15:{query}", download=False)
                if 'entries' in res:
                    for entry in res['entries']:
                        f.write(f"[{entry['id']}] {entry.get('title', 'No title')}\n")
            except Exception as e:
                f.write(str(e) + "\n")

    search_live("India traffic camera CCTV intersection")
    search_live("Mumbai busy intersection CCTV")
    search_live("Delhi traffic signal crossing drone")
