import os
import requests
from tqdm import tqdm

ACCESS_KEY = "7TmF6WFfMA382r9XAPxZt41rQFniwYYMHTozOw6VRMc"

SAVE_DIR = "C:/workspace/srcipt/raw/images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 多關鍵字搜尋（Unsplash 不支援 . 語法）
queries = [
    "car",
    "sedan car",
    "suv car",
    "taxi car",
    "car with open door",
    "vehicle door open",
    "car door opening",
    "car door ajar",
    "car with door fully open"
]

per_page = 12
pages = 10

url = "https://api.unsplash.com/search/photos"

downloaded = set()

for query in queries:

    print(f"\nSearching: {query}")

    for page in range(1, pages + 1):

        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "client_id": ACCESS_KEY
        }

        r = requests.get(url, params=params)

        if r.status_code != 200:
            print("API error:", r.text)
            continue

        data = r.json()

        for item in data["results"]:

            img_url = item["urls"]["regular"]

            # 防止重複
            if img_url in downloaded:
                continue

            downloaded.add(img_url)

            try:

                img_data = requests.get(img_url, timeout=10).content

                filename = f"{query.replace(' ','_')}_{page}_{len(downloaded)}.jpg"

                path = os.path.join(SAVE_DIR, filename)

                with open(path, "wb") as f:
                    f.write(img_data)

                print("Saved:", filename)

            except Exception as e:

                print("Download error:", e)