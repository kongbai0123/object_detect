import os
import requests

API_KEY = "Bu49eVz9MBtbomXaXANhxtgIqH4CGMTrgnsRpiRl7qcHspjVrlL4cq2P"

SAVE_DIR = "C:/workspace/srcipt/raw/images"
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {
    "Authorization": API_KEY
}

queries = [
    "car",
    "sedan",
    "suv",
    "taxi",
    "car door open"
]

for query in queries:

    url = f"https://api.pexels.com/v1/search?query={query}&per_page=50"

    r = requests.get(url, headers=headers)

    data = r.json()

    for photo in data["photos"]:

        img_url = photo["src"]["large"]

        img_id = photo["id"]

        filename = f"pexels_{img_id}.jpg"

        path = os.path.join(SAVE_DIR, filename)

        if os.path.exists(path):
            continue

        img = requests.get(img_url).content

        with open(path,"wb") as f:
            f.write(img)

        print("Saved", filename)