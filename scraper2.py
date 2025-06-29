import requests, os, cv2, csv
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from tqdm import tqdm

FOLDER = "data\\cards"

def save_html(html, path):
    with open(path, 'wb') as f:
        f.write(html)

def getData():
    base_url = "https://pkmncards.com/?s="
    headers = {"User-Agent": "Mozilla/5.0"}

    with open('pokemon.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            pokemonURL = base_url + row['Name'].strip()
            r = requests.get(pokemonURL)
            
            soup = BeautifulSoup(r.content, 'html.parser')
            articles = soup.find_all("article", class_="type-pkmn_card")

            folder = os.path.join(FOLDER, row['Name'])
            os.makedirs(folder, exist_ok=True)

            limit = 10

            count = 0
            for article in articles:
                if count >= limit:
                    break
                img_tag = article.find("img")
                if img_tag and "src" in img_tag.attrs:
                    img_url = img_tag["src"]
                    img_data = requests.get(img_url).content

                    img_path = os.path.join(folder, f"{row['Name']}_{count}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    count += 1
                    print(f"✅ Downloaded: {img_path}")

            if count == 0:
                print(f"❌ No images saved for {row['Name']}")



getData()