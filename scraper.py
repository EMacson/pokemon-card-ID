import requests
import csv
import os
import re
import json
import copy
from bs4 import BeautifulSoup

#url = 'https://pkmncards.com/card/lumineon-v-crown-zenith-crz-gg39/'

#r = requests.get(url)

BASE_URL = "https://pkmncards.com/card/"
HTML_FOLDER = "data\\html"
JSON_FOLDER = "data\\json"
IMG_FOLDER = "data\\img"

def save_html(html, path):
    with open(path, 'wb') as f:
        f.write(html)

def open_html(path):
    with open(path, 'rb') as f:
        return f.read()
    
def getHTML():
    with open('pokemon.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemonURL = BASE_URL + row['Name'].strip()
            pokemonFile = os.path.join(HTML_FOLDER, row['Name'] + '.html')
            r = requests.get(pokemonURL)
            save_html(r.content, pokemonFile)
            print(row['Name'])

def parseHP(str):
    hp = str.split()
    return hp[0]

def parseType(str):
    type = str.strip("{}")
    return type

def parseStage(str):
    if str =="Basic":
        return "0"
    else:
        stage = str.split()
        return stage[1]
    
def parseEvolve(str, stage):  
    line = str.split()
    level = int(stage)

    if level == 0:
        return {"evolvesTo": line[2], "evolvesFrom": ""}
    elif stage == 1:
        if len(line) >= 6:
            return {"evolvesTo": line[5], "evolvesFrom": line[2]}
        else:
            return {"evolvesTo": "", "evolvesFrom": line[2]}
    else:
        return {"evolvesTo": "", "evolvesFrom": line[2]}
    
def parseAttack(s):
    attack = {
      "name": "",
      "damage": "",
      "cost": "",
      "effect": ""
    }

    line = s.split()
    length = len(line)
    
    # get cost
    cost = ""
    for i in range(len(line[0])):
        if line[0][i] != '{' and line[0][i] != '}':
            cost += line[0][i]

    attack['cost'] = cost

    # get name
    i = 2
    name = ""
    while line[i] != ":":
        name += line[i] + " "
        i += 1

    attack['name'] = name
    i += 1

    # get damge
    pat = r'\d+'
    match = re.search(pat, line[i])
    attack['damage'] = match.group()

    # get description
    i += 1
    effect = ""
    while i < length:
        effect += line[i] + " " 
        i += 1
    
    attack['effect'] = effect

    return attack

def parseWeak(s):
    if s == "weak: n/a":
        return ""
    
    type = s[7]
    weak = ""
    match = re.search(r'×(\d+)', s)
    for i in range(int(match.group(1))):
        weak += type
    
    return weak

def parseResist(s):
    if s == "resist: n/a":
        return ""
    
def parseRetreat(s):
    match = re.search(r'retreat:\s*(\d+)', s)
    return match.group(1)

def test():
    file = "data\\html\\Abomasnow.html"
    html = open_html(file)
    soup = BeautifulSoup(html, 'html.parser')
    name = soup.select_one('.name').text.strip()
    hp = soup.select_one('.hp').text.strip()
    type = soup.select_one('.color').text.strip()
    stage = soup.select_one('.stage').text.strip()
    evolvesFrom = soup.select_one('.evolves').text.strip()
    attacks = soup.select('.text p')
    weak = soup.select_one('.weak').text.strip()
    resist = soup.select_one('.resist').text.strip()
    retreat = soup.select_one('.retreat').text.strip()
    print(name)
    print(parseHP(hp))
    print(parseType(type))
    print(parseStage(stage))
    print(parseEvolve(evolvesFrom, parseStage(stage)))
    print("num of attacks: ", len(attacks))
    for i in range(0, len(attacks)):
        #print(attacks[i].text.strip())
        #print(attacks[i].text.strip().split())
        print(parseAttack(attacks[i].text.strip()))
    print(parseWeak(weak))
    print(parseResist(resist))
    print(parseRetreat(retreat))

    print("\ncreate json")
    print("=============")

    with open('template.json', 'r') as f:
        template = json.load(f)

    new_card = copy.deepcopy(template)

    new_card['name'] = name
    new_card['hp'] = parseHP(hp)
    new_card['type'] = parseType(type)
    new_card['stage'] = parseStage(stage)
    evolution = parseEvolve(evolvesFrom, parseStage(stage))
    new_card['evolvesTo'] = evolution["evolvesTo"]
    new_card['evolvesFrom'] = evolution["evolvesFrom"]
    new_card['weak'] = parseWeak(weak)
    new_card['resist'] = parseResist(resist)
    new_card['retreat'] = parseRetreat(retreat)
    attackArray = []
    for i in range(len(attacks)):
        print('test')
        attackArray.append(parseAttack(attacks[i].text.strip()))
    print(attackArray)
    new_card['attacks'] = attackArray
    new_card['abilities'] = []

    with open('grovyle_card.json', 'w', encoding="utf-8") as f:
        json.dump(new_card, f, indent=2, ensure_ascii=False)

    print("\nget image")
    print('=========')
    img = soup.select_one('.card-image-link img')
    img_url = img['src']
    print(img_url)
    response = requests.get(img_url)
    if response.status_code == 200:
        with open('pokemon_card.jpg', 'wb') as f:
            f.write(response.content)
        print("Image downloaded successfully.")
    else:
        print("Failed to download image.")
    
def buildData():
    with open('pokemon.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
    
        for row in reader:
            pokemonFile = os.path.join(HTML_FOLDER, row['Name'] + '.html')
            jsonFile = os.path.join(JSON_FOLDER, row['Name'] + '.json')
            imgFile = os.path.join(IMG_FOLDER, row['Name'] + '.jpg')
            pokemonDataFile = "data\\pokemonData.csv"

            # parse html
            try:
                html = open_html(pokemonFile)
                soup = BeautifulSoup(html, 'html.parser')
                name = soup.select_one('.name').text.strip()
                hp = soup.select_one('.hp').text.strip()
                type = soup.select_one('.color').text.strip()
                stage = soup.select_one('.stage').text.strip()
                evolvesFrom = soup.select_one('.evolves').text.strip()
                attacks = soup.select('.text p')
                weak = soup.select_one('.weak').text.strip()
                resist = soup.select_one('.resist').text.strip()
                retreat = soup.select_one('.retreat').text.strip()
            except Exception as e:
                print("Something went wrong when parse html:", e)
                continue

            # write json
            try:
                with open('template.json', 'r') as f:
                    template = json.load(f)

                new_card = copy.deepcopy(template)

                new_card['name'] = name
                new_card['hp'] = parseHP(hp)
                new_card['type'] = parseType(type)
                new_card['stage'] = parseStage(stage)
                evolution = parseEvolve(evolvesFrom, parseStage(stage))
                new_card['evolvesTo'] = evolution["evolvesTo"]
                new_card['evolvesFrom'] = evolution["evolvesFrom"]
                new_card['weak'] = parseWeak(weak)
                new_card['resist'] = parseResist(resist)
                new_card['retreat'] = parseRetreat(retreat)
                attackArray = []
                for i in range(len(attacks)):
                    #print('test')
                    attackArray.append(parseAttack(attacks[i].text.strip()))
                #print(attackArray)
                new_card['attacks'] = attackArray
                new_card['abilities'] = []

                with open(jsonFile, 'w', encoding="utf-8") as f:
                    json.dump(new_card, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print("Something went wrong when writing json:", e)
                continue


            # get image
            try:
                img = soup.select_one('.card-image-link img')
                img_url = img['src']
                print(img_url)
                response = requests.get(img_url)
                if response.status_code == 200:
                    with open(imgFile, 'wb') as f:
                        f.write(response.content)
                    print("Image downloaded successfully.")
                else:
                    print("Failed to download image.")
            except Exception as e:
                print("Something went wrong when getting image:", e)


def deleteHTML():
    for file in os.listdir(HTML_FOLDER):
        if file.endswith('.json'):
            file_path = os.path.join(HTML_FOLDER, file)
            os.remove(file_path)
            print(f"Deleted {file_path}")

#deleteHTML()
#buildData()

#save_html(r.content, 'google_com')