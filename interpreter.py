import cv2
import pytesseract
import re
from PIL import Image

# OPTIONAL: Set path to Tesseract executable (for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = "031.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: preprocess to improve OCR
gray = cv2.bilateralFilter(gray, 11, 17, 17)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Use Tesseract to extract text
text = pytesseract.image_to_string(thresh)
print("FULL OCR TEXT:\n", text)

# Extract name (usually first word or first line)
name_match = re.search(r"^\s*([A-Za-z]+)", text)
name = name_match.group(1) if name_match else "Not Found"

# Extract HP
hp_match = re.search(r"(\d+)\s*HP", text)
hp = hp_match.group(1) if hp_match else "Not Found"

# Extract attacks (simplified — looks for capitalized phrases with damage numbers)
attack_matches = re.findall(r"([A-Za-z\s]+)\n\d{1,3}", text)
attacks = [attack.strip() for attack in attack_matches]

# Display extracted info
print("\n🔍 Extracted Info:")
print(f"Name: {name}")
print(f"HP: {hp}")
print(f"Attacks: {attacks}")
