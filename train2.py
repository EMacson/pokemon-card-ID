import os, torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

DATA_FOLDER = "data\\cards"
BATCH_SIZE = 16
NUM_EPOCH = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- TRANSFORMS -----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

'''
for root, dirs, files in os.walk(DATA_FOLDER):
    if not files:
        print(root, len(files))
        os.rmdir(root)
'''

# ----- DATASET -----
dataset = datasets.ImageFolder(DATA_FOLDER, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- MODEL -----
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(DEVICE)

# ----- OPTIMIZER / LOSS -----
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ----- TRAINING LOOP -----
for epoch in range(NUM_EPOCH):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCH} - Loss: {running_loss/len(train_loader):.4f}")

# ----- SAVE MODEL -----
torch.save(model.state_dict(), "pokemon_model.pt")
print("Model saved as pokemon_model.pt")