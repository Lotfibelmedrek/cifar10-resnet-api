import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 1. Reproducibility
torch.manual_seed(42)

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Using CIFAR10 dataset (downloads automatically)
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

# 3. Model Definition (Transfer Learning)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10) # 10 classes in CIFAR-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 5. Training Loop
model.train()
for epoch in range(5):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
print("Training complete.")
torch.save(model.state_dict(), "resnet18_cifar10.pth")