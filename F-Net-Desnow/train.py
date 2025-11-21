import torch
from tqdm import tqdm
from model import FusionNet
from dataset import get_dataloaders
from loss import FocalDiceLoss
from utils import setup_device

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    device = setup_device()
    model = FusionNet(in_channels=3).to(device)
    criterion = FocalDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader, val_loader = get_dataloaders(
        train_snow="data/train/snow",
        train_mask="data/train/mask",
        val_snow="data/val/snow",
        val_mask="data/val/mask",
        batch_size=4
    )

    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)
