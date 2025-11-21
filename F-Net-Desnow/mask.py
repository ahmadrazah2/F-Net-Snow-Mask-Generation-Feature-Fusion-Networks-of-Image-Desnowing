import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import load_model

def predict_single(model, device, image_path, output_path="results/predicted_masks", threshold=0.3):
    """
    Predicts mask for a single image and saves a cleaned (binary) version.

    Args:
        model: Trained model
        device: Device (CPU/GPU)
        image_path: Path to input snowy image
        output_path: Folder to save predicted binary mask
        threshold: Threshold for binary mask (default: 0.3)
    """
    os.makedirs(output_path, exist_ok=True)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    # Predict mask
    with torch.no_grad():
        pred = model(tensor)
        pred_mask = pred.squeeze().cpu().numpy()

    # If multi-channel output, average to single channel
    if pred_mask.ndim == 3:
        pred_mask = np.mean(pred_mask, axis=0)

    # Normalize mask to [0, 1]
    mask_norm = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)

    # Apply threshold to remove noise and make binary mask
    mask_binary = (mask_norm > threshold).astype(np.uint8) * 255

    # Save clean binary mask
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_path, f"{file_name}_binary_mask.png")
    Image.fromarray(mask_binary).save(save_path)

    print(f"✅ Saved clean binary mask at: {save_path}")
    return save_path


if __name__ == "__main__":
    model, device = load_model("checkpoints/fusion_mask.pth")
    test_image = "/Users/ahmadraza/Documents/Desnow_Model/data/val/beautiful_smile_00004.jpg"
    predict_single(model, device, test_image)
