"""
clean.py
--------
Generates a clean (desnowed) image using the predicted snow mask and OpenCV inpainting.

Steps:
1. Load the snowy image.
2. Predict the mask using trained F-Net model.
3. Convert mask → binary mask.
4. Perform inpainting using Telea's algorithm.
5. Visualize and save results.
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import load_model


# -------------------------------
# 🔧 Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def preprocess_image(image_path, device):
    """Load and resize an image to 256x256 for model input."""
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((256, 256))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
    return img_resized, input_tensor


# -------------------------------
# 🧠 Mask Prediction
# -------------------------------
def predict_mask(model, input_tensor):
    """Predict a snow mask using the trained model."""
    with torch.no_grad():
        pred_mask = model(input_tensor)
    return pred_mask


# -------------------------------
# 🧹 Inpainting Process
# -------------------------------
def inpaint(original_img_256, pred_mask, threshold=0.5):
    """
    Inpaint snowy regions from the image using the predicted mask.
    Args:
        original_img_256: PIL Image resized to 256x256
        pred_mask: predicted mask tensor from the model
        threshold: threshold for binary mask (default=0.5)
    Returns:
        clean_img (PIL), model_mask (np.array), binary_mask (np.array)
    """
    # Convert mask tensor to numpy array
    mask_tensor = pred_mask.squeeze(0).cpu().numpy()
    if mask_tensor.ndim == 3:
        mask_tensor = np.mean(mask_tensor, axis=0)

    # Normalize mask
    mask_normalized = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)

    # Binary mask
    mask_binary = (mask_normalized > threshold).astype(np.uint8) * 255

    # Convert images to OpenCV format (BGR)
    img_cv = np.array(original_img_256)[:, :, ::-1]
    mask_cv = mask_binary.astype(np.uint8)

    # Apply Telea’s inpainting
    inpainted = cv2.inpaint(img_cv, mask_cv, 5, cv2.INPAINT_TELEA)
    inpainted_rgb = inpainted[:, :, ::-1]  # Convert back to RGB

    clean_img = Image.fromarray(inpainted_rgb)
    return clean_img, mask_normalized, mask_binary


# -------------------------------
# 🖼️ Visualization
# -------------------------------
def visualize_all_stages(original_img, model_mask, binary_mask, clean_img, save_path=None):
    """Visualize all steps: original, model mask, binary mask, and clean image."""
    plt.figure(figsize=(20, 5))

    # 1. Original Snowy Image
    plt.subplot(1, 4, 1)
    plt.imshow(original_img)
    plt.title("Original Snowy Image (256×256)")
    plt.axis("off")

    # 2. Model Generated Mask
    plt.subplot(1, 4, 2)
    plt.imshow(model_mask, cmap="gray")
    plt.title("Model Generated Mask")
    plt.axis("off")

    # 3. Binary Mask
    plt.subplot(1, 4, 3)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Binary Mask (Thresholded)")
    plt.axis("off")

    # 4. Cleaned Image
    plt.subplot(1, 4, 4)
    plt.imshow(clean_img)
    plt.title("Cleaned Image (Inpainted)")
    plt.axis("off")

    plt.tight_layout()
    # plt.show()

    if save_path:
    # Ensure extension exists and valid
      if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        save_path = str(save_path).replace(']', '').strip() + ".png"

      clean_img.save(save_path)
      print(f"✅ Clean image saved as '{save_path}' (256x256)")

# -------------------------------
# 🚀 Main Execution
# -------------------------------
if __name__ == "__main__":
    # Load trained model
    model, device = load_model("checkpoints/fusion_mask.pth")

    # Input image path
    image_path = "/Users/ahmadraza/Documents/Desnow_Model/data/val/beautiful_smile_00004.jpg"  # Change to your own test image
    save_path = "/Users/ahmadraza/Documents/Desnow_Model/results/clean_images"
    os.makedirs(save_path, exist_ok=True)

    # Generate output filename automatically
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_path, f"{base_name}_clean.png")
    # Step 1: Preprocess and predict mask
    original_img_256, input_tensor = preprocess_image(image_path, device)
    pred_mask = predict_mask(model, input_tensor)

    # Step 2: Inpaint to remove snow
    clean_img, model_mask, binary_mask = inpaint(original_img_256, pred_mask, threshold=0.3)

    # Step 3: Visualize and save
    visualize_all_stages(original_img_256, model_mask, binary_mask, clean_img, save_path=save_path)
