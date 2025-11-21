

# 📘 F-Net: Snow Mask Generation Feature-Fusion Networks of Image Desnowing

This repository contains the official implementation of **F-Net**, a deep learning–based framework for:

✔️ Snow mask generation
✔️ Image desnowing (clean image reconstruction)
✔️ Feature-fusion–based learning for snow localization

F-Net predicts a soft mask, converts it into a clean binary mask, and uses inpainting to generate a clean snow-free image.

---

## 🧠 Overview

**F-Net** consists of:

* **FusionNet** – extracts multi-scale snow features
* **Mask Generator (`mask.py`)** – converts predicted mask to a binary, noise-free mask
* **Clean Image Generator (`clean.py`)** – uses predicted mask + Telea inpainting for desnowing
* **Training Framework (`train.py`)** – with validation support
* **Focal + Dice Loss** – for accurate and stable mask generation

---

## 🗂️ Repository Structure

```
F-Net-Snow-Mask-Generation/
│
├── model.py                # F-Net architecture (ConvBlock + FusionNet)
├── dataset.py              # Dataset + dataloaders (train/val)
├── loss.py                 # Focal + Dice hybrid loss
├── train.py                # Training + validation loop
├── mask.py                 # Generate clean binary snow masks (formerly test.py)
├── clean.py                # Generate clean snow-free images using binary mask
├── utils.py                # Device setup, model loading utilities
│
├── checkpoints/            # Saved model weights
│   └── fusion_net.pth
│
├── data/
│   ├── train/
│   │   ├── snow/
│   │   └── mask/
│   └── val/
│       ├── snow/
│       └── mask/
│
├── results/
│   ├── predicted_masks/    # Output masks from mask.py
│   ├── clean_images/       # Output clean images from clean.py
│
└── requirements.txt        # Python dependencies
```

---

## 🔧 Installation

```bash
git clone https://github.com/<your-username>/F-Net-Snow-Mask-Generation.git
cd F-Net-Snow-Mask-Generation
pip install -r requirements.txt
```

---

## 🧩 Dataset Format

Your dataset must follow this structure:

```
data/
├── train/
│   ├── snow/
│   └── mask/
└── val/
    ├── snow/
    └── mask/
```

Filenames in `snow/` and `mask/` must match (e.g., `001.jpg` → `001.jpg`).

---

# 🎯 Usage Instructions (Important)

## ✅ 1. Generate a Snow Mask (mask.py)

If you want to generate a **binary snow mask**, open the file:

```
mask.py
```

Go to **line 55** and replace the image path:

```python
image_path = "path/to/your/snowy_image.jpg"
```

Then run:

```bash
python mask.py
```

Your cleaned, binary mask will be saved automatically in:

```
results/predicted_masks/
```

---

## ✅ 2. Generate a Clean (Desnowed) Image (clean.py)

If you want a **clean snow-free image**, open the file:

```
clean.py
```

Go to **line 139** and replace your image path:

```python
image_path = "path/to/your/snowy_image.jpg"
```

Then run:

```bash
python clean.py
```

Your clean image will be saved in:

```
results/clean_images/
```

---

## 🔥 Quick Summary

| Task                 | File       | Line to Edit | Run Command       |
| -------------------- | ---------- | ------------ | ----------------- |
| Generate Mask        | `mask.py`  | **55**       | `python mask.py`  |
| Generate Clean Image | `clean.py` | **139**      | `python clean.py` |

---

## 🖼️ Example Outputs

*(You can add images later)*

| Snow Image | Predicted Mask   | Binary Mask     | Clean Image          |
| ---------- | ---------------- | --------------- | -------------------- |
| *(input)*  | *(model output)* | *(thresholded)* | *(inpainted result)* |

---

## 📊 Training (Optional)

To train your model:

```bash
python train.py --epochs 50 --batch_size 8 --lr 1e-4
```

Checkpoints will be saved in:

```
checkpoints/
```

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.0.0
opencv-python>=4.7.0
matplotlib>=3.7.0
scikit-image>=0.20.0
tqdm>=4.65.0
```



---

## 👨‍💻 Author

**Ahmad Raza Hussain**
Master’s Student — Kyungsung University, Busan, South Korea
📧 [email:ahmadrazah2@gmail.com)


