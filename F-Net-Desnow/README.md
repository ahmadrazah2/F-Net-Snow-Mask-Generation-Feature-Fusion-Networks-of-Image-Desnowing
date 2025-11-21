
📄 **F-Net: Snow Mask Generation Feature-Fusion Networks of Image Desnowing**

---

## 📘 **README.md**

```markdown
# F-Net: Snow Mask Generation Feature-Fusion Networks of Image Desnowing

This repository contains the official implementation of **F-Net**, a deep learning-based approach for **snow mask generation** and **image desnowing**.  
F-Net introduces a feature-fusion strategy that combines hierarchical representations to accurately detect and remove snow artifacts from images.

---

## 🧠 Overview

**F-Net** consists of:
- A **Fusion Network** to learn multi-scale snow features.
- A **Snow Mask Generation** mechanism that outputs a binary or soft snow region mask.
- A reconstruction module (can be extended) to recover the clean, snow-free image.

---

## 🏗️ Repository Structure



F-Net-Snow-Mask-Generation/
│
├── model.py           # F-Net architecture (ConvBlock + FusionNet)
├── dataset.py         # Dataset and dataloaders for training and validation
├── loss.py            # Focal + Dice hybrid loss function
├── train.py           # Training + validation loop
├── test.py            # Inference and mask prediction
├── utils.py           # Device setup, model loading utilities
│
├── checkpoints/       # Saved model weights
│   └── fusion_mask.pth
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
│   ├── predicted_masks/
│   └── clean_images/
│
├── requirements.txt   # Python dependencies
└── README.md

````

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ahmadraza/F-Net-Snow-Mask-Generation.git
   cd F-Net-Snow-Mask-Generation
````

2. **Create and activate a virtual environment**

   ```bash
   python -m venv fnet_env
   source fnet_env/bin/activate   # For Mac/Linux
   fnet_env\Scripts\activate      # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧩 Dataset Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── snow/   # Snowy images
│   └── mask/   # Ground truth masks
└── val/
    ├── snow/
    └── mask/
```

✅ Both snow and mask folders should contain images with the same filenames (e.g., `0001.jpg` in both).

---

## 🚀 Training

Train the model with:

```bash
python train.py
```

You can adjust parameters like batch size, learning rate, and number of epochs inside `train.py`.

---

## 🔍 Testing / Inference

To generate a snow mask for a new image:

```bash
python test.py
```

Default paths:

* Input image: `data/val/snow/sample.jpg`
* Output mask: `results/predicted_masks/sample.jpg`

---

## 🧪 Loss Function

The training uses a **Focal + Dice hybrid loss** that enhances learning for unbalanced snow/no-snow regions:

[
L = (1 + \text{DiceLoss}) \times \text{FocalLoss}
]

---

## 🖼️ Results

| Snowy Image                               | Ground Truth Mask                      | Predicted Mask                                | Enhanced (Desnowed)                              |
| ----------------------------------------- | -------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| ![Snowy](results/visualizations/snow.jpg) | ![GT](results/visualizations/mask.jpg) | ![Pred](results/visualizations/pred_mask.jpg) | ![Desnowed](results/visualizations/desnowed.jpg) |

---

## ⚙️ Model Checkpoints

Trained weights are automatically saved in:

```
checkpoints/epoch_1.pth
checkpoints/epoch_2.pth
...
```

You can specify a checkpoint to resume or test in `load_model()` inside `utils.py`.

---

## 🧑‍💻 Citation

If you use this code in your research, please cite:

```
@article{ahmad2025fnet,
  title={F-Net: Snow Mask Generation Feature-Fusion Networks of Image Desnowing},
  author={Ahmad Raza and [Your Supervisor's Name]},
  year={2025},
  journal={Under Review}
}
```

---

## 🌍 Acknowledgments

This work was developed at **Kyungsung University, Busan**, under the supervision of **Prof. 김민주**.
Special thanks to the open-source PyTorch community and prior works in image desnowing research.

---

## 📫 Contact

For queries or collaborations:
**Ahmad Raza**
📧 [[ahmadrazah2@gmail.com]]
🎓 M.S. IT Engineering, Kyungsung University
🇰🇷 Busan, South Korea

````

---

## 📦 **requirements.txt**

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
tqdm>=4.66.0

opencv-python>=4.8.0
matplotlib>=3.8.0
````

---
