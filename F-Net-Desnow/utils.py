import torch
from model import FusionNet

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def load_model(model_path="checkpoints/fusion_net.pth"):
    device = setup_device()
    model = FusionNet(in_channels=3)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    return model, device
