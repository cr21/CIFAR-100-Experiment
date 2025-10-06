import torch
import torch.nn as nn
import torchvision
from pathlib import Path

# 1. Define the model architecture (same as in cifar_100.py)
def get_model(name="resnet18", num_classes=100):
    model = getattr(torchvision.models, name)(weights=None)
    if hasattr(model, 'fc'):
        featin = model.fc.in_features
        model.fc = nn.Linear(featin, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

if __name__ == "__main__":
    # Define paths
    model_name = "resnet18"
    num_classes = 100
    model_dir = Path("/Users/chiragtagadiya/Downloads/MyProjects/ERA4/CIFAR-100-Experiment")
    pth_path = model_dir / "best.pth"
    pt_path = model_dir / f"{model_name}_cifar100.pt"

    # 2. Instantiate the model
    model = get_model(name=model_name, num_classes=num_classes)
    model.eval() # Set to evaluation mode

    # 3. Load the state_dict from best.pth
    if pth_path.exists():
        print(f"Loading model state from {pth_path}")
        # The best.pth saves a dictionary with 'epoch', 'model_state', 'best_acc'
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
        print("Model state loaded successfully.")
    else:
        print(f"Error: {pth_path} not found. Please ensure the path is correct.")
        exit()

    # 4. Convert the model to TorchScript (.pt file)
    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
    print(f"Tracing model with dummy input of size {dummy_input.shape}...")
    traced_model = torch.jit.trace(model, dummy_input)

    # 5. Save the TorchScript model
    torch.jit.save(traced_model, pt_path)
    print(f"Model successfully saved to {pt_path}")

    # Optional: Verify loading the .pt model
    # loaded_model = torch.jit.load(pt_path)
    # print("Model loaded from .pt file for verification.")
    # output = loaded_model(dummy_input)
    # print(f"Output from loaded .pt model: {output.shape}")