import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
from PIL import Image
import glob
import os
from model import NonexpansiveDenoiser

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = NonexpansiveDenoiser(radius=7, hidden_channels=16, num_layers=2)
    weights_path = 'weights/denoiser_unified.pth'
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    
    # Key mapping
    key_mapping = {
        'weight_net.conv1.weight': 'weight_net.net.0.weight',
        'weight_net.conv1.bias': 'weight_net.net.0.bias',
        'weight_net.conv2.weight': 'weight_net.net.2.weight',
        'weight_net.conv2.bias': 'weight_net.net.2.bias',
    }
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in key_mapping:
            new_state_dict[key_mapping[k]] = v
        elif k not in ['inv_dy_cuda', 'inv_dx_cuda']:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # Use local test images
    test_dir = "test_images"
    paths = [
        os.path.join(test_dir, "clean_aeroplane.png"),
        os.path.join(test_dir, "clean_tree.png"),
        os.path.join(test_dir, "clean_building.png")
    ]

    clean_tensors = []
    noisy_tensors = []
    
    noise_sigma = 25.0 / 255.0

    for path in paths:
        img = Image.open(path).convert("RGB")
        # Resize/crop to 256x256
        img = TF.center_crop(TF.resize(img, 256), 256)
        clean = TF.to_tensor(img).to(device)
        
        noise = torch.randn_like(clean) * noise_sigma
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        
        clean_tensors.append(clean)
        noisy_tensors.append(noisy)
        
    clean_batch = torch.stack(clean_tensors)
    noisy_batch = torch.stack(noisy_tensors)
    
    with torch.no_grad():
        denoised_batch = model(noisy_batch)
        
    # Make grids
    clean_grid = make_grid(clean_batch, nrow=3, padding=4, pad_value=1.0)
    noisy_grid = make_grid(noisy_batch, nrow=3, padding=4, pad_value=1.0)
    denoised_grid = make_grid(denoised_batch, nrow=3, padding=4, pad_value=1.0)
    
    os.makedirs("assets", exist_ok=True)
    save_image(noisy_grid, "assets/corrupted_input.png")
    save_image(denoised_grid, "assets/denoised_output.png")
    save_image(clean_grid, "assets/perfect_target.png")
    print("Successfully replaced assets with real images!")

if __name__ == "__main__":
    main()
