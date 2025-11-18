# =====================
# Local VSC Setup
# =====================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================
# Positional Encoding
# =====================
class PositionalEncoding(nn.Module):
    def __init__(self, L=10, include_input=True):
        super().__init__()
        self.L = L
        self.include_input = include_input
        self.output_dim = 2 + 4 * L if include_input else 4 * L

    def forward(self, x):
        """
        x: (B, 2) normalized coordinates [0, 1]
        returns: (B, 2 + 4*L) encoded coordinates
        """
        B = x.shape[0]
        encoded = [x] if self.include_input else []

        for i in range(self.L):
            freq = 2 ** i
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))

        return torch.cat(encoded, dim=-1)

# =====================
# MLP Network
# =====================
class NeuralField2D(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, L=10):
        super().__init__()
        self.pe = PositionalEncoding(L=L)
        input_dim = self.pe.output_dim

        print(f"Positional Encoding: 2 -> {input_dim} dimensions")
        print(f"MLP input dimension: {input_dim}")

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        encoded_coords = self.pe(coords)
        return self.net(encoded_coords)

# =====================
# DataLoader
# =====================
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, batch_size=10000):
        # Load and normalize image
        self.image = Image.open(image_path).convert('RGB')
        self.img_tensor = transforms.ToTensor()(self.image)  # (3, H, W)
        self.H, self.W = self.img_tensor.shape[1], self.img_tensor.shape[2]

        # Load image and force correct orientation
        self.image = Image.open(image_path).convert('RGB')
        
        # Force transpose to ensure consistent orientation
        # This assumes your images are in landscape orientation
        # if self.image.size[1] < self.image.size[0]:  # If height > width
        #     self.image = self.image.transpose(Image.ROTATE_90)  # Rotate 90 degrees
        
        self.img_tensor = transforms.ToTensor()(self.image)
        self.H, self.W = self.img_tensor.shape[1], self.img_tensor.shape[2]

        print(f"Image loaded: {self.H}x{self.W}")

        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, self.H),
            torch.linspace(0, 1, self.W),
            indexing='ij'
        )
        self.coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
        self.colors = self.img_tensor.permute(1, 2, 0)  # (H, W, 3)

        self.batch_size = batch_size
        self.num_pixels = self.H * self.W

    def __len__(self):
        return self.num_pixels // self.batch_size

    def __getitem__(self, idx):
        # Randomly sample batch_size pixels
        indices = torch.randint(0, self.num_pixels, (self.batch_size,))
        coords_flat = self.coords.reshape(-1, 2)
        colors_flat = self.colors.reshape(-1, 3)

        return coords_flat[indices], colors_flat[indices]

# =====================
# Training Function
# =====================
def train_neural_field(image_path, hidden_dim=256, L=10, num_iters=2000,
                      batch_size=10000, lr=1e-2, save_interval=500):

    # Model and optimizer
    model = NeuralField2D(hidden_dim=hidden_dim, L=L).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data
    dataset = ImageDataset(image_path, batch_size=batch_size)

    # Training
    losses = []
    psnrs = []
    images = []

    model.train()
    for iteration in range(num_iters):
        coords, targets = dataset[0]  # Get random batch
        coords, targets = coords.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Metrics
        mse = loss.item()
        psnr = 10 * torch.log10(1.0 / torch.tensor(mse)).item() if mse > 0 else 0

        losses.append(mse)
        psnrs.append(psnr)

        if iteration % 100 == 0:
            print(f"Iter {iteration:04d}, MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

        # Save reconstruction at intervals
        if iteration % save_interval == 0 or iteration == num_iters - 1:
            with torch.no_grad():
                model.eval()
                # Reconstruct full image in chunks to avoid memory issues
                chunk_size = 10000
                reconstructed = torch.zeros_like(dataset.colors)
                coords_full = dataset.coords.reshape(-1, 2)

                for i in range(0, len(coords_full), chunk_size):
                    end_idx = min(i + chunk_size, len(coords_full))
                    coords_chunk = coords_full[i:end_idx].to(device)
                    reconstructed_chunk = model(coords_chunk)
                    reconstructed.reshape(-1, 3)[i:end_idx] = reconstructed_chunk.cpu()

                images.append((iteration, reconstructed))
                model.train()

    return model, losses, psnrs, images

# =====================
# Visualization
# =====================
def plot_training_progress(images, original_image, config_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Original image
    axes[0].imshow(original_image.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Reconstructions at different iterations
    for idx, (iter_num, recon_img) in enumerate(images[:5]):
        axes[idx+1].imshow(recon_img)
        axes[idx+1].set_title(f'Iteration {iter_num}')
        axes[idx+1].axis('off')

    plt.suptitle(f'Training Progress - {config_name}')
    plt.tight_layout()
    plt.show()

def plot_psnr_curve(psnrs, config_name):
    plt.figure(figsize=(10, 6))
    plt.plot(psnrs)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR Curve - {config_name}')
    plt.grid(True)
    plt.show()

# =====================
# Hyperparameter Experiments
# =====================
def run_experiments(image_path):
    experiments = [
        {'hidden_dim': 128, 'L': 4, 'name': 'Small (128, L=4)'},
        {'hidden_dim': 128, 'L': 10, 'name': 'Small (128, L=10)'},
        {'hidden_dim': 256, 'L': 4, 'name': 'Medium (256, L=4)'},
        {'hidden_dim': 256, 'L': 10, 'name': 'Medium (256, L=10)'},
    ]

    dataset = ImageDataset(image_path)
    original_image = dataset.img_tensor

    results = {}
    for config in experiments:
        print(f"\n=== Running experiment: {config['name']} ===")
        model, losses, psnrs, images = train_neural_field(
            image_path,
            hidden_dim=config['hidden_dim'],
            L=config['L'],
            num_iters=2000,
            batch_size=10000,
            lr=1e-2
        )

        results[config['name']] = {
            'model': model,
            'losses': losses,
            'psnrs': psnrs,
            'images': images
        }

        # Plot training progress
        plot_training_progress(images, original_image, config['name'])

        # Plot PSNR curve
        plot_psnr_curve(psnrs, config['name'])

    return results

# =====================
# Main Execution
# =====================
print("=== Part 1: 2D Neural Field ===")
print("Model Architecture:")
print("- 4-layer MLP with ReLU activations")
print("- Positional Encoding with frequency L")
print("- Sigmoid output for RGB values in [0,1]")
print("- Adam optimizer with lr=1e-2")
print("- Batch size: 10,000 pixels")
print("- Training iterations: 2,000")

# List of images you want to run experiments on
image_paths = ["animal.jpg", "oski_bear.jpg"]

for image_path in image_paths:
    print(f"\n=== Processing {image_path} ===")
    if not os.path.exists(image_path):
        print(f"Warning: Image file '{image_path}' not found!")
        print("Please update the image_path variable to point to your image file.")
        continue

    # Run all experiments for this image
    results = run_experiments(image_path)

    # Show final comparison grid
    print(f"\n=== Final Results Comparison for {image_path} ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    configs = list(results.keys())

    for idx, config_name in enumerate(configs):
        row, col = idx // 2, idx % 2
        final_image = results[config_name]['images'][-1][1]  # Last reconstruction
        axes[row, col].imshow(final_image)
        axes[row, col].set_title(
            f'{config_name}\nFinal PSNR: {results[config_name]["psnrs"][-1]:.2f} dB'
        )
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # Save models (optional)
    # print("\nSaving trained models...")
    # for config_name, result in results.items():
    #     save_name = f'model_{Path(image_path).stem}_{config_name.replace(" ", "_").lower()}.pth'
    #     torch.save(result['model'].state_dict(), save_name)

    print(f"Training complete for {image_path}! Models saved.")
