"""
model_manager.py - PyTorch MobileNetV2 per-component classifier + Grad-CAM
Multi-Component LHB Bogie Diagnostic System (RAIL-SENSE v2.0)

This module handles the training, loading, and inference of PyTorch MobileNetV2 models 
for different components of LHB bogies. It also includes Grad-CAM for visualizing 
defect regions and OpenCV features for rule-based secondary analysis.
"""

# Import standard library for interacting with the operating system
import os
# Import math module for mathematical operations like infinity
import math
# Import numpy for numerical array manipulations
import numpy as np
# Import core PyTorch library
import torch
# Import PyTorch neural network modules
import torch.nn as nn
# Import PyTorch optimization algorithms
import torch.optim as optim
# Import PyTorch dataset and data loader utilities for batching and sampling
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
# Import torchvision transforms for image preprocessing and augmentation
import torchvision.transforms as T
# Import torchvision pre-trained models (like MobileNetV2)
import torchvision.models as models
# Import OpenCV for traditional computer vision and image processing operations
import cv2
# Import PIL Image for loading and handling image data
from PIL import Image

# Read the base dataset directory from environment variables, defaulting to current directory
DATASET_ROOT = os.getenv("DATASET_PATH", ".")
# Read the model directory from environment variables, parsing the directory from the file path
MODEL_DIR    = os.path.dirname(os.getenv("MODEL_PATH", "backend/coupler_model.pth"))
# Define the standard image size for MobileNetV2 (224x224 pixels)
IMG_SIZE     = 224
# Set device to GPU (cuda) if available for faster computation, else fallback to CPU
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Component Registry
# ─────────────────────────────────────────────────────────────
# Registry mapping each bogie component to its configuration parameters, 
# model file, folder name, and computer vision thresholds.
COMPONENT_REGISTRY = {
    # Configuration for the CBC Coupler
    "coupler": {
        "folder":     "cbc_image",               # Folder containing coupler dataset
        "model_file": "coupler_model.pth",       # Filename for saving/loading model weights
        "label":      "CBC Coupler",             # Human-readable label
        "cv_profile": {                          # OpenCV feature extraction thresholds
            "rust_lower": [5, 50, 50],           # Lower HSV bounds for detecting rust
            "rust_upper": [20, 255, 200],        # Upper HSV bounds for detecting rust
            "edge_align_threshold": 0.12,        # Threshold to determine structural alignment based on edge density
            "anomaly_mode": "edges",             # Target anomaly type to focus on in traditional CV
        },
    },
    # Configuration for the Axle Box
    "axle_box": {
        "folder":     "axle_box",
        "model_file": "axle_box_model.pth",
        "label":      "Axle Box",
        "cv_profile": {
            "rust_lower": [5, 40, 40], "rust_upper": [25, 255, 180],
            "edge_align_threshold": 0.10,
            "anomaly_mode": "oil_stain",         # Focus on oil stains for axle box
        },
    },
    # Configuration for the Brake Disk
    "brake_disk": {
        "folder":     "brake_disk",
        "model_file": "brake_disk_model.pth",
        "label":      "Brake Disk",
        "cv_profile": {
            "rust_lower": [0, 60, 30], "rust_upper": [18, 255, 180],
            "edge_align_threshold": 0.18,
            "anomaly_mode": "groove_density",    # Focus on wear/groove density for brake disks
        },
    },
    # Configuration for the Damper
    "damper": {
        "folder":     "damper",
        "model_file": "damper_model.pth",
        "label":      "Damper",
        "cv_profile": {
            "rust_lower": [5, 50, 30], "rust_upper": [22, 255, 160],
            "edge_align_threshold": 0.09,
            "anomaly_mode": "oil_stain",         # Dampers are prone to oil leaks
        },
    },
    # Configuration for the Coil Spring
    "spring": {
        "folder":     "spring",
        "model_file": "spring_model.pth",
        "label":      "Coil Spring",
        "cv_profile": {
            "rust_lower": [5, 50, 50], "rust_upper": [20, 255, 200],
            "edge_align_threshold": 0.08,
            "anomaly_mode": "edges",             # Spring cracks/breaks show up as edge anomalies
        },
    },
    # Configuration for the Wheel
    "wheel": {
        "folder":     "wheel",
        "model_file": "wheel_model.pth",
        "label":      "Wheel",
        "cv_profile": {
            "rust_lower": [0, 40, 30], "rust_upper": [20, 255, 160],
            "edge_align_threshold": 0.15,
            "anomaly_mode": "groove_density",    # Focus on wheel tread wear and grooves
        },
    },
}

# Global dictionary to cache loaded PyTorch models in memory, keyed by component name
_model_cache: dict = {}


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class ComponentDataset(Dataset):
    """
    A custom PyTorch Dataset class designed to load images from 
    'normal' and 'defect' subdirectories for binary classification.
    """
    # Define class mapping: normal is 0, defect is 1
    CLASSES = {"normal": 0, "defect": 1}

    def __init__(self, root: str, transform=None):
        # Initialize an empty list to store image paths and their corresponding labels
        self.samples   = []
        # Store the torchvision transformations to be applied to the images
        self.transform = transform
        # Iterate over the defined classes and their integer labels
        for label, idx in self.CLASSES.items():
            # Construct the path to the class folder (e.g., .../normal or .../defect)
            folder = os.path.join(root, label)
            # Skip if the directory does not exist for this component
            if not os.path.isdir(folder):
                continue
            # Iterate through all files in the class folder in alphabetical order
            for fname in sorted(os.listdir(folder)):
                # Only include valid image file extensions
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    # Append a tuple of (full_image_path, class_index) to the samples list
                    self.samples.append((os.path.join(folder, fname), idx))

    def __len__(self):
        # Return the total number of samples collected in the dataset
        return len(self.samples)

    def __getitem__(self, i):
        # Retrieve the image path and label for the given index
        path, label = self.samples[i]
        # Open the image using PIL and convert to RGB to ensure 3 color channels
        img = Image.open(path).convert("RGB")
        # Apply the provided transformations (resizing, augmentation, normalization, to tensor)
        if self.transform:
            img = self.transform(img)
        # Return the transformed image tensor and its ground truth label
        return img, label


# ─────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    """
    Constructs and returns a MobileNetV2 model tailored for binary classification.
    """
    # Load a pretrained MobileNetV2 model using ImageNet weights for transfer learning
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Get the number of input features to the original classifier layer
    in_features = model.classifier[1].in_features
    # Replace the default ImageNet classifier (1000 classes) with a custom sequential block
    model.classifier = nn.Sequential(
        # Add dropout with 30% probability to prevent overfitting
        nn.Dropout(0.3),
        # Add a final linear layer for binary classification (2 output nodes)
        nn.Linear(in_features, 2),
    )
    # Move the instantiated model to the target device (GPU or CPU)
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_model(component: str = "coupler", epochs: int = 30, progress_cb=None):
    """
    Trains a MobileNetV2 model for a specific bogie component using transfer learning.
    """
    # Retrieve the configuration for the specified component
    cfg = COMPONENT_REGISTRY.get(component)
    # Throw an error if the component is not defined in our registry
    if not cfg:
        raise ValueError(f"Unknown component: {component}")

    # Construct the full path to the dataset folder for this component
    dataset_path = os.path.join(DATASET_ROOT, cfg["folder"])
    # Construct the full path where the trained model weights will be saved
    model_path   = os.path.join(MODEL_DIR, cfg["model_file"])

    # Define a sequence of image transformations for training (Data Augmentation)
    train_tf = T.Compose([
        # Resize images to the standard 224x224 expected by MobileNetV2
        T.Resize((IMG_SIZE, IMG_SIZE)),
        # Randomly flip images horizontally
        T.RandomHorizontalFlip(),
        # Randomly flip images vertically
        T.RandomVerticalFlip(),
        # Randomly rotate the image by up to 20 degrees
        T.RandomRotation(20),
        # Randomly adjust brightness, contrast, and saturation to increase robustness
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        # Randomly translate the image by up to 10% in both directions
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # Convert the PIL image to a PyTorch float tensor
        T.ToTensor(),
        # Normalize the tensor with ImageNet mean and standard deviation
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Instantiate the custom dataset using the specified path and transforms
    dataset = ComponentDataset(dataset_path, transform=train_tf)
    # Ensure there's at least one sample to train on
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {dataset_path}")

    # Extract all labels from the dataset to compute class distribution
    labels      = [s[1] for s in dataset.samples]
    # Count occurrences of class 0 (normal) and class 1 (defect)
    class_count = [labels.count(0), labels.count(1)]
    # Calculate sampling weights inversely proportional to class frequencies to handle class imbalance
    weights     = [1.0 / class_count[l] for l in labels]
    # Create a sampler that samples data based on the computed weights, ensuring balanced batches
    sampler     = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    # Create a DataLoader to feed data in batches of 16 using the weighted sampler
    loader      = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=0)

    # Initialize a new model using our builder function
    model     = build_model()
    # Define the loss function (Cross Entropy Loss for classification)
    criterion = nn.CrossEntropyLoss()
    # Use AdamW optimizer with a learning rate of 1e-4 and weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Set up a Cosine Annealing learning rate scheduler that gradually decreases the LR over the epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Initialize best_loss to infinity to track the best performing epoch
    best_loss = math.inf
    # Ensure the directory for saving the model exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Begin the training loop over the specified number of epochs
    for epoch in range(1, epochs + 1):
        # Set the model to training mode (enables dropout, batchnorm updates, etc.)
        model.train()
        # Track cumulative loss and total correct predictions for this epoch
        running_loss, correct = 0.0, 0
        # Iterate over batches of images and labels provided by the DataLoader
        for imgs, labels_batch in loader:
            # Move the input batch to the target device (GPU or CPU)
            imgs, labels_batch = imgs.to(DEVICE), labels_batch.to(DEVICE)
            # Clear previously calculated gradients before computing new ones
            optimizer.zero_grad()
            # Perform a forward pass through the network to get raw class scores (logits)
            out  = model(imgs)
            # Compute the batch loss between predictions and actual labels
            loss = criterion(out, labels_batch)
            # Perform a backward pass to compute gradients of the loss with respect to model weights
            loss.backward()
            # Update the model weights using the calculated gradients
            optimizer.step()
            # Accumulate the total loss (multiply batch average loss by batch size)
            running_loss += loss.item() * imgs.size(0)
            # Count how many predictions in the batch match the ground truth labels
            correct += (out.argmax(1) == labels_batch).sum().item()
        
        # Step the learning rate scheduler after each epoch
        scheduler.step()

        # Compute average loss over the entire dataset for this epoch
        epoch_loss = running_loss / len(dataset)
        # Compute overall accuracy as a percentage
        accuracy   = correct / len(dataset) * 100
        # Construct an informational message logging the current epoch's metrics
        msg = f"[{cfg['label']}] Epoch {epoch}/{epochs}  loss={epoch_loss:.4f}  acc={accuracy:.1f}%"
        # Print the progress to standard output
        print(msg)
        # If a callback function is provided (e.g., for updating a UI), invoke it
        if progress_cb:
            progress_cb(msg)

        # Check if the model improved (achieved a lower training loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save the model's state dictionary (weights) to the designated file path
            torch.save(model.state_dict(), model_path)

    # After training, remove the potentially outdated model from the memory cache
    _model_cache.pop(component, None)
    # Print a final summary indicating completion and the best achieved loss
    print(f"[Model] Training complete. Best loss: {best_loss:.4f}. Saved to {model_path}")
    # Return the best loss achieved during training
    return best_loss


# ─────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────
def load_model(component: str = "coupler") -> nn.Module:
    """
    Loads and caches the model for a given component for inference.
    """
    # Return the model immediately if it's already in the global cache
    if component in _model_cache:
        return _model_cache[component]

    # Fetch configuration for the component, defaulting to "coupler" if not found
    cfg        = COMPONENT_REGISTRY.get(component, COMPONENT_REGISTRY["coupler"])
    # Construct the path to the saved model weights
    model_path = os.path.join(MODEL_DIR, cfg["model_file"])
    # Build the base MobileNetV2 architecture with our custom classifier head
    model      = build_model()

    # Check if a previously trained weights file exists
    if os.path.exists(model_path):
        # Load the saved state dict, mapping it to the active device (CPU or GPU)
        state = torch.load(model_path, map_location=DEVICE)
        # Apply the loaded weights to the model instance
        model.load_state_dict(state)
        print(f"[Model] Loaded {component} weights from {model_path}")
    else:
        # Warn if no trained weights exist, meaning it will fall back to ImageNet feature extractor + random head
        print(f"[Model] WARNING: No weights for {component}. Using ImageNet pretrained only.")

    # Set the model to evaluation mode (disables dropout, fixes batchnorm stats) for deterministic inference
    model.eval()
    # Save the loaded model in the cache to speed up subsequent requests
    _model_cache[component] = model
    # Return the ready-to-use model
    return model


def get_model_status() -> dict:
    """
    Checks the registry and file system to return the status of all component models.
    """
    # Initialize an empty dictionary to hold the status report
    status = {}
    # Iterate through all configured components
    for comp, cfg in COMPONENT_REGISTRY.items():
        # Build the full path to the component's expected model file
        path = os.path.join(MODEL_DIR, cfg["model_file"])
        # Record the component's label, whether its model file actually exists, and its path
        status[comp] = {
            "label":  cfg["label"],
            "exists": os.path.exists(path),
            "path":   path,
        }
    # Return the complete status dictionary
    return status


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    Applies standard transformations to prepare a single PIL image for inference.
    """
    # Define a sequence of inference-time transformations
    tf = T.Compose([
        # Resize to match the model's expected input dimension (224x224)
        T.Resize((IMG_SIZE, IMG_SIZE)),
        # Convert PIL image to PyTorch tensor format (scaling pixel values to [0.0, 1.0])
        T.ToTensor(),
        # Normalize using standard ImageNet mean and standard deviation
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Apply transforms, add a batch dimension (B, C, H, W), and move to the target device
    return tf(pil_img).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────
def _grad_cam_bbox(
    model: nn.Module, tensor: torch.Tensor, orig_w: int, orig_h: int
) -> dict | None:
    """
    Generates a bounding box around the most salient "defect" features 
    using Gradient-weighted Class Activation Mapping (Grad-CAM).
    """
    # List to capture gradients from the backward pass
    gradients:   list = []
    # List to capture activation feature maps from the forward pass
    activations: list = []
    # Identify the final convolutional feature layer of MobileNetV2
    target_layer = model.features[-1]

    # Register a forward hook on the target layer to intercept and save its output (activations)
    fwd_h = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o.detach())
    )
    # Register a backward hook to intercept and save the gradients flowing back through this layer
    bwd_h = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach())
    )

    try:
        # Perform a forward pass with the input tensor to get logits
        out = model(tensor)
        # If the model predicts 'normal' (class 0), return None as there's no defect to highlight
        if out.argmax(1).item() == 0:
            return None

        # Extract the specific raw score (logit) for the 'defect' class (index 1)
        score = out[0, 1]
        # Zero out any existing gradients in the model
        model.zero_grad()
        # Backpropagate specifically from the 'defect' score to calculate gradients w.r.t the feature maps
        score.backward()

        # Retrieve the captured gradients from the target layer
        grads   = gradients[0]
        # Retrieve the captured activation maps from the target layer
        acts    = activations[0]
        # Calculate the channel-wise mean of the gradients, which acts as importance weights for each feature map
        weights = grads.mean(dim=(2, 3), keepdim=True)
        # Multiply the activation maps by their importance weights and sum across channels to create the CAM
        cam     = (weights * acts).sum(dim=1).squeeze()
        # Apply ReLU to only keep features that have a positive influence on the 'defect' class
        cam     = torch.relu(cam).cpu().numpy()

        # If all CAM values are zero (no positive influence found), return None
        if cam.max() == 0:
            return None

        # Normalize the CAM values to be between 0 and 1
        cam     = cam / cam.max()
        # Scale values to 0-255 range and convert to unsigned 8-bit integers for OpenCV compatibility
        cam_u8  = (cam * 255).astype(np.uint8)
        # Resize the low-resolution CAM heatmap back to the original image dimensions
        cam_rs  = cv2.resize(cam_u8, (orig_w, orig_h))

        # Apply a binary threshold at 40% of max intensity to isolate the most active regions
        _, thresh = cv2.threshold(cam_rs, int(255 * 0.4), 255, cv2.THRESH_BINARY)
        # Find continuous boundaries (contours) around the highlighted regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Return None if no distinct contours were found
        if not contours:
            return None

        # Identify the largest contour by area and compute its bounding rectangle (x, y, width, height)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Calculate a 5% padding for the bounding box based on width to encompass the region generously
        pad_x = int(w * 0.05)
        # Calculate a 5% padding for the bounding box based on height
        pad_y = int(h * 0.05)
        
        # Return the bounding box coordinates, clamping them to ensure they stay within image bounds
        return {
            "xmin": max(0, x - pad_x),
            "ymin": max(0, y - pad_y),
            "xmax": min(orig_w, x + w + pad_x),
            "ymax": min(orig_h, y + h + pad_y),
        }
    finally:
        # Clean up hooks to prevent memory leaks and unintended interference with future runs
        fwd_h.remove()
        bwd_h.remove()


# ─────────────────────────────────────────────────────────────
# Per-component OpenCV feature extraction
# ─────────────────────────────────────────────────────────────
def _opencv_features(pil_img: Image.Image, component: str = "coupler") -> dict:
    """
    Computes traditional image features (rust percentage, edge density, oil stains)
    using rule-based OpenCV techniques based on component-specific profiles.
    """
    # Retrieve configuration rules for the given component, fallback to coupler
    cfg     = COMPONENT_REGISTRY.get(component, COMPONENT_REGISTRY["coupler"])
    # Get the specific computer vision bounds and thresholds from the config
    profile = cfg["cv_profile"]
    
    # Convert PIL Image (RGB format) to numpy array, then to BGR format for OpenCV
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # Convert BGR image to HSV color space, which is better for color isolation (like rust)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Convert the configured lower rust HSV bounds to a numpy array
    lower_rust = np.array(profile["rust_lower"])
    # Convert the configured upper rust HSV bounds to a numpy array
    upper_rust = np.array(profile["rust_upper"])
    # Create a binary mask isolating pixels falling within the rust color range
    rust_mask  = cv2.inRange(hsv, lower_rust, upper_rust)
    # Calculate the percentage of rust pixels relative to the total number of pixels in the image
    rust_level = float(rust_mask.sum()) / (img_bgr.shape[0] * img_bgr.shape[1] * 255)

    # Convert the original BGR image to grayscale for edge detection
    gray         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur to reduce noise and spurious edges
    blurred      = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform Canny edge detection with hysteresis thresholds 50 and 150
    edges        = cv2.Canny(blurred, 50, 150)
    # Calculate the ratio of edge pixels to the total image area
    edge_density = float(edges.sum()) / (edges.shape[0] * edges.shape[1] * 255)

    # Initialize the oil stain level to zero by default
    oil_level = 0.0
    # If the component's focus anomaly is an oil stain, run specific detection
    if profile["anomaly_mode"] == "oil_stain":
        # Create a mask for dark, very low saturation colors (typical of old, dark oil/grease)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 80, 60]))
        # Compute the proportion of oil stain pixels over the entire image
        oil_level = float(dark_mask.sum()) / (img_bgr.shape[0] * img_bgr.shape[1] * 255)

    # Return a dictionary of the calculated metrics, rounding values to percentages
    return {
        "rust_level":   round(rust_level * 100, 2),                   # Convert rust ratio to a percentage
        "edge_density": round(edge_density * 100, 2),                 # Convert edge ratio to a percentage
        "oil_level":    round(oil_level * 100, 2),                    # Convert oil ratio to a percentage
        # Determine if the structural alignment is okay based on whether edge density is below the defined threshold
        "alignment_ok": edge_density < profile["edge_align_threshold"],
    }


def _generate_mock_wheel_features(defect_score: float, seed_val: int) -> dict:
    """
    Helper function to generate simulated but realistic and deterministic 
    RDSO-compliant wheel parameters based on the image's seed.
    """
    import random
    # Use deterministic random generator based on image seed
    rng = random.Random(seed_val)
    
    # Standard nominal healthy limits (RDSO compliant)
    fh = round(rng.uniform(28.5, 31.8), 1)     # 28.5 to 32 is GOOD
    ft = round(rng.uniform(25.5, 29.4), 1)     # >= 25 is GOOD
    qr = round(rng.uniform(6.8, 8.5), 1)       # > 6.5 is GOOD
    sharp = "No sharpness detected"
    thin = "No thin flange detected"
    hollow = round(rng.uniform(0.5, 2.5), 1)   # <= 5.0 is GOOD
    shelling = "0%"
    spalling = "None"
    wd = round(rng.uniform(885.0, 915.0), 1)   # > 880 is GOOD
    wd_diff = round(rng.uniform(0.1, 0.4), 1)  # <= 0.5 is GOOD
    w_flat_len = round(rng.uniform(0, 15), 1)  # 0-20 is GOOD
    w_flat_dep = round(rng.uniform(0, 0.8), 1)  # 0-1 is GOOD
    thermal_crack = "None"
    rim_crack = "None"
    web_crack = "None"
    hub_crack = "None"
    
    # If defect score indicates a warning or unfit condition, generate anomalous values
    if defect_score > 50:
        # Pick one or two parameters to fail deterministically
        fail_mode = rng.randint(0, 5)
        if fail_mode == 0:
            # Flange Height critical/warning
            fh = round(rng.uniform(32.5, 36.2), 1)
        elif fail_mode == 1:
            # Flange Thickness critical/warning
            ft = round(rng.uniform(20.5, 24.8), 1)
        elif fail_mode == 2:
            # High Flat wheel
            w_flat_len = round(rng.uniform(25, 62), 1)
            w_flat_dep = round(rng.uniform(1.2, 2.8), 1)
        elif fail_mode == 3:
            # Cracks
            thermal_crack = f"{round(rng.uniform(5, 15), 1)} mm thermal crack"
            rim_crack = f"{round(rng.uniform(2, 8), 1)} mm rim crack"
        elif fail_mode == 4:
            # Shelling & Spalling
            shelling = f"{round(rng.uniform(6, 22), 1)}% surface area shelling"
            spalling = "Localized spalling detected"
        else:
            # Diameter difference
            wd_diff = round(rng.uniform(0.6, 1.8), 1)
            
    return {
        "flange_height": f"{fh} mm",
        "flange_thickness": f"{ft} mm",
        "qr_value": f"{qr} mm",
        "sharp_flange": sharp,
        "thin_flange": thin,
        "tread_hollow": f"{hollow} mm",
        "shelling": shelling,
        "spalling": spalling,
        "wheel_diameter": f"{wd} mm",
        "wheel_diameter_diff": f"{wd_diff} mm",
        "wheel_flat_length": f"{w_flat_len} mm",
        "wheel_flat_depth": f"{w_flat_dep} mm",
        "thermal_crack": thermal_crack,
        "rim_crack": rim_crack,
        "web_crack": web_crack,
        "hub_crack": hub_crack,
    }


# ─────────────────────────────────────────────────────────────
# Main inference pipeline
# ─────────────────────────────────────────────────────────────
def infer(pil_img: Image.Image, component: str = "coupler") -> dict:
    """
    The main prediction pipeline combining the deep learning classifier, 
    Grad-CAM bounding box localization, and OpenCV heuristics.
    """
    # Load (or retrieve from cache) the specific MobileNetV2 model for the component
    model   = load_model(component)
    # Preprocess the input PIL image into a normalized PyTorch tensor batch
    tensor  = _preprocess(pil_img)
    # Store the original width and height of the image for Grad-CAM bounding box scaling
    orig_w, orig_h = pil_img.size

    # Use torch.no_grad() to disable gradient calculations during inference for speed and memory savings
    with torch.no_grad():
        # Get raw prediction scores (logits) from the model
        logits      = model(tensor)
        # Apply softmax to convert the logits to a normalized probability distribution (summing to 1)
        probs       = torch.softmax(logits, dim=1)[0]
        # Identify the predicted class index (0 for normal, 1 for defect) by finding the highest probability
        pred        = probs.argmax().item()
        # Extract the raw probability for the 'defect' class
        defect_prob = probs[1].item()
        # Extract the raw probability for the 'normal' class
        normal_prob = probs[0].item()

    # Initialize the bounding box result as None
    bbox = None
    # If the model predicts a defect (class 1), generate a Grad-CAM bounding box to locate it
    if pred == 1:
        bbox = _grad_cam_bbox(model, tensor, orig_w, orig_h)

    # Calculate additional traditional computer vision metrics for the specific component
    cv_feats   = _opencv_features(pil_img, component)
    # The overall confidence is simply the highest probability score returned by the model
    confidence = max(defect_prob, normal_prob)

    # Determine final high-level status string based on the defect probability thresholding
    if pred == 1 and defect_prob >= 0.7:
        # High confidence defect maps to UNFIT
        status = "UNFIT"
    elif pred == 1 and defect_prob >= 0.5:
        # Marginal defect confidence maps to MONITOR (warning state)
        status = "MONITOR"
    else:
        # Otherwise, the component is deemed normal/healthy
        status = "FIT"

    # For wheel component, generate mock/simulated feature measurements deterministically
    wheel_features = {}
    if component == "wheel":
        # Create a simple deterministic seed from image pixel intensities
        img_data = np.array(pil_img.resize((16, 16)))
        pixel_seed = int(img_data.sum())
        defect_pct = round(defect_prob * 100, 2)
        wheel_features = _generate_mock_wheel_features(defect_pct, pixel_seed)

    # Assemble and return a comprehensive dictionary of all diagnostic information
    return {
        "status":       status,                            # The high-level decision (FIT, MONITOR, UNFIT)
        "confidence":   round(confidence * 100, 2),        # Confidence of the winning class as a percentage
        "defect_score": round(defect_prob * 100, 2),       # Probability percentage that it's a defect
        "normal_score": round(normal_prob * 100, 2),       # Probability percentage that it's normal
        "bbox":         bbox,                              # Bounding box dict (xmin, ymin, xmax, ymax) or None
        "rust_level":   cv_feats["rust_level"],            # Percent of image classified as rusted
        "edge_density": cv_feats["edge_density"],          # Percent of image classified as hard edges
        "oil_level":    cv_feats["oil_level"],             # Percent of image classified as oil stains
        "alignment_ok": cv_feats["alignment_ok"],          # Boolean flag indicating if alignment/structure looks okay
        "wheel_features": wheel_features,                  # Simulated feature measurements for wheel component
    }


# Back-compat alias
# Expose a default global model path targeting the coupler model for backward compatibility
MODEL_PATH = os.path.join(MODEL_DIR, COMPONENT_REGISTRY["coupler"]["model_file"])
