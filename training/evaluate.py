import os
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

from utils.equations import FUNC_TYPES, generate_equation_string
from utils.plotting import create_comparison_graphs

def evaluate_model(model, dataset, num_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    type_names = FUNC_TYPES
    
    print("\n=== Model Evaluation ===")
    
    param_errors = {t: [] for t in type_names}
    
    # Test on more samples for better statistics
    test_indices = random.sample(range(len(dataset)), min(500, len(dataset)))
    
    for idx in test_indices:
        sample = dataset[idx]
        true_type = type_names[sample['type_idx']]
        
        with torch.no_grad():
            image = sample['image'].unsqueeze(0).to(device)
            param_pred = model(image)
            
            # Parameter error
            true_params = sample['parameters'].numpy()
            pred_params = param_pred.squeeze().cpu().numpy()
            param_error = np.mean(np.abs(true_params - pred_params))
            param_errors[true_type].append(param_error)
    
    # Create visual comparisons
    create_detailed_comparison(model, dataset, num_samples=6)

def predict_from_image(model, image_path):
    """Predict equation from a single image file"""

    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    print(f"\n🔍 Analyzing image: {image_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        param_pred = model(image_tensor)
        predicted_params = param_pred.squeeze().cpu().numpy()
    
    # Generate equation string
    equation = generate_equation_string("polynomial", predicted_params)
    
    return {
        'equation': equation,
        'parameters': predicted_params,
    }
    
def create_detailed_comparison(model, dataset, num_samples=6):
    """Create detailed comparison plots"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    guesses = []
    for _ in range(num_samples):
        with torch.no_grad():
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]

            image = sample['image'].unsqueeze(0).to(device)
            param_pred = model(image)
            
            pred_params = param_pred.squeeze().cpu().numpy()
            
            true_params = sample['parameters'].numpy()

            guesses.append({
                "pred_params": pred_params,
                "true_params": true_params,
            })

    create_comparison_graphs(num_samples, guesses)