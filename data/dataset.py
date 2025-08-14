import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset

from utils.plotting import render_plot_to_tensor
from utils.equations import FUNC_TYPES, generate_equation_string, sample_polynomial_params

class EquationDataset(Dataset):
    def __init__(self, num_samples=15000, image_size=224, split='train'):
        self.num_samples = int(num_samples)
        self.image_size = image_size
        self.split = split
        self.function_types = FUNC_TYPES

        # Define transform
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),  # Reduced augmentation
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Balance the dataset: rotate through function types
        func_type = self.function_types[idx % len(FUNC_TYPES)]
        func_idx = self.function_types.index(func_type)

        # Generate function parameters
        if func_type == 'polynomial':
            # sample_polynomial_params returns ascending order [a0, a1, ..., a6]
            params_ascending = sample_polynomial_params(max_deg=6)
            params = params_ascending
            
            # DEBUG: Print first few samples to verify
            #if idx < 5:
                #print(f"Sample {idx}: params = {params}, equation = {generate_equation_string(func_type, params)}")

        equation = generate_equation_string(func_type, params)

        # Generate image on-the-fly
        img = render_plot_to_tensor(func_type, params)
        img_tensor = self.transform(img)

        # Save sample images less frequently for debugging
        if idx % 500 == 0:
            try:
                root_dir = os.path.dirname(os.path.dirname(__file__))  
                output_dir = os.path.join(root_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"sample.png")
                img.save(save_path)
            except Exception as e:
                print("ERROR: ", str(e))

        return {
            'image': img_tensor,
            'equation': equation,
            'type_idx': func_idx,
            'parameters': torch.tensor(params, dtype=torch.float32)
        }