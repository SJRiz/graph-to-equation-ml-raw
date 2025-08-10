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
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        # Up to degree 6
        if func_type == 'polynomial':
            params = sample_polynomial_params(max_deg=6)[::-1]

        equation = generate_equation_string(func_type, params)

        # Generate image on-the-fly
        img = render_plot_to_tensor(func_type, params)
        img_tensor = self.transform(img)

        if idx % 3 == 0:
            # Go up from /data to root
            root_dir = os.path.dirname(os.path.dirname(__file__))  
            output_dir = os.path.join(root_dir, "output")
            os.makedirs(output_dir, exist_ok=True)  # make sure the folder exists

            save_path = os.path.join(output_dir, "sample.png")
            img.save(save_path)

        return {
            'image': img_tensor,
            'equation': equation,
            'type_idx': func_idx,
            'parameters': torch.tensor(params, dtype=torch.float32)
        }
