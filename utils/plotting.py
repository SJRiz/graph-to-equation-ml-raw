import matplotlib
import matplotlib.pyplot as plt
import io
import numpy as np
import random
import os
from utils.equations import generate_equation
from PIL import Image
matplotlib.use('Agg')

def render_plot_to_tensor(func_type, params):
        """Create plot with styling and variety"""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # -5 to +5
        x = np.linspace(-5.0, 5.0, 300)
        
        # Add slight noise to make it more realistic
        noise_level = random.uniform(0, 0.005)
        
        try:
            y = generate_equation(func_type, params, x)
            
            # Add slight noise
            if noise_level > 0:
                y += np.random.normal(0, noise_level * 25, len(y))
            
            # Vary plot style slightly
            colors = ['blue', 'darkblue', 'navy', 'mediumblue']
            line_styles = ['-', '-', '-', '-.']  # Mostly solid
            line_widths = [2, 2.5, 3]
            
            ax.plot(x, y, 
                   color=random.choice(colors),
                   linestyle=random.choice(line_styles),
                   linewidth=random.choice(line_widths))
            
            # FIXED axis limits: 10x10 units
            ax.set_xlim(-5.0, 5.0)
            ax.set_ylim(-5.0, 5.0)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Vary background color slightly
            bg_colors = ['white', 'white', 'white', 'whitesmoke']
            fig.patch.set_facecolor(random.choice(bg_colors))
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            # Fallback
            y = x
            ax.plot(x, y, 'b-', linewidth=2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0.1)
        buf.seek(0)
        
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        
        return img

def create_comparison_graphs(num_samples, guesses):
    _, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(num_samples):
        row = i // 3
        col = i % 3
        
        # Plot comparison
        x = np.linspace(-5, 5, 200)
        
        # unpack
        print(num_samples, i, len(guesses))
        true_params = guesses[i]['true_params']
        pred_params = guesses[i]['pred_params']

        # True function
        y_true = generate_equation("polynomial", true_params, x)
        
        # Predicted function
        y_pred = generate_equation("polynomial", pred_params, x)
        
        axes[row, col].plot(x, y_true, 'g-', linewidth=3, label=f'True')
        axes[row, col].plot(x, y_pred, 'r--', linewidth=2, label=f'Pred')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend()
        
        # Set fixed y-axis limits
        axes[row, col].set_ylim(-5, 5)
        
        param_error = np.mean(np.abs(true_params[:3] - pred_params[:3]))
        axes[row, col].set_title(f'Sample {i+1} - Param Error: {param_error:.3f}')
    
    plt.tight_layout()

    os.makedirs('output', exist_ok=True)
    output_path = os.path.join('output', 'detailed_comparison.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Detailed comparison saved as 'detailed_comparison.png'")
    plt.show()