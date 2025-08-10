import matplotlib.pyplot as plt
import io
from PIL import Image

def plot_points(points):
    fig, ax = plt.subplots(figsize=(4, 4))
    
    xs, ys = zip(*points)  # unzip into two lists
    ax.plot(xs, ys, color="darkblue", linewidth=2.5)
    
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.patch.set_facecolor("white")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0.1)
    buf.seek(0)
    
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    
    return img