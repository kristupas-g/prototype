from mmdeploy_runtime import RotatedDetector
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import io
from PIL import Image

class Detector:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device

        self.classes = {
            4: {'name': 'Class 4', 'color': 'blue'},
            5: {'name': 'Class 5', 'color': 'green'}
        }

        self.detector = RotatedDetector(
            model_path = self.model_path,
            device_name = self.device,
            device_id = 0
        )

    def run(self, img):
        return self.detector(img)

    def visualize_results(self, image, bboxes, labels, score_threshold=0.5, thickness=2):
        fig, ax = plt.subplots()
        ax.imshow(image)

        for bbox, label in zip(bboxes, labels):
            xc, yc, w, h, ag, score = bbox
            
            if score >= score_threshold and label in self.classes:
                color = self.classes[label]['color']
                
                wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
                hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
                p1 = (xc - wx - hx, yc - wy - hy)
                p2 = (xc + wx - hx, yc + wy - hy)
                p3 = (xc + wx + hx, yc + wy + hy)
                p4 = (xc - wx + hx, yc - wy + hy)
                poly = np.array([p1, p2, p3, p4], dtype=np.float64)
                
                rect = Polygon(poly, closed=True, facecolor='none', edgecolor=color, linewidth=thickness, alpha=0.7)
                ax.add_patch(rect)
                
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)

        buf.seek(0)  
        plt.close()
        return Image.open(buf)


    def generate_legend():
        fig, ax = plt.subplots()
        for class_id, attributes in self.classes.items():
            ax.plot([], [], color=attributes['color'], label=attributes['name'])
        legend = ax.legend(frameon=False)
        ax.set_axis_off()

        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format='png',
            bbox_inches=legend
                    .get_window_extent()
                    .transformed(fig.dpi_scale_trans.inverted())
                    .expanded(1.2, 1.2),
            transparent=False
        ) 

        plt.close(fig) 
        buffer.seek(0) 

        return buffer.getvalue()