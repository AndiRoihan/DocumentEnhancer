import cv2
import numpy as np
from tkinter import Tk, filedialog
import warnings
import matplotlib.pyplot as plt
from jiwer import wer, cer

warnings.filterwarnings('ignore')

class DocumentEnhancer:
    def __init__(self, image_path):
        # Load the image or raise error
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image: {image_path}")
        # Prepare storage for intermediate steps
        self.history = [('Original', self.original.copy())]
        self.processed = self.original.copy()

    def detect_and_warp(self):
        gray = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)
        self.history.append(('Grayscale', gray))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.history.append(('Blurred', blurred))

        edges = cv2.Canny(blurred, 50, 150)
        self.history.append(('Edges', edges))

        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        self.history.append(('Closed', closed))

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 10000:
                pts = approx.reshape(4, 2)
                break
        else:
            h, w = gray.shape
            pts = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)

        ys = pts[:,1].argsort()
        top, bottom = pts[ys[:2]], pts[ys[2:]]
        tl, tr = top[np.argsort(top[:,0])]
        bl, br = bottom[np.argsort(bottom[:,0])]
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        width = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
        height = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
        dst = np.array([[0,0],[width,0],[width,height],[0,height]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.processed, M, (width, height))
        self.history.append(('Warped', warped))
        self.processed = warped

    def enhance(self):
        gray = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)
        self.history.append(('Warped Grayscale', gray))

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=10
        )
        self.history.append(('Adaptive Threshold', adaptive))

        final = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        self.history.append(('Final', final))
        self.processed = final

    def evaluate(self):
        # Convert to grayscale for metrics
        gray_proc = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)

        # Contrast-to-Noise Ratio (CNR)
        # Binary mask from adaptive threshold
        _, binary = cv2.threshold(gray_proc, 127, 255, cv2.THRESH_BINARY)
        text_mask = binary == 0
        bg_mask = binary == 255
        mu_text = gray_proc[text_mask].mean() if np.any(text_mask) else 0
        mu_bg = gray_proc[bg_mask].mean() if np.any(bg_mask) else 0
        sigma_text = gray_proc[text_mask].std() if np.any(text_mask) else 1
        sigma_bg = gray_proc[bg_mask].std() if np.any(bg_mask) else 1
        cnr = abs(mu_bg - mu_text) / (sigma_bg + sigma_text)
        print(f"CNR: {cnr:.4f}")

        # Edge density
        edges = cv2.Canny(gray_proc, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        print(f"Edge Density: {edge_density:.4f}")

    def process(self):
        print("Applying perspective correction...")
        self.detect_and_warp()
        print("Applying adaptive threshold enhancement...")
        self.enhance()
        print("Evaluating metrics...")
        self.evaluate()
        print("Processing complete.")

    def show_results(self):
        # Grid display of all steps
        n = len(self.history)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, self.history):
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')

        for ax in axes[len(self.history):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Separate figure: Input and Final Output
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
        ax2[0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        ax2[0].set_title('Input Image')
        ax2[0].axis('off')
        ax2[1].imshow(cv2.cvtColor(self.processed, cv2.COLOR_BGR2RGB))
        ax2[1].set_title('Final Output')
        ax2[1].axis('off')

        plt.tight_layout()
        plt.show()

    def save(self, path):
        cv2.imwrite(path, self.processed)
        print(f"Saved enhanced image to {path}")

if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select document image",
        filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")] )
    if not path:
        print("No file chosen, exiting.")
    else:
        enhancer = DocumentEnhancer(path)
        enhancer.process()
        enhancer.show_results()
        enhancer.save('output/output_final.jpg')
