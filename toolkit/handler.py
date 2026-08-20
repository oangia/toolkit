import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from sklearn.cluster import KMeans

class ImageHandler:
    def __init__(self):
        pass

    # 1. LOADING & SAVING
    def load(self, path, size=None, normalize=True):
        """Loads and optionally resizes an image."""
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        if normalize:
            arr /= 255.0
        return arr

    def save(self, arr, path):
        """Saves a numpy image array to disk."""
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        img.save(path)
        return path

    # 2. BASIC TRANSFORMS (Resize, Crop)
    def resize(self, arr, size):
        """Resizes an image array given a (width, height) tuple."""
        is_norm = arr.max() <= 1.0
        img = Image.fromarray((arr * 255).astype(np.uint8) if is_norm else arr.astype(np.uint8))
        img_res = img.resize(size, Image.Resampling.LANCZOS)
        out = np.array(img_res, dtype=np.float32)
        return out / 255.0 if is_norm else out

    def crop(self, arr, box):
        """Crops an image array given a box tuple: (left, upper, right, lower)."""
        is_norm = arr.max() <= 1.0
        img = Image.fromarray((arr * 255).astype(np.uint8) if is_norm else arr.astype(np.uint8))
        img_crop = img.crop(box)
        out = np.array(img_crop, dtype=np.float32)
        return out / 255.0 if is_norm else out

    # 3. EDGE EXTRACTION (from your EdgeLoss logic)
    def extract_edges(self, arr):
        """Extracts Sobel edge magnitude map from an image array [H, W, C]."""
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        
        gx = convolve2d(gray, kx, mode='same', boundary='symm')
        gy = convolve2d(gray, ky, mode='same', boundary='symm')
        edge_mag = np.sqrt(gx**2 + gy**2 + 1e-6)
        
        max_val = edge_mag.max()
        if max_val > 0:
            edge_mag /= max_val
            
        return np.stack([edge_mag]*3, axis=-1)

    # 4. COLOR GROUPING / QUANTIZATION (from show_color_groups)
    def group_colors(self, arr, bins=8):
        """Reduces smooth gradients into distinct color steps based on bins."""
        is_norm = arr.max() <= 1.0
        working_arr = arr if is_norm else arr / 255.0
        grouped = np.round(working_arr * bins) / bins
        return grouped if is_norm else (grouped * 255.0)

    # 5. COLOR HISTOGRAM (from ColorHistogramLoss)
    def get_histogram(self, arr, bins=32):
        """Calculates color channel histograms for an image array."""
        h, w, c = arr.shape
        pixels = arr.reshape(c, -1) # Shape: [C, H*W]
        
        hist_list = []
        for i in range(bins):
            center = i / (bins - 1)
            # Triangular soft bin distance
            dist = np.maximum(0.0, 1.0 - np.abs(pixels - center) * bins)
            hist_list.append(dist.sum(axis=1))
            
        hist = np.stack(hist_list, axis=1) # Shape: [C, bins]
        return hist / (h * w) # Normalize to probabilities

    # 6. K-MEANS CLUSTERING & SHARED PALETTES (from show_shared_clusters)
    def cluster_colors(self, arr, n_clusters=8):
        """Clusters an image's colors using K-Means."""
        h, w, c = arr.shape
        pixels = arr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[kmeans.labels_].reshape(h, w, c)

    def shared_cluster_mapping(self, arr1, arr2, n_clusters=8):
        """Computes a single unified K-Means palette from two images and maps both to it."""
        h1, w1, _ = arr1.shape
        h2, w2, _ = arr2.shape
        
        p1 = arr1.reshape(-1, 3)
        p2 = arr2.reshape(-1, 3)
        
        combined = np.vstack([p1, p2])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(combined)
        shared_colors = kmeans.cluster_centers_
        
        def map_pixels(pixels, h, w):
            distances = np.linalg.norm(pixels[:, None, :] - shared_colors[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            return shared_colors[labels].reshape(h, w, 3)
            
        return map_pixels(p1, h1, w1), map_pixels(p2, h2, w2)