import io
import base64
import requests
from io import BytesIO
from PIL import Image as PILImage, ImageOps, ImageFilter
from IPython.display import display, HTML
import numpy as np
import random

class BaseImage:    
    def __init__(self, image_source, channels=3):
        self.channels = channels
        self.image_path = image_source
        """Initialize the processor by loading an image from multiple possible source types."""
        try:
            # 1. If it's already a PIL Image object
            if isinstance(image_source, PILImage.Image):
                self.image = image_source
            # 2. If it's raw bytes
            elif isinstance(image_source, bytes):
                self.image = PILImage.open(BytesIO(image_source))

            # 3. If it's a BytesIO stream
            elif isinstance(image_source, BytesIO):
                self.image = PILImage.open(image_source)
            # 4. If it's a NumPy array
            elif isinstance(image_source, np.ndarray):
                self.image = PILImage.fromarray(image_source)
            # 5. If it's a string (local path or URL)
            elif isinstance(image_source, str):
                if image_source.startswith(('http://', 'https://')):
                    response = requests.get(image_source)
                    response.raise_for_status()
                    self.image = PILImage.open(BytesIO(response.content))
                else:
                    self.image = PILImage.open(image_source)
            else:
                raise TypeError(f"Unsupported image source type: {type(image_source)}")
                
        except Exception as e:
            raise ValueError(f"Failed to load image from source: {e}")

        if self.channels == 4:
            self.image = self.image.convert("RGBA")
        
    def resize(self, width: int, height: int, keep_aspect_ratio: bool = True) -> 'BaseImage':
        """Resize the image to specified dimensions."""
        if keep_aspect_ratio:
            self.image.thumbnail((width, height))
        else:
            self.image = self.image.resize((width, height))
        return self

    def to_gray(self) -> 'BaseImage':
        """Convert the image to black and white (grayscale)."""
        self.image = ImageOps.grayscale(self.image)
        return self

    def rotate(self, angle: float) -> 'BaseImage':
        """Rotate the image counter-clockwise by a given angle."""
        self.image = self.image.rotate(angle, expand=True)
        return self

    def crop(self, box: tuple) -> 'BaseImage':
        """Crop the image using a bounding box tuple (left, upper, right, lower)."""
        self.image = self.image.crop(box)
        return self

    def show(self) -> 'BaseImage':
        """Display the current state of the image."""
        display(self.image)
        return self

    def toPil(self):
        return self.image

    def save(self, output_path: str) -> None:
        """Save the processed image to disk."""
        try:
            self.image.save(output_path)
            print(f"Image successfully saved to {output_path}")
        except Exception as e:
            raise IOError(f"Failed to save image: {e}")

class UImage(BaseImage):
    def __init__(self, image_source, channels=3):
        super().__init__(image_source, channels)
        self.x_coords = []
        self.y_coords = []

    def scale(self, factor: float = 2.0, method: str = "random") -> 'AdvancedImage':
        """Scale the image up or down by a factor using a specified or random interpolation algorithm.
        
        - factor > 1.0: Upscales (enlarges) the image.
        - factor < 1.0: Downscales (shrinks) the image.
        """
        methods_map = {
            "nearest": PILImage.Resampling.NEAREST if hasattr(PILImage, 'Resampling') else PILImage.NEAREST,
            "bilinear": PILImage.Resampling.BILINEAR if hasattr(PILImage, 'Resampling') else PILImage.BILINEAR,
            "bicubic": PILImage.Resampling.BICUBIC if hasattr(PILImage, 'Resampling') else PILImage.BICUBIC,
            "lanczos": PILImage.Resampling.LANCZOS if hasattr(PILImage, 'Resampling') else PILImage.LANCZOS,
        }
        
        if method.lower() == "random":
            chosen_key = random.choice(list(methods_map.keys()))
            resample_filter = methods_map[chosen_key]
        else:
            resample_filter = methods_map.get(method.lower(), methods_map["bicubic"])
        
        orig_width, orig_height = self.image.size
        new_width = max(1, int(orig_width * factor))
        new_height = max(1, int(orig_height * factor))
        
        self.image = self.image.resize((new_width, new_height), resample=resample_filter)
        return self

    def crop_center(self, width: int = 512, height: int = 512) -> 'AdvancedImage':
        """Automatically crop a box of the specified size from the center of the image."""
        img_width, img_height = self.image.size

        if img_width < width or img_height < height:
            raise ValueError(f"Image size ({img_width}x{img_height}) is smaller than target crop size ({width}x{height}).")

        left = (img_width - width) // 2
        upper = (img_height - height) // 2
        right = left + width
        lower = upper + height

        self.image = self.image.crop((left, upper, right, lower))
        return self

    def edge(self, grayscale: bool = True) -> 'AdvancedImage':
        """Detects edges in the image using Pillow's FIND_EDGES filter."""
        if grayscale and self.image.mode != 'L':
            self.image = ImageOps.grayscale(self.image)

        self.image = self.image.filter(ImageFilter.FIND_EDGES)
        return self

    def slice_image(self, tile_size: int = 512) -> list:
        """Automatically slices the image into overlapping tiles of `tile_size`, padding with zeros if smaller."""
        img_width, img_height = self.image.size
    
        # Pad the image with zeros if smaller than the tile size
        if img_width < tile_size or img_height < tile_size:
            new_width = max(img_width, tile_size)
            new_height = max(img_height, tile_size)
            
            # Create a new background image filled with 0 and paste the original
            padded_image = PILImage.new(self.image.mode, (new_width, new_height), color=0)
            padded_image.paste(self.image, (0, 0))
            self.image = padded_image
            img_width, img_height = self.image.size
    
        def get_even_coords(size, tile):
            if size <= tile:
                return [0]
    
            travel = size - tile
            ideal_step = tile // 2
    
            num_intervals = round(travel / ideal_step)
            if num_intervals < 1:
                num_intervals = 1
    
            coords = []
            for i in range(num_intervals + 1):
                pos = int(i * travel / num_intervals)
                if not coords or pos != coords[-1]:
                    coords.append(pos)
            return coords
    
        self.x_coords = get_even_coords(img_width, tile_size)
        self.y_coords = get_even_coords(img_height, tile_size)
    
        tiles = []
        for y in self.y_coords:
            for x in self.x_coords:
                box = (x, y, x + tile_size, y + tile_size)
                tile = self.image.crop(box)
                tiles.append(tile)
    
        return tiles

    def get_random_crop(self, tile_size: int = 512):
        """Directly crops a single random tile of `tile_size`, padding with zeros if smaller."""
        img_width, img_height = self.image.size
    
        # Pad the image with zeros if smaller than the tile size
        if img_width < tile_size or img_height < tile_size:
            new_width = max(img_width, tile_size)
            new_height = max(img_height, tile_size)
            
            # Create a new background image filled with 0 and paste the original
            padded_image = PILImage.new(self.image.mode, (new_width, new_height), color=0)
            padded_image.paste(self.image, (0, 0))
            self.image = padded_image
            img_width, img_height = self.image.size
    
        # Pick a random top-left (x, y) position within valid boundaries
        x = random.randint(0, img_width - tile_size)
        y = random.randint(0, img_height - tile_size)
    
        box = (x, y, x + tile_size, y + tile_size)
        return self.image.crop(box)
    def display_grid(self, tiles: list, display_size: int = 128) -> 'AdvancedImage':
        """Display the sliced tiles in a 2D horizontal/vertical grid layout."""
        cols = len(self.x_coords)
        if cols == 0:
            raise ValueError("No coordinates found. Please run slice_image() first.")

        html = f'<div style="display: grid; grid-template-columns: repeat({cols}, max-content); gap: 4px; background: #eee; padding: 4px; width: max-content;">'

        for tile in tiles:
            thumb = tile.copy()
            thumb.thumbnail((display_size, display_size))

            buffered = io.BytesIO()
            thumb.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            html += f'<img src="data:image/jpeg;base64,{img_str}" style="display: block;" />'

        html += '</div>'
        display(HTML(html))
        return self

    def quantize_colors(self, k: int = 8) -> 'AdvancedImage':
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        img_np = np.array(self.image, dtype=np.float64)

        if k <= 1:
            global_palette = np.array([[0, 0, 0]], dtype=np.float64)
        else:
            indices = np.arange(k, dtype=np.float64)
            values = (indices / (k - 1)) * 255.0
            global_palette = np.stack([values, values, values], axis=-1)

        distances = np.sqrt(np.sum((img_np[:, :, None, :] - global_palette[None, None, :, :]) ** 2, axis=3))
        nearest_indices = np.argmin(distances, axis=2)

        quantized_np = global_palette[nearest_indices].astype(np.uint8)

        self.image = PILImage.fromarray(quantized_np)
        return self

    def lower_quality(self):
        w, h = self.image.size    
        scale_factor = 4 #random.randint(8, 16)  
        low_h = max(1, h // scale_factor)
        low_w = max(1, w // scale_factor)
    
        # Map string modes to PIL Resampling filters
        methods_map = {
            "nearest": PILImage.Resampling.NEAREST if hasattr(PILImage, 'Resampling') else PILImage.NEAREST,
            "bilinear": PILImage.Resampling.BILINEAR if hasattr(PILImage, 'Resampling') else PILImage.BILINEAR,
            "bicubic": PILImage.Resampling.BICUBIC if hasattr(PILImage, 'Resampling') else PILImage.BICUBIC,
        }
        
        interp_name = random.choice(list(methods_map.keys()))
        downscale_filter = methods_map[interp_name]
        upscale_filter = PILImage.Resampling.NEAREST if hasattr(PILImage, 'Resampling') else PILImage.NEAREST
    
        # Handle RGBA images separately to bypass premultiplied alpha conversion
        if self.image.mode == "RGBA":
            r, g, b, a = self.image.split()
            rgb_img = PILImage.merge("RGB", (r, g, b))
            
            # 1. Downscale & upscale RGB channels independently
            rgb_low = rgb_img.resize((low_w, low_h), resample=downscale_filter)
            rgb_high = rgb_low.resize((w, h), resample=upscale_filter)
            
            # 2. Downscale & upscale Alpha channel independently
            a_low = a.resize((low_w, low_h), resample=downscale_filter)
            a_high = a_low.resize((w, h), resample=upscale_filter)
            
            # 3. Merge them back into RGBA
            r_fin, g_fin, b_fin = rgb_high.split()
            self.image = PILImage.merge("RGBA", (r_fin, g_fin, b_fin, a_high))
        else:
            # Standard resize for non-RGBA images
            low_img = self.image.resize((low_w, low_h), resample=downscale_filter)
            self.image = low_img.resize((w, h), resample=upscale_filter)
    
        return self
