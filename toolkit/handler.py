import io
import base64
import requests
from io import BytesIO
from PIL import Image as PILImage, ImageOps
from IPython.display import display, HTML
from PIL import ImageFilter
import numpy as np

class Image:
    def __init__(self, image_source: str):
        """Initialize the processor by loading an image from a local file path or a web URL."""
        try:
            self.image_path = image_source
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                response.raise_for_status()
                self.image = PILImage.open(BytesIO(response.content))
            else:
                self.image = PILImage.open(image_source)
            self.x_coords = []
            self.y_coords = []
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_source}: {e}")

    def resize(self, width: int, height: int, keep_aspect_ratio: bool = True) -> 'Image':
        """Resize the image to specified dimensions."""
        if keep_aspect_ratio:
            self.image.thumbnail((width, height))
        else:
            self.image = self.image.resize((width, height))
        return self

    def convert_to_grayscale(self) -> 'Image':
        """Convert the image to black and white (grayscale)."""
        self.image = ImageOps.grayscale(self.image)
        return self

    def rotate(self, angle: float) -> 'Image':
        """Rotate the image counter-clockwise by a given angle."""
        self.image = self.image.rotate(angle, expand=True)
        return self

    def crop(self, box: tuple) -> 'Image':
        """Crop the image using a bounding box tuple (left, upper, right, lower)."""
        self.image = self.image.crop(box)
        return self

    def crop_center(self, width: int = 512, height: int = 512) -> 'Image':
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

    def edge(self, grayscale: bool = True) -> 'Image':
        """Detects edges in the image using Pillow's FIND_EDGES filter."""
        if grayscale and self.image.mode != 'L':
            self.image = ImageOps.grayscale(self.image)

        self.image = self.image.filter(ImageFilter.FIND_EDGES)
        return self

    def slice_image(self, tile_size: int = 512) -> list:
        """Automatically slices the image into overlapping tiles of `tile_size`."""
        img_width, img_height = self.image.size

        if img_width < tile_size or img_height < tile_size:
            raise ValueError(f"Image size ({img_width}x{img_height}) is smaller than tile size ({tile_size}x{tile_size}).")

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

    def display_grid(self, tiles: list, display_size: int = 128) -> 'Image':
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

    def quantize_colors(self, k: int = 8) -> 'Image':
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

    def show(self) -> 'Image':
        """Display the current state of the image."""
        display(self.image)
        return self

    def save(self, output_path: str) -> None:
        """Save the processed image to disk."""
        try:
            self.image.save(output_path)
            print(f"Image successfully saved to {output_path}")
        except Exception as e:
            raise IOError(f"Failed to save image: {e}")
