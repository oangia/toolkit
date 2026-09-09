import os
import urllib.request
import zipfile
from torchvision import transforms
from .image import Image
from .dataset import BaseImageDataset
 
def download(url, folder='./'):
    """Downloads a zip file from a URL and extracts it."""
    zip_name = 'temp_download.zip'
    
    print('Downloading...')
    urllib.request.urlretrieve(url, zip_name)
    
    os.makedirs(folder, exist_ok=True)
    
    print('Extracting...')
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(folder)
        
    os.remove(zip_name)
    print(f'Completed! Extracted to: "{folder}"')

normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales [0, 1] to [-1, 1]
        ])
