import os
import sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def is_corrupted(img_path):
    """
    Checks if an image is corrupted.
    Returns True if corrupted, False otherwise.
    """
    try:
        # Image.open() identifies the file and reads the header.
        # img.verify() checks the integrity of the file without decoding it.
        with Image.open(img_path) as img:
            img.verify()
        
        # Some corruptions are only caught when actually trying to load/decode.
        # However, verify() is much faster and catches structural issues.
        # If we want to be 100% sure we can use img.transpose(Image.FLIP_LEFT_RIGHT)
        # or img.load(), but that is much slower.
        # For 'PIL.UnidentifiedImageError', verify() or even just open() is enough.
        return False
    except Exception:
        return True

def find_images(root_dir):
    """
    Recursively finds all image files in the root_dir.
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    # usage: python corrupt.py [IMAGE_ROOT] [OUTPUT_FILE]
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "Test_Data/JPEGImages"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "corrupted_images.txt"
    num_threads = 10
    
    print(f"Scanning directory: {root_dir}")
    image_paths = find_images(root_dir)
    print(f"Found {len(image_paths)} images.")
    
    print(f"Checking for corruption using {num_threads} threads...")
    corrupted_images = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Map the check function to all paths
        results = list(tqdm(executor.map(lambda p: (p, is_corrupted(p)), image_paths), total=len(image_paths)))
    
    # Filter for corrupted images
    for path, corrupted in results:
        if corrupted:
            corrupted_images.append(path)
    
    print(f"\nFound {len(corrupted_images)} corrupted images.")
    
    # Save the list to a file
    with open(output_file, 'w') as f:
        for path in corrupted_images:
            f.write(path + '\n')
    
    print(f"Corrupted image paths saved to: {output_file}")

if __name__ == "__main__":
    main()
