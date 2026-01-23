import argparse
import os
import cv2
import numpy as np
import glob

def get_center_of_mass(mask):
    """
    Calculates the center of mass (centroid) of a grayscale mask.
    Returns (cx, cy). If mask is empty, returns center of image.
    """
    # Threshold to clear noise
    _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    
    h, w = mask.shape
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # Fallback to image center if map is empty/black
        cx, cy = w // 2, h // 2
        
    return cx, cy

def smart_crop(image, center, crop_size):
    """
    Crops the image around the given center (cx, cy) with size (crop_size, crop_size).
    Ensures the crop stays within image boundaries.
    """
    h, w = image.shape[:2]
    cx, cy = center
    hs = crop_size // 2
    
    # Calculate boundaries ensuring we don't go out of bounds
    x1 = max(0, cx - hs)
    y1 = max(0, cy - hs)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # Shift back if we hit the right/bottom edge
    if x2 > w:
        x2 = w
        x1 = w - crop_size
    if y2 > h:
        y2 = h
        y1 = h - crop_size
        
    # Final safety check for images smaller than crop size
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    
    crop = image[int(y1):int(y2), int(x1):int(x2)]
    
    # If image was smaller than crop_size, resize it up
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))
        
    return crop

def add_label(image, text):
    """Adds a small text label to the image."""
    h, w = image.shape[:2]
    # Add a small black strip at the top
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(labeled, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return labeled

def main():
    parser = argparse.ArgumentParser(description="Generate Smart Crop comparison grid.")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to original RGB images')
    parser.add_argument('--saliency_dir', type=str, required=True, help='Path to Deep Learning saliency maps')
    parser.add_argument('--output_path', type=str, default='crop_comparison.jpg', help='Output filename')
    parser.add_argument('--crop_size', type=int, default=300, help='Size of the square crop (pixels)')
    parser.add_argument('--limit', type=int, default=10, help='Number of rows to generate')
    
    args = parser.parse_args()

    # Get list of images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(args.img_dir, ext)))
    
    img_list.sort()
    img_list = img_list[:args.limit] # Limit number of rows

    if not img_list:
        print(f"No images found in {args.img_dir}")
        return

    rows = []

    for img_path in img_list:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # 1. Load Original Image
        original = cv2.imread(img_path)
        if original is None: continue
        h, w = original.shape[:2]
        
        # --- A. Center Crop (Blind) ---
        center_x, center_y = w // 2, h // 2
        crop_blind = smart_crop(original, (center_x, center_y), args.crop_size)

        # --- B. Samba Smart Crop ---
        # Try to find corresponding map (png or jpg)
        samba_path = os.path.join(args.saliency_dir, name_no_ext + '.png')
        if not os.path.exists(samba_path):
             samba_path = os.path.join(args.saliency_dir, filename) # Try original name
        
        if os.path.exists(samba_path):
            samba_map = cv2.imread(samba_path, cv2.IMREAD_GRAYSCALE)
            samba_center = get_center_of_mass(samba_map)
            crop_samba = smart_crop(original, samba_center, args.crop_size)
        else:
            print(f"Warning: Samba map not found for {filename}")
            crop_samba = np.zeros((args.crop_size, args.crop_size, 3), dtype=np.uint8) # Black placeholder

        # --- Resize Original for Display ---
        # Resize original to have height = crop_size, maintaining aspect ratio
        aspect = w / h
        new_w = int(args.crop_size * aspect)
        disp_orig = cv2.resize(original, (new_w, args.crop_size))
        
        # If original is too wide, crop the center for display to avoid huge width
        if new_w > args.crop_size * 2:
             start_x = (new_w - args.crop_size * 2) // 2
             disp_orig = disp_orig[:, start_x:start_x + args.crop_size*2]

        # Labeling (Only on the first row)
        if len(rows) == 0:
            disp_orig = add_label(disp_orig, "Original")
            crop_blind = add_label(crop_blind, "Center Crop")
            crop_samba = add_label(crop_samba, "Saliency ROI")

        # Stack Horizontal
        row = np.hstack([disp_orig, crop_blind, crop_samba])
        rows.append(row)

    # Find max width to align rows (since originals vary in width)
    max_width = max([r.shape[1] for r in rows])
    final_rows = []
    
    for r in rows:
        h, w = r.shape[:2]
        if w < max_width:
            pad = np.zeros((h, max_width - w, 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        final_rows.append(r)

    # Stack Vertical
    grid = np.vstack(final_rows)
    
    cv2.imwrite(args.output_path, grid)
    print(f"Successfully saved comparison grid to: {args.output_path}")

if __name__ == "__main__":
    main()
