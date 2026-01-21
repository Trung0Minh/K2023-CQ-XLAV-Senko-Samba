from torch.utils import data
import os
from PIL import Image
import cv2
import numpy as np


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        # List all valid image files in prediction directory
        pred_files = [f for f in os.listdir(pred_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.image_path = []
        self.label_path = []

        for p_name in pred_files:
            # Construct path for prediction
            p_path = os.path.join(pred_root, p_name)
            
            # Find corresponding GT file (handling potential extension mismatch)
            base_name = os.path.splitext(p_name)[0]
            
            # Check possible GT extensions
            gt_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                candidate = os.path.join(label_root, base_name + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            
            if gt_path:
                self.image_path.append(p_path)
                self.label_path.append(gt_path)
            # else:
            #     print(f"Warning: GT for {p_name} not found in {label_root}")

        print(f"  EvalDataset: Found {len(self.image_path)} valid pairs.")

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')

        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
