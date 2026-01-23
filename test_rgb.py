import argparse
import os
from dataloaders.rgb_dataset import Dataset
import torch
from torchvision import transforms
from dataloaders import transform_single
from torch.utils import data
from models.SalMamba_single import Model
import numpy as np
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda
#CUDA_VISIBLE_DEVICES=1 python test_tri.py
# test
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=4)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--spatial_ckpt', type=str, default=None)
parser.add_argument('--flow_ckpt', type=str, default=None)
parser.add_argument('--depth_ckpt', type=str, default=None)
parser.add_argument('--model_path', type=str, default='./checkpoints/Samba_rgb.pth')
parser.add_argument('--test_dataset', nargs='+', default=['DUTS-TE','DUT-OMRON','ECSSD','PASCAL-S','HKU-IS'])
parser.add_argument('--testsavefold', type=str, default='./results')
parser.add_argument('--test_limit', type=int, default=None, help='Limit the number of images/batches to process')
parser.add_argument('--source_path', type=str, default=None, help='Path to a custom folder of images to test')

# Misc
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
config = parser.parse_args()

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_list.append(os.path.join(root, file))

    def __getitem__(self, item):
        img_path = self.image_list[item]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        size = (h, w)
        name = os.path.basename(img_path)
        
        # Split name and ext to save png later
        name_no_ext = os.path.splitext(name)[0] + '.png'

        # Create dummy label for transforms
        label = np.zeros((h, w), dtype=np.uint8)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
            
        sample['size'] = torch.tensor(size)
        sample['dataset'] = 'Custom' # Dummy dataset name
        sample['name'] = name_no_ext

        return sample

    def __len__(self):
        return len(self.image_list)

composed_transforms_te = transforms.Compose([
    transform_single.FixedResize(size=(config.input_size, config.input_size)),
    transform_single.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform_single.ToTensor()])

if config.source_path:
    print(f"Loading images from custom folder: {config.source_path}")
    dataset = CustomDataset(root_dir=config.source_path, transform=composed_transforms_te)
    # Update save path to include the folder name
    config.testsavefold = os.path.join(config.testsavefold, os.path.basename(os.path.normpath(config.source_path)))
else:
    dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')

test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread,
                              drop_last=True, shuffle=False)

print('mode: {}'.format(config.mode))
print('------------------------------------------')
model = Model()

if config.cuda:
    model = model.cuda()
assert (config.model_path != ''), ('Test mode, please import pretrained model path!')
assert (os.path.exists(config.model_path)), ('please import correct pretrained model path!')
print('load model……all checkpoints')

model.load_pretrain_model(config.model_path)
model.eval()

if not os.path.exists(config.testsavefold):
    os.makedirs(config.testsavefold)
num=0
total_batches = len(test_loader)
if config.test_limit is not None:
    total_batches = min(total_batches, config.test_limit)

for i, data_batch in enumerate(test_loader):
    if config.test_limit is not None and i >= config.test_limit:
        print(f"Reached limit of {config.test_limit} images/batches. Stopping.")
        break
    print("progress {}/{}\n".format(i + 1, total_batches))
    image, name, size = data_batch['image'], data_batch['name'], data_batch['size']
    dataset = data_batch['dataset']

    if config.cuda:
        image = image[0].cuda()
        image = image.unsqueeze(0)
    with torch.no_grad():

        out, saliency = model(image)

        for i in range(config.test_batch_size):
            if config.source_path:
                presavefold = config.testsavefold # Use the flat directory for custom source
            else:
                presavefold = os.path.join(config.testsavefold, dataset[i])

            if not os.path.exists(presavefold):
                os.makedirs(presavefold)
                print(f"Saving results to: {presavefold}")
            pre1 = torch.nn.Sigmoid()(out[0][i])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
            pre1 = cv2.resize(pre1, (int(size[0][1]), int(size[0][0])))
            pre1 = pre1.astype(np.uint8)
            cv2.imwrite(presavefold + '/' + name[i], pre1)
