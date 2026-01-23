# [CVPR 2025-Highlight] Samba: A Unified Mamba-based Framework for General Salient Object Detection

[[PDF]](https://www.kerenfu.top/sources/CVPR2025_Samba.pdf) | [[Original Repo]](https://github.com/Jia-hao999/Samba)

**Note:** This project is referenced from the original source code of Samba (CVPR 2025) and has been refactored to focus specifically on the **RGB SOD** task.

**Samba** is a new unified framework based on pure Mamba architecture to flexibly handle general SOD tasks. It introduces the Saliency-Guided Mamba Block (SGMB) and Context-Aware Upsampling (CAU) method to enhance feature representation and alignment.

---

## 📂 Data & Pre-trained Weights

### Pre-trained Weights
*   **VMamba-S Backbone:** [[Baidu]](https://pan.baidu.com/s/1SaEV237VCzSEn558gEBiXg) (Code: zsxa)
*   **Samba Full Weights:** [[Baidu]](https://pan.baidu.com/s/15787DVEmW59ftztopv-yMg) (Code: bkvw)

### Datasets
*   **RGB SOD:** DUTS, ECSSD, HKU-IS, PASCAL-S, DUT-O. [[Baidu]](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g) (Code: x7kn)

Or you can download via the following Drive link [[Google Drive]](https://drive.google.com/drive/folders/1gvHI9cKB7koM9c9RyNpTYuTWPtMLf0lD?usp=sharing)

---

## 🛠 Environment Setup

### 1. Install PyTorch & CUDA
This project requires **PyTorch 1.13.1** and **CUDA 11.7** (or compatible versions). You need to install them first depending on your machine configuration:

```bash
# Example for Linux with CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 2. Install other Python libraries
Install the remaining libraries:

```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

Before running training or testing, you need to manually prepare the `data` folder in the project root directory.

1.  **Download:** Download the datasets from the Baidu link above.
2.  **Unzip and Organize:** After downloading, unzip and place them in the `data` folder according to the following structure:
    ```text
    K2023-CQ-XLAV-Senko-Samba/
    ├── data/
    │   ├── DUTS-TR/
    │   ├── DUTS-TE/
    │   ├── DUT-OMRON/
    │   ├── ECSSD/
    │   ├── HKU-IS/
    │   ├── PASCAL-S/
    │   └── SOD/
    ├── checkpoints/
    ├── models/
    └── ...
    ```
    *Note: The `results` folder will be automatically created when you run the test script.*

---

## 🚀 Running the Program

### 1. Training

```bash
python train_rgb.py
```

### 2. Test
To generate saliency maps from trained checkpoints:

```bash
python test_rgb.py --model_path ./checkpoints/Samba_rgb.pth --testsavefold ./results
```
*   **Run with custom images:** To run tests on any image folder (e.g. `./original`), use the `--source_path` parameter:
    ```bash
    python test_rgb.py --source_path ./original --testsavefold ./results --model_path ./checkpoints/Samba_rgb.pth
    ```

### 3. Evaluation
To evaluate the quality of the generated maps against the Ground Truth (GT):

```bash
python eval.py
```

*All three support custom parameter sets. You can refer to the original source code for more details.*

---

## 🖼️ Real-world Application: Smart Image Cropping

The team provides a real-world application to demonstrate the power of Saliency Map: **Content-Aware Smart Image Cropping**.
Instead of blindly cropping the center of the image, this tool uses the Saliency Map to automatically align the frame to the most important object.

### How to run:
1.  **Generate Saliency Maps:** First, run `test_rgb.py` for any original image folder (see **Test** section above).
    ```bash
    python test_rgb.py --source_path ./original --testsavefold ./results
    ```
2.  **Run Application:** Run the application script to see the visual comparison.
    ```bash
    python app_smart_crop.py --img_dir ./original --saliency_dir ./results/original
    ```

The tool will output a comparison grid between: **Original Image vs. Center Crop vs. Smart Crop (Saliency-based)**.

---

## 📄 Citation
```bibtex
@InProceedings{He_2025_CVPR,
    author    = {He, Jiahao and Fu, Keren and Liu, Xiaohong and Zhao, Qijun},
    title     = {Samba: A Unified Mamba-based Framework for General Salient Object Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {25314-25324}
}
```