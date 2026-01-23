# [CVPR 2025-Highlight] Samba: A Unified Mamba-based Framework for General Salient Object Detection

[[PDF]](https://www.kerenfu.top/sources/CVPR2025_Samba.pdf) | [[Original Repo]](https://github.com/Jia-hao999/Samba)

**Lưu ý:** Dự án này được tham khảo từ mã nguồn gốc của Samba (CVPR 2025) và đã được refactor để tập trung chuyên biệt cho tác vụ **RGB SOD**.

**Samba** là một framework thống nhất mới dựa trên kiến trúc Mamba thuần túy để xử lý linh hoạt các tác vụ SOD tổng quát. Nó giới thiệu khối Saliency-Guided Mamba Block (SGMB) và phương pháp Context-Aware Upsampling (CAU) để tăng cường khả năng biểu diễn và căn chỉnh đặc trưng.

---

## 🛠 Cài đặt môi trường

### 1. Cài đặt PyTorch & CUDA
Dự án này yêu cầu **PyTorch 1.13.1** và **CUDA 11.7** (hoặc các phiên bản tương thích). Cần cài đặt chúng trước tiên tùy theo cấu hình máy:

```bash
# Ví dụ cho Linux với CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 2. Cài đặt các thư viện Python khác
Cài đặt các thư viện còn lại:

```bash
pip install -r requirements.txt
```

### Training

```bash
python train_rgb.py
```

### 2. Test
Để tạo ra các bản đồ nổi bật (saliency maps) từ checkpoint đã huấn luyện:

```bash
python test_rgb.py --model_path ./checkpoints/Samba_rgb.pth --testsavefold ./results
```
*   **Chạy với ảnh tùy chỉnh:** Để chạy test trên một thư mục ảnh bất kỳ (ví dụ `./original`), sử dụng tham số `--source_path`:
    ```bash
    python test_rgb.py --source_path ./original --testsavefold ./results --model_path ./checkpoints/Samba_rgb.pth
    ```

### 3. Evaluation
Để đánh giá chất lượng bản đồ sinh ra so với Ground Truth (GT):

```bash
python eval.py
```

*Cả ba đều hỗ trợ các bộ tham số tùy chỉnh. Có thể tham khảo thêm trong mã nguồn gốc.*

---

## 🖼️ Ứng dụng Thực tế: Cắt ảnh Thông minh

Nhóm cung cấp một ứng dụng thực tế để chứng minh sức mạnh của Saliency Map: **Cắt ảnh Thông minh dựa trên nội dung**.
Thay vì cắt chính giữa bức ảnh một cách mù quáng, công cụ này sử dụng Saliency Map để tự động căn chỉnh khung hình vào đối tượng quan trọng nhất.

### Cách chạy:
1.  **Tạo Saliency Maps:** Trước tiên, chạy `test_rgb.py` cho thư mục ảnh gốc bất kì (xem mục **Test** ở trên).
    ```bash
    python test_rgb.py --source_path ./original --testsavefold ./results
    ```
2.  **Chạy Ứng dụng:** Chạy script ứng dụng để xem so sánh trực quan.
    ```bash
    python app_smart_crop.py --img_dir ./original --deep_dir ./results/original
    ```

Công cụ sẽ hiển thị một lưới so sánh giữa: **Ảnh gốc vs. Cắt chính giữa (Center Crop) vs. Cắt theo Saliency (Smart Crop)**.

---

## 📂 Dữ liệu & Pre-trained Weights

### Pre-trained Weights
*   **VMamba-S Backbone:** [[Baidu]](https://pan.baidu.com/s/1SaEV237VCzSEn558gEBiXg) (Mã: zsxa)
*   **Samba Full Weights:** [[Baidu]](https://pan.baidu.com/s/15787DVEmW59ftztopv-yMg) (Mã: bkvw)

### Datasets
*   **RGB SOD:** DUTS, ECSSD, HKU-IS, PASCAL-S, DUT-O. [[Link]](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g) (Mã: x7kn)

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