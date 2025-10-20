# 🚗 Seatbelt Detection - YOLOv11 All-in-One UI

Công cụ hoàn chỉnh cho phát hiện thắt dây an toàn sử dụng YOLOv11 với giao diện Gradio thân thiện.

## 🎯 Tính năng chính

- **🧠 Training**: Huấn luyện mô hình YOLOv11 với dataset tùy chỉnh
- **🔍 Testing**: Test model trên ảnh đơn, folder, hoặc random sample
- **📊 Visualization**: Hiển thị kết quả với metrics chi tiết
- **🏷️ Labeling**: Gán nhãn thủ công với công cụ tương tác
- **🔄 Combined Training**: Training kết hợp data gốc + custom data

## 🚀 Cài đặt nhanh

### 1. Clone repository
```bash
git clone https://github.com/LTaPhong/Seatbelt-detection.git
cd seatbelt-detection
```

### 2. Tạo môi trường ảo
```bash
python -m venv seatbelt_env
seatbelt_env\Scripts\activate  # Windows
# hoặc
source seatbelt_env/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng

#### Option A: Sử dụng Scripts (Khuyến nghị)
```bash
# Windows (PowerShell):
.\run_app.ps1

# Windows (Command Prompt):
run_app.bat
```

#### Option B: Chạy thủ công
```bash
# 1. Kích hoạt môi trường ảo
.\seatbelt_env\Scripts\Activate.ps1

# 2. Vào thư mục ui
cd ui

# 3. Chạy ứng dụng
python app.py
```

Truy cập: http://127.0.0.1:7860

## 📥 Tải Dataset

### Roboflow Dataset
- **Link**: [Seatbelt Detection Dataset](https://universe.roboflow.com/traffic-violations/seatbelt-detection-esut6)
- **Format**: YOLOv8
- **Classes**: person-seatbelt, person-noseatbelt

### Custom Dataset
- Chuẩn bị ảnh theo format: JPG, PNG, BMP, TIFF
- Sử dụng Labeling Tool để gán nhãn
- Ứng dụng sẽ tự động tạo cấu trúc YOLO

## 📊 Dataset Structure

```
your_dataset/
├── data.yaml              # Cấu hình dataset
├── train/
│   ├── images/            # Ảnh training
│   └── labels/            # Labels training (.txt files)
└── valid/
    ├── images/            # Ảnh validation
    └── labels/            # Labels validation (.txt files)
```


## 📖 Hướng dẫn sử dụng

### 🧠 Training
1. **Chọn dataset**: Click "📁 Chọn Folder" để chọn folder chứa ảnh
2. **Cấu trúc tự động**: Ứng dụng sẽ tự động tạo cấu trúc YOLO (train/valid/images, labels)
3. **Tham số training**: Chọn model size, epochs, batch size, learning rate
4. **Start Training**: Click "🚀 Start Training"

### 🔍 Testing
- **Test ảnh đơn**: Upload ảnh và chọn confidence threshold
- **Test folder**: Nhập đường dẫn folder chứa ảnh
- **Random sample**: Test ngẫu nhiên từ dataset
- **Validation**: Đánh giá model trên validation set

### 🏷️ Labeling
1. **Chọn dataset**: Click "📁 Chọn Folder" để chọn folder chứa ảnh cần gán nhãn
2. **Prepare Dataset**: Click "📁 Prepare Dataset" để tạo cấu trúc
3. **Gán nhãn**: Sử dụng công cụ labeling để vẽ bounding box
4. **Lưu labels**: Labels được lưu tự động theo format YOLO

### 🔄 Combined Training
1. **Original Dataset**: Chọn folder dataset gốc
2. **Custom Data**: Chọn folder custom data đã gán nhãn
3. **Training**: Model sẽ được train với data kết hợp
4. **Cải thiện**: Model được cải thiện với custom labels


## 📊 Metrics

- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision at IoU 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **Model not found**
   - Kiểm tra đường dẫn model trong `models/`
   - Đảm bảo đã train model trước khi test

2. **Dataset not found**
   - Kiểm tra cấu trúc thư mục dataset
   - Sử dụng "📁 Chọn Folder" để tự động tạo cấu trúc

3. **CUDA out of memory**
   - Giảm batch size
   - Sử dụng device="cpu"

4. **Labeling tool không mở**
   - Kiểm tra OpenCV installation
   - Đảm bảo có quyền truy cập display

### Performance Tips

1. **Training nhanh hơn**:
   - Sử dụng GPU (device="cuda")
   - Tăng batch size
   - Giảm image size

2. **Testing chính xác hơn**:
   - Tăng confidence threshold
   - Sử dụng model size lớn hơn
   - Augment data

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://universe.roboflow.com/traffic-violations/seatbelt-detection-esut6)
- [Gradio](https://gradio.app)
- [OpenCV](https://opencv.org)

## 📞 Support

- GitHub Issues: [Tạo issue](https://github.com/LTaPhong/Seatbelt-detection/issues)
- Email: phong570ltp@gmail.com

---

**Seatbelt Detection - YOLOv11 All-in-One UI**  
🚗 Phát hiện thắt dây an toàn với AI  
📧 Hỗ trợ: [GitHub Issues](https://github.com/LTaPhong/Seatbelt-detection/issues)