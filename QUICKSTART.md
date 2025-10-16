# ⚡ Quick Start Guide

## 🚀 Chạy nhanh ứng dụng

### 1. Kích hoạt môi trường
```bash
# Windows (CMD)
seatbelt_env\Scripts\activate

# Windows (PowerShell)
seatbelt_env\Scripts\Activate.ps1

# Linux/Mac
source seatbelt_env/bin/activate
```

### 2. Chạy ứng dụng
```bash
cd ui
python app.py
```

### 3. Truy cập UI
- Mở trình duyệt: http://127.0.0.1:7860

## 🎯 Tính năng chính

### 🧠 Training Tab
- **Chọn Dataset**: Click "📁 Chọn Folder" → Chọn folder chứa ảnh
- **Tự động tạo cấu trúc**: Ứng dụng sẽ tự động tạo cấu trúc YOLO
- **Tham số**: Model size (s/m/l/x), epochs, batch size, learning rate
- **Start Training**: Click "🚀 Start Training"

### 🔍 Testing Tab
- **Test ảnh đơn**: Upload ảnh + chọn confidence threshold
- **Test folder**: Nhập đường dẫn folder chứa ảnh
- **Random sample**: Test ngẫu nhiên từ dataset
- **Validation**: Đánh giá model với metrics chi tiết

### 🏷️ Labeling Tab
- **Chọn dataset**: Click "📁 Chọn Folder" → Chọn folder chứa ảnh cần gán nhãn
- **Prepare Dataset**: Click "📁 Prepare Dataset" để tạo cấu trúc
- **Gán nhãn**: Sử dụng công cụ labeling để vẽ bounding box
- **Lưu labels**: Labels được lưu tự động theo format YOLO

### 🔄 Combined Training Tab
- **Original Dataset**: Chọn folder dataset gốc
- **Custom Data**: Chọn folder custom data đã gán nhãn
- **Training**: Model sẽ được train với data kết hợp
- **Cải thiện**: Model được cải thiện với custom labels

## 📥 Dataset Sources

### 1. Roboflow Dataset
- **Link**: [Seatbelt Detection Dataset](https://universe.roboflow.com/traffic-violations/seatbelt-detection-esut6)
- **Format**: YOLOv8
- **Classes**: person-seatbelt, person-noseatbelt
- **Download**: Cần API key từ Roboflow

### 2. Custom Dataset
- **Chuẩn bị**: Ảnh theo format JPG, PNG, BMP, TIFF
- **Gán nhãn**: Sử dụng Labeling Tool trong ứng dụng
- **Cấu trúc**: Ứng dụng tự động tạo cấu trúc YOLO

## 📁 Cấu trúc project

```
Seatbelt_detection/
├── 📄 README.md                    # Hướng dẫn chi tiết
├── 📄 QUICKSTART.md               # Hướng dẫn nhanh
├── 📄 requirements.txt             # Dependencies
├── 📁 seatbelt_env/               # Môi trường ảo
├── 📁 ui/                         # Ứng dụng chính
│   ├── 📄 app.py                  # Gradio UI
│   ├── 📄 train_utils.py          # Training module
│   ├── 📄 test_utils.py           # Testing module
│   ├── 📄 visual_utils.py         # Visualization module
│   └── 📄 label_tool.py           # Labeling module
├── 📁 data/                       # Dataset mẫu
│   ├── 📄 data.yaml              # Config dataset
│   ├── 📁 train/                 # Training data
│   ├── 📁 valid/                 # Validation data
│   └── 📁 test/                  # Test data
├── 📁 models/                     # Trained models
├── 📁 runs/                       # Results
│   ├── 📁 training_logs/          # Training logs
│   └── 📁 test_results/          # Test results
└── 📁 custom_data/                # Custom labeled data
    ├── 📁 images/                 # Custom images
    └── 📁 labels/                 # Custom labels
```

## 🔧 Các lệnh hữu ích

### Kiểm tra cài đặt
```bash
python -c "import ultralytics, gradio, cv2; print('✅ Tất cả dependencies OK')"
```

### Chạy trực tiếp
```bash
cd ui
python app.py
```

### Cài đặt thêm dependencies
```bash
pip install -r requirements.txt
```

### Deactivate môi trường
```bash
deactivate
```

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **Module not found**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**
   ```bash
   # Tìm process sử dụng port 7860
   netstat -ano | findstr :7860
   # Kill process
   taskkill /PID <PID> /F
   ```

3. **CUDA not available**
   - Ứng dụng tự động sử dụng CPU
   - Không cần cài đặt thêm

### Performance Tips

- **Training nhanh**: Sử dụng GPU (nếu có)
- **Testing chính xác**: Tăng confidence threshold
- **Memory issues**: Giảm batch size

## 📞 Hỗ trợ

- 📖 **Documentation**: README.md
- ⚡ **Quick Start**: QUICKSTART.md
- 🐛 **Issues**: GitHub Issues
- 💬 **Community**: Discord/Forum

---

## 🎉 CHÚC MỪNG!

Bạn đã sẵn sàng sử dụng **Seatbelt Detection - YOLOv11 All-in-One UI**!

### Bước tiếp theo:
1. **Chạy ứng dụng**: `cd ui && python app.py`
2. **Truy cập UI**: http://127.0.0.1:7860
3. **Chọn dataset**: Sử dụng "📁 Chọn Folder" để chọn ảnh
4. **Bắt đầu training**: Chọn tham số và click "🚀 Start Training"
5. **Test model**: Upload ảnh để test kết quả
6. **Gán nhãn**: Sử dụng Labeling Tool để cải thiện dataset
7. **Combined Training**: Kết hợp data gốc + custom data

**Lưu ý**: Đảm bảo luôn kích hoạt môi trường ảo trước khi chạy ứng dụng!