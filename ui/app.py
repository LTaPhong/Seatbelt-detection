"""
Seatbelt Detection - YOLOv11 All-in-One UI
Gradio application for training, testing, visualization, and labeling
"""
import gradio as gr
import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

# Import our custom modules
try:
    from .train_utils import SeatbeltTrainer, download_roboflow_dataset
    from .test_utils import SeatbeltTester
    from .visual_utils import SeatbeltVisualizer
    from .label_tool import SeatbeltLabeler
except ImportError:
    from train_utils import SeatbeltTrainer, download_roboflow_dataset
    from test_utils import SeatbeltTester
    from visual_utils import SeatbeltVisualizer
    from label_tool import SeatbeltLabeler

class SeatbeltApp:
    def __init__(self):
        """Initialize the application"""
        self.trainer = SeatbeltTrainer()
        self.tester = SeatbeltTester()
        self.visualizer = SeatbeltVisualizer()
        self.labeler = SeatbeltLabeler()
        
        # Create necessary directories
        os.makedirs('runs/training_logs', exist_ok=True)
        os.makedirs('runs/test_results', exist_ok=True)
        os.makedirs('custom_data/images', exist_ok=True)
        os.makedirs('custom_data/labels', exist_ok=True)
    
    def prepare_dataset_structure(self, data_folder):
        """Prepare dataset structure for training"""
        return self.check_and_prepare_dataset(data_folder)
    
    def validate_training_inputs(self, data_folder, model_size, epochs, imgsz, batch, lr, device):
        """Validate training input parameters"""
        errors = []
        
        # Check data folder
        if not data_folder or not os.path.exists(data_folder):
            errors.append("❌ Vui lòng chọn folder chứa dataset")
        
        # Check epochs
        if not epochs or epochs < 1 or epochs > 1000:
            errors.append("❌ Epochs phải từ 1 đến 1000")
        
        # Check image size
        if not imgsz or imgsz < 32 or imgsz > 2048:
            errors.append("❌ Image size phải từ 32 đến 2048")
        
        # Check batch size
        if not batch or batch < 1 or batch > 128:
            errors.append("❌ Batch size phải từ 1 đến 128")
        
        # Check learning rate
        if not lr or lr <= 0 or lr > 1:
            errors.append("❌ Learning rate phải từ 0.001 đến 1.0")
        
        return errors
    
    def train_model(self, data_folder, model_size, epochs, imgsz, batch, lr, device, progress=gr.Progress()):
        """Train YOLOv11 model with selected dataset"""
        try:
            # Validate inputs
            validation_errors = self.validate_training_inputs(data_folder, model_size, epochs, imgsz, batch, lr, device)
            if validation_errors:
                return "\n".join(validation_errors), "Chưa có model nào được train"
            
            progress(0.1, desc="🔍 Kiểm tra dataset...")
            
            # Prepare dataset structure
            prepare_result = self.prepare_dataset_structure(data_folder)
            if "❌" in prepare_result:
                return prepare_result, "Chưa có model nào được train"
            
            progress(0.2, desc="📁 Chuẩn bị cấu trúc dataset...")
            
            # Create data.yaml path
            data_yaml_path = os.path.join(data_folder, "data.yaml")
            
            # Update trainer with new parameters
            self.trainer = SeatbeltTrainer(data_path=data_yaml_path, model_size=model_size)
            
            progress(0.3, desc="🚀 Khởi tạo model...")
            
            # Start training with progress tracking
            results = self.trainer.start_training(
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                lr=float(lr),
                device=device,
                progress_callback=lambda p: progress(0.3 + 0.6 * p, desc=f"🧠 Training... Epoch {p:.0f}/{epochs}")
            )
            
            progress(0.9, desc="💾 Lưu model...")
            
            if results:
                progress(1.0, desc="✅ Training hoàn thành!")
                
                # Update model status
                model_status = f"""✅ Model đã được train thành công!
📊 YOLOv11-{model_size} | Epochs: {epochs} | Size: {imgsz}
📁 Path: runs/train/weights/best.pt
⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
                
                output_text = f"""✅ Training hoàn thành thành công!

📊 Thông tin training:
- Model: YOLOv11-{model_size}
- Epochs: {epochs}
- Image Size: {imgsz}
- Batch Size: {batch}
- Learning Rate: {lr}
- Device: {device}

📁 Kết quả được lưu tại: runs/train/
🎯 Model tốt nhất: runs/train/weights/best.pt
📈 Logs: runs/train/results.csv

Bạn có thể sử dụng model này để test trong tab "Test / Visualize"!"""
                
                return output_text, model_status
            else:
                return "❌ Training thất bại. Vui lòng kiểm tra log để biết lỗi chi tiết.", "Chưa có model nào được train"
                
        except Exception as e:
            error_msg = f"❌ Lỗi training: {str(e)}\n\n💡 Gợi ý:\n- Kiểm tra đường dẫn dataset\n- Đảm bảo có đủ RAM/VRAM\n- Thử giảm batch size nếu bị lỗi memory"
            return error_msg, "Chưa có model nào được train"
    
    def open_training_results(self):
        """Open training results folder"""
        try:
            import subprocess
            import platform
            
            results_path = os.path.abspath("runs/train")
            if not os.path.exists(results_path):
                return "❌ Chưa có kết quả training nào. Hãy train model trước!"
            
            if platform.system() == "Windows":
                subprocess.run(["explorer", results_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", results_path])
            else:  # Linux
                subprocess.run(["xdg-open", results_path])
            
            return f"✅ Đã mở thư mục kết quả: {results_path}"
            
        except Exception as e:
            return f"❌ Lỗi mở thư mục: {str(e)}"
    
    def test_single_image(self, image, conf_threshold):
        """Test single image"""
        try:
            if image is None:
                return None, "❌ Vui lòng upload ảnh"
            
            # Save uploaded image temporarily
            temp_path = "temp_image.jpg"
            cv2.imwrite(temp_path, image)
            
            # Test image
            result = self.tester.test_single_image(temp_path, float(conf_threshold))
            
            if result and 'detections' in result:
                # Draw detections
                result_image = self.visualizer.draw_detection(image, result['detections'], float(conf_threshold))
                
                # Create summary
                summary = f"🔍 Phát hiện {len(result['detections'])} objects:\n"
                for i, det in enumerate(result['detections']):
                    summary += f"{i+1}. {det['class_name']}: {det['confidence']:.3f}\n"
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return result_image, summary
            else:
                return image, "❌ Không phát hiện được object nào"
                
        except Exception as e:
            return image, f"❌ Lỗi test: {str(e)}"
    
    def test_folder(self, folder_path, conf_threshold):
        """Test folder of images"""
        try:
            if not folder_path or not os.path.exists(folder_path):
                return "❌ Vui lòng chọn folder hợp lệ"
            
            # Test folder
            results = self.tester.test_folder(folder_path, float(conf_threshold))
            
            if results:
                metrics = results.get('metrics', {})
                summary = f"📊 Kết quả test folder:\n"
                summary += f"- Tổng ảnh: {results['total_images']}\n"
                summary += f"- Ảnh đã xử lý: {results['processed_images']}\n"
                summary += f"- Tổng detections: {metrics.get('total_detections', 0)}\n"
                summary += f"- Trung bình detections/ảnh: {metrics.get('avg_detections_per_image', 0):.2f}\n"
                summary += f"- Confidence trung bình: {metrics.get('avg_confidence', 0):.3f}\n"
                
                # Class distribution
                class_dist = metrics.get('class_distribution', {})
                if class_dist:
                    summary += f"\n📈 Phân bố class:\n"
                    for class_name, count in class_dist.items():
                        summary += f"- {class_name}: {count}\n"
                
                return summary
            else:
                return "❌ Test folder thất bại"
                
        except Exception as e:
            return f"❌ Lỗi test folder: {str(e)}"
    
    def test_random_sample(self, num_samples, conf_threshold):
        """Test random sample"""
        try:
            results = self.tester.test_random_sample(
                data_path="data",
                num_samples=int(num_samples),
                conf_threshold=float(conf_threshold)
            )
            
            if results:
                metrics = results.get('metrics', {})
                summary = f"🎲 Kết quả random sample:\n"
                summary += f"- Số ảnh test: {results['actual_samples']}\n"
                summary += f"- Tổng detections: {metrics.get('total_detections', 0)}\n"
                summary += f"- Confidence trung bình: {metrics.get('avg_confidence', 0):.3f}\n"
                
                # Class distribution
                class_dist = metrics.get('class_distribution', {})
                if class_dist:
                    summary += f"\n📈 Phân bố class:\n"
                    for class_name, count in class_dist.items():
                        summary += f"- {class_name}: {count}\n"
                
                return summary
            else:
                return "❌ Random test thất bại"
                
        except Exception as e:
            return f"❌ Lỗi random test: {str(e)}"
    
    def validate_model(self, conf_threshold):
        """Validate model"""
        try:
            metrics = self.tester.validate_model(conf_threshold=float(conf_threshold))
            
            if metrics:
                summary = f"📊 Kết quả validation:\n"
                summary += f"- mAP@0.5: {metrics['mAP50']:.3f}\n"
                summary += f"- mAP@0.5:0.95: {metrics['mAP50_95']:.3f}\n"
                summary += f"- Precision: {metrics['precision']:.3f}\n"
                summary += f"- Recall: {metrics['recall']:.3f}\n"
                summary += f"- F1 Score: {metrics['f1']:.3f}\n"
                
                return summary
            else:
                return "❌ Validation thất bại"
                
        except Exception as e:
            return f"❌ Lỗi validation: {str(e)}"
    
    def upload_image_for_labeling(self, image):
        """Upload image for labeling"""
        try:
            if image is None:
                return None, "❌ Vui lòng upload ảnh"
            
            # Save image to custom data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_{timestamp}.jpg"
            image_path = os.path.join("custom_data/images", filename)
            cv2.imwrite(image_path, image)
            
            return image, f"✅ Đã upload ảnh: {filename}"
            
        except Exception as e:
            return image, f"❌ Lỗi upload: {str(e)}"
    
    def start_labeling(self, data_folder):
        """Start labeling session for a dataset folder"""
        try:
            if not data_folder or not os.path.exists(data_folder):
                return "❌ Vui lòng chọn folder chứa dataset"
            
            # Check if folder has images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(data_folder).glob(f"*{ext}"))
                image_files.extend(Path(data_folder).glob(f"*{ext.upper()}"))
            
            if not image_files:
                return "❌ Không tìm thấy ảnh trong folder"
            
            # Create proper structure for labeling
            images_dir = os.path.join(data_folder, "images")
            labels_dir = os.path.join(data_folder, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Move images to images folder
            moved_count = 0
            for img_file in image_files:
                dest_path = os.path.join(images_dir, img_file.name)
                if not os.path.exists(dest_path):
                    import shutil
                    shutil.move(str(img_file), dest_path)
                    moved_count += 1
            
            # Update labeler to use this folder
            self.labeler = SeatbeltLabeler(custom_data_path=data_folder)
            
            return f"✅ Đã chuẩn bị {moved_count} ảnh cho labeling. Cấu trúc folder đã được tạo."
                
        except Exception as e:
            return f"❌ Lỗi chuẩn bị labeling: {str(e)}"
    
    def select_folder(self):
        """Open folder selection dialog using subprocess"""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                # Use Windows folder picker
                result = subprocess.run([
                    "powershell", "-Command", 
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    "$folderBrowser.Description = 'Chọn folder chứa dataset'; "
                    "$result = $folderBrowser.ShowDialog(); "
                    "if ($result -eq 'OK') { Write-Output $folderBrowser.SelectedPath }"
                ], capture_output=True, text=True, shell=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    folder_path = result.stdout.strip()
                    # Check and prepare dataset structure
                    return self.check_and_prepare_dataset(folder_path)
                else:
                    return ""
            else:
                # For Linux/Mac, use zenity or kdialog
                try:
                    result = subprocess.run([
                        "zenity", "--file-selection", "--directory", 
                        "--title=Chọn folder chứa dataset"
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        folder_path = result.stdout.strip()
                        # Check and prepare dataset structure
                        return self.check_and_prepare_dataset(folder_path)
                except:
                    pass
                
                return ""
                
        except Exception as e:
            return f"❌ Lỗi chọn folder: {str(e)}"
    
    def check_and_prepare_dataset(self, folder_path):
        """Check dataset structure and prepare if needed"""
        try:
            if not folder_path or not os.path.exists(folder_path):
                return "❌ Folder không tồn tại"
            
            # Check if already has proper YOLO structure
            has_train_images = os.path.exists(os.path.join(folder_path, "train", "images"))
            has_train_labels = os.path.exists(os.path.join(folder_path, "train", "labels"))
            has_valid_images = os.path.exists(os.path.join(folder_path, "valid", "images"))
            has_valid_labels = os.path.exists(os.path.join(folder_path, "valid", "labels"))
            has_data_yaml = os.path.exists(os.path.join(folder_path, "data.yaml"))
            
            if has_train_images and has_train_labels and has_valid_images and has_valid_labels and has_data_yaml:
                return f"✅ Dataset đã có cấu trúc YOLO hợp lệ: {folder_path}"
            
            # Check if folder has images directly
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))
                image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            if not image_files:
                return f"❌ Không tìm thấy ảnh trong folder: {folder_path}"
            
            # Create proper YOLO structure
            train_images = os.path.join(folder_path, "train", "images")
            train_labels = os.path.join(folder_path, "train", "labels")
            valid_images = os.path.join(folder_path, "valid", "images")
            valid_labels = os.path.join(folder_path, "valid", "labels")
            
            # Create directories
            for dir_path in [train_images, train_labels, valid_images, valid_labels]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Move images to train/images (80% for training)
            import random
            random.shuffle(image_files)
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            valid_files = image_files[split_idx:]
            
            # Move training images
            for img_file in train_files:
                dest_path = os.path.join(train_images, img_file.name)
                if not os.path.exists(dest_path):
                    import shutil
                    shutil.move(str(img_file), dest_path)
            
            # Move validation images
            for img_file in valid_files:
                dest_path = os.path.join(valid_images, img_file.name)
                if not os.path.exists(dest_path):
                    import shutil
                    shutil.move(str(img_file), dest_path)
            
            # Create data.yaml
            data_yaml_path = os.path.join(folder_path, "data.yaml")
            data_config = {
                'path': folder_path,
                'train': 'train/images',
                'val': 'valid/images',
                'nc': 2,
                'names': ['person-seatbelt', 'person-noseatbelt']
            }
            
            import yaml
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            return f"✅ Đã tạo cấu trúc YOLO cho dataset:\n📁 {folder_path}\n📊 Training: {len(train_files)} ảnh\n📊 Validation: {len(valid_files)} ảnh\n📄 data.yaml đã được tạo"
            
        except Exception as e:
            return f"❌ Lỗi chuẩn bị dataset: {str(e)}"
    
    def train_combined_data(self, original_data_folder, custom_data_folder, model_size, epochs, imgsz, batch, lr, device):
        """Train model with combined original data and custom labeled data"""
        try:
            if not original_data_folder or not os.path.exists(original_data_folder):
                return "❌ Vui lòng chọn folder dataset gốc"
            
            if not custom_data_folder or not os.path.exists(custom_data_folder):
                return "❌ Vui lòng chọn folder custom data đã gán nhãn"
            
            # Create combined dataset structure
            combined_folder = "combined_dataset"
            os.makedirs(combined_folder, exist_ok=True)
            
            # Copy original data
            import shutil
            train_images = os.path.join(combined_folder, "train", "images")
            train_labels = os.path.join(combined_folder, "train", "labels")
            val_images = os.path.join(combined_folder, "valid", "images")
            val_labels = os.path.join(combined_folder, "valid", "labels")
            
            for dir_path in [train_images, train_labels, val_images, val_labels]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Copy original training data
            orig_train_images = os.path.join(original_data_folder, "train", "images")
            orig_train_labels = os.path.join(original_data_folder, "train", "labels")
            if os.path.exists(orig_train_images):
                for file in os.listdir(orig_train_images):
                    shutil.copy2(os.path.join(orig_train_images, file), train_images)
            if os.path.exists(orig_train_labels):
                for file in os.listdir(orig_train_labels):
                    shutil.copy2(os.path.join(orig_train_labels, file), train_labels)
            
            # Copy custom data
            custom_images = os.path.join(custom_data_folder, "images")
            custom_labels = os.path.join(custom_data_folder, "labels")
            if os.path.exists(custom_images):
                for file in os.listdir(custom_images):
                    shutil.copy2(os.path.join(custom_images, file), train_images)
            if os.path.exists(custom_labels):
                for file in os.listdir(custom_labels):
                    shutil.copy2(os.path.join(custom_labels, file), train_labels)
            
            # Copy validation data
            orig_val_images = os.path.join(original_data_folder, "valid", "images")
            orig_val_labels = os.path.join(original_data_folder, "valid", "labels")
            if os.path.exists(orig_val_images):
                for file in os.listdir(orig_val_images):
                    shutil.copy2(os.path.join(orig_val_images, file), val_images)
            if os.path.exists(orig_val_labels):
                for file in os.listdir(orig_val_labels):
                    shutil.copy2(os.path.join(orig_val_labels, file), val_labels)
            
            # Create data.yaml for combined dataset
            data_yaml_path = os.path.join(combined_folder, "data.yaml")
            data_config = {
                'path': combined_folder,
                'train': 'train/images',
                'val': 'valid/images',
                'nc': 2,
                'names': ['person-seatbelt', 'person-noseatbelt']
            }
            
            import yaml
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            # Train with combined data
            self.trainer = SeatbeltTrainer(data_path=data_yaml_path, model_size=model_size)
            results = self.trainer.start_training(
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                lr=float(lr),
                device=device
            )
            
            if results:
                return "✅ Training kết hợp hoàn thành! Model đã được cải thiện với custom data."
            else:
                return "❌ Training kết hợp thất bại"
                
        except Exception as e:
            return f"❌ Lỗi training kết hợp: {str(e)}"
    

def create_interface():
    """Create Gradio interface"""
    app = SeatbeltApp()
    
    with gr.Blocks(title="Seatbelt Detection - YOLOv11", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🚗 Seatbelt Detection - YOLOv11 All-in-One UI")
        gr.Markdown("Công cụ hoàn chỉnh cho training, testing, visualization và labeling")
        
        with gr.Tabs():
            # Tab 1: Training
            with gr.Tab("🧠 Training"):
                gr.Markdown("## Huấn luyện mô hình YOLOv11")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Dataset Selection
                        gr.Markdown("### 📁 Dataset")
                        data_folder = gr.Textbox(
                            label="Dataset Folder Path",
                            placeholder="Nhập đường dẫn đến folder chứa dataset",
                            info="Chọn folder chứa dataset hoặc click nút bên cạnh để chọn"
                        )
                        folder_btn = gr.Button("📁 Chọn Folder", variant="secondary", size="sm")
                        
                        # Model Configuration
                        gr.Markdown("### ⚙️ Model Configuration")
                        model_size = gr.Dropdown(
                            choices=["s", "m", "l", "x"],
                            value="s",
                            label="Model Size",
                            info="s=small, m=medium, l=large, x=xlarge"
                        )
                        
                        # Training Parameters
                        gr.Markdown("### 🎯 Training Parameters")
                        with gr.Row():
                            epochs = gr.Number(
                                value=50, 
                                label="Epochs",
                                info="Số lần lặp qua toàn bộ dataset",
                                precision=0
                            )
                            imgsz = gr.Number(
                                value=640, 
                                label="Image Size",
                                info="Kích thước ảnh input",
                                precision=0
                            )
                        
                        with gr.Row():
                            batch = gr.Number(
                                value=16, 
                                label="Batch Size",
                                info="Số ảnh xử lý cùng lúc",
                                precision=0
                            )
                            lr = gr.Number(
                                value=0.01, 
                                label="Learning Rate",
                                info="Tốc độ học của model",
                                precision=4
                            )
                        
                        device = gr.Dropdown(
                            choices=["auto", "cpu", "cuda"],
                            value="auto",
                            label="Device",
                            info="auto= tự động chọn, cpu= CPU, cuda= GPU"
                        )
                        
                        # Training Controls
                        gr.Markdown("### 🚀 Training Controls")
                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        
                        # Model Status
                        gr.Markdown("### 📊 Model Status")
                        model_status = gr.Textbox(
                            label="Current Model",
                            value="Chưa có model nào được train",
                            interactive=False,
                            lines=2
                        )
                        
                    with gr.Column(scale=1):
                        # Training Progress
                        gr.Markdown("### 📈 Training Progress")
                        training_progress = gr.Progress()
                        
                        # Training Output
                        training_output = gr.Textbox(
                            label="Training Output",
                            lines=15,
                            interactive=False,
                            placeholder="Kết quả training sẽ hiển thị ở đây..."
                        )
                        
                        # Quick Actions
                        gr.Markdown("### ⚡ Quick Actions")
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Output", variant="secondary", size="sm")
                            open_results_btn = gr.Button("📁 Open Results", variant="secondary", size="sm")
                
                # Training event handlers
                train_btn.click(
                    app.train_model,
                    inputs=[data_folder, model_size, epochs, imgsz, batch, lr, device, training_progress],
                    outputs=[training_output, model_status]
                )
                
                # Folder selection
                folder_btn.click(
                    app.select_folder,
                    outputs=data_folder
                )
                
                # Quick actions
                clear_btn.click(
                    lambda: ("", "Chưa có model nào được train"),
                    outputs=[training_output, model_status]
                )
                
                open_results_btn.click(
                    app.open_training_results,
                    outputs=training_output
                )
            
            # Tab 2: Testing & Visualization
            with gr.Tab("🔍 Test / Visualize"):
                gr.Markdown("## Test và Visualize kết quả")
                
                with gr.Row():
                    with gr.Column():
                        # Single image test
                        gr.Markdown("### Test ảnh đơn")
                        test_image = gr.Image(label="Upload ảnh", type="numpy")
                        test_conf = gr.Slider(0.1, 1.0, 0.25, label="Confidence Threshold")
                        test_single_btn = gr.Button("🔍 Test Image", variant="primary")
                        
                        # Folder test
                        gr.Markdown("### Test folder")
                        test_folder_path = gr.Textbox(label="Folder Path")
                        test_folder_btn = gr.Button("📁 Test Folder")
                        
                        # Random sample test
                        gr.Markdown("### Random Sample")
                        num_samples = gr.Number(value=10, label="Number of Samples")
                        test_random_btn = gr.Button("🎲 Random Test")
                        
                        # Validation
                        gr.Markdown("### Validation")
                        val_conf = gr.Slider(0.1, 1.0, 0.25, label="Confidence Threshold")
                        val_btn = gr.Button("📊 Validate Model")
                        
                    with gr.Column():
                        test_result_image = gr.Image(label="Kết quả", type="numpy")
                        test_output = gr.Textbox(
                            label="Test Output",
                            lines=15,
                            interactive=False
                        )
                
                test_single_btn.click(
                    app.test_single_image,
                    inputs=[test_image, test_conf],
                    outputs=[test_result_image, test_output]
                )
                
                test_folder_btn.click(
                    app.test_folder,
                    inputs=[test_folder_path, test_conf],
                    outputs=test_output
                )
                
                test_random_btn.click(
                    app.test_random_sample,
                    inputs=[num_samples, test_conf],
                    outputs=test_output
                )
                
                val_btn.click(
                    app.validate_model,
                    inputs=[val_conf],
                    outputs=test_output
                )
            
            # Tab 3: Label Tool
            with gr.Tab("🏷️ Label Tool"):
                gr.Markdown("## Gán nhãn thủ công cho dataset")
                
                with gr.Row():
                    with gr.Column():
                        data_folder = gr.Textbox(
                            label="Dataset Folder Path",
                            placeholder="Nhập đường dẫn đến folder chứa ảnh cần gán nhãn"
                        )
                        folder_btn = gr.Button("📁 Chọn Folder", variant="secondary")
                        prepare_btn = gr.Button("📁 Prepare Dataset", variant="primary")
                        
                        gr.Markdown("""
                        **Hướng dẫn sử dụng:**
                        1. Chọn folder chứa ảnh cần gán nhãn
                        2. Click "Prepare Dataset" để tạo cấu trúc
                        3. Sử dụng công cụ labeling để gán nhãn từng ảnh
                        4. Labels sẽ được lưu tự động theo format YOLO
                        
                        **Cấu trúc folder sau khi prepare:**
                        ```
                        your_dataset/
                        ├── images/          # Ảnh gốc
                        └── labels/          # Labels (.txt files)
                        ```
                        """)
                        
                    with gr.Column():
                        label_output = gr.Textbox(
                            label="Labeling Output",
                            lines=10,
                            interactive=False
                        )
                
                prepare_btn.click(
                    app.start_labeling,
                    inputs=[data_folder],
                    outputs=label_output
                )
                
                folder_btn.click(
                    app.select_folder,
                    outputs=data_folder
                )
            
            # Tab 4: Combined Training
            with gr.Tab("🔄 Combined Training"):
                gr.Markdown("## Training kết hợp Data gốc + Custom Data")
                
                with gr.Row():
                    with gr.Column():
                        original_data_folder = gr.Textbox(
                            label="Original Dataset Folder",
                            placeholder="Nhập đường dẫn đến dataset gốc"
                        )
                        original_folder_btn = gr.Button("📁 Chọn Original Folder", variant="secondary")
                        custom_data_folder = gr.Textbox(
                            label="Custom Labeled Data Folder",
                            placeholder="Nhập đường dẫn đến custom data đã gán nhãn"
                        )
                        custom_folder_btn = gr.Button("📁 Chọn Custom Folder", variant="secondary")
                        model_size = gr.Dropdown(
                            choices=["s", "m", "l", "x"],
                            value="s",
                            label="Model Size"
                        )
                        epochs = gr.Number(value=30, label="Epochs")
                        imgsz = gr.Number(value=640, label="Image Size")
                        batch = gr.Number(value=16, label="Batch Size")
                        lr = gr.Number(value=0.01, label="Learning Rate")
                        device = gr.Dropdown(
                            choices=["auto", "cpu", "cuda"],
                            value="auto",
                            label="Device"
                        )
                        
                        combined_train_btn = gr.Button("🔄 Start Combined Training", variant="primary")
                        
                        gr.Markdown("""
                        **Combined Training Process:**
                        1. Kết hợp dataset gốc + custom data đã gán nhãn
                        2. Tạo cấu trúc dataset mới tự động
                        3. Training model với data mở rộng
                        4. Model được cải thiện với custom labels
                        """)
                        
                    with gr.Column():
                        combined_output = gr.Textbox(
                            label="Combined Training Output",
                            lines=10,
                            interactive=False
                        )
                
                combined_train_btn.click(
                    app.train_combined_data,
                    inputs=[original_data_folder, custom_data_folder, model_size, epochs, imgsz, batch, lr, device],
                    outputs=combined_output
                )
                
                original_folder_btn.click(
                    app.select_folder,
                    outputs=original_data_folder
                )
                
                custom_folder_btn.click(
                    app.select_folder,
                    outputs=custom_data_folder
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        **Seatbelt Detection - YOLOv11 All-in-One UI**  
        🚗 Phát hiện thắt dây an toàn với YOLOv11  
        📧 Hỗ trợ: [GitHub Issues](https://github.com/your-repo/issues)
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )
