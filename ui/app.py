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
import threading

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
        
        # Training control variables
        self.training_active = False
        self.training_paused = False
        self.training_stopped = False
        self.current_training_process = None
        
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
        
        # Check data folder - extract actual path if it contains status message
        actual_path = data_folder
        if "✅" in data_folder and ":" in data_folder:
            # Extract path from status message like "✅ Dataset đã có cấu trúc YOLO hợp lệ: E:\Projects\Seatbelt_detection\data"
            parts = data_folder.split(":")
            if len(parts) > 1:
                actual_path = parts[-1].strip()
        
        if not actual_path or not os.path.exists(actual_path):
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
    
    def train_model(self, data_folder, model_size, epochs, imgsz, batch, lr, device):
        """Train YOLOv11 model with selected dataset"""
        try:
            # Check if training is already active
            if self.training_active:
                yield ("⚠️ Training đang chạy! Vui lòng dừng training hiện tại trước khi bắt đầu mới.", 
                        "Training đang chạy", "🔄 Training in progress", 0, 0.0, 0.0, 0.0, None,
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=False), gr.update(scale=1))
                return
            
            # Extract actual path from data_folder (might contain status message)
            actual_path = data_folder
            if "✅" in data_folder and ":" in data_folder:
                parts = data_folder.split(":")
                if len(parts) > 1:
                    actual_path = parts[-1].strip()
            
            # Validate inputs
            validation_errors = self.validate_training_inputs(actual_path, model_size, epochs, imgsz, batch, lr, device)
            if validation_errors:
                yield ("\n".join(validation_errors), "Chưa có model nào được train", 
                        "❌ Validation failed", 0, 0.0, 0.0, 0.0, None,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=True), gr.update(scale=1))
                return
            
            # Prepare dataset structure
            prepare_result = self.prepare_dataset_structure(actual_path)
            if "❌" in prepare_result:
                yield (prepare_result, "Chưa có model nào được train", 
                        "❌ Dataset preparation failed", 0, 0.0, 0.0, 0.0, None,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=True), gr.update(scale=1))
                return
            
            # Create data.yaml path
            data_yaml_path = os.path.join(actual_path, "data.yaml")
            
            # Fix data.yaml paths to use absolute paths
            self._fix_data_yaml_paths(data_yaml_path, actual_path)
            
            # Update trainer with new parameters
            self.trainer = SeatbeltTrainer(data_path=data_yaml_path, model_size=model_size)
            
            # Set training state
            self.training_active = True
            self.training_paused = False
            self.training_stopped = False
            
            # Fix device selection - use CPU if CUDA not available
            if device == "auto":
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                    print("⚠️ CUDA không khả dụng, sử dụng CPU")
            
            # Show initial status
            yield (f"🚀 Training đã bắt đầu!\n\n📊 Thông tin training:\n- Model: YOLOv11-{model_size}\n- Epochs: {epochs}\n- Image Size: {imgsz}\n- Batch Size: {batch}\n- Learning Rate: {lr}\n- Device: {device}\n\n⏳ Đang training...", 
                    f"🔄 Training đang chạy...\n📊 YOLOv11-{model_size} | Epochs: {epochs} | Size: {imgsz}\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    "🔄 Training in progress...", 
                    0, 0.0, 0.0, float(lr), None,
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=False), gr.update(scale=1))
            
            # Start training with progress updates
            results = self.trainer.start_training(
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                lr=float(lr),
                device=device
            )
            
            # Reset training state
            self.training_active = False
            
            if results and not self.training_stopped:
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
                
                # Get final metrics and create chart
                final_metrics = self._get_training_metrics()
                training_chart = self._create_training_chart()
                
                yield (output_text, model_status, "✅ Training completed successfully!", 
                        int(epochs), final_metrics.get('loss', 0.0), 
                        final_metrics.get('map50', 0.0), float(lr), training_chart,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=True), gr.update(scale=1))
            elif self.training_stopped:
                yield ("⏹️ Training đã được dừng bởi người dùng.", 
                        "Training stopped", "⏹️ Training stopped by user", 
                        0, 0.0, 0.0, 0.0, None,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=True), gr.update(scale=1))
            else:
                yield ("❌ Training thất bại. Vui lòng kiểm tra log để biết lỗi chi tiết.", 
                        "Chưa có model nào được train", "❌ Training failed", 
                        0, 0.0, 0.0, 0.0, None,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=True), gr.update(scale=1))
                
        except Exception as e:
            self.training_active = False
            error_msg = f"❌ Lỗi training: {str(e)}\n\n💡 Gợi ý:\n- Kiểm tra đường dẫn dataset\n- Đảm bảo có đủ RAM/VRAM\n- Thử giảm batch size nếu bị lỗi memory"
            yield (error_msg, "Chưa có model nào được train", "❌ Training error", 
                    0, 0.0, 0.0, 0.0, None,
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=True), gr.update(scale=1))

    
    def get_training_progress(self):
        """Get current training progress for real-time updates"""
        try:
            # Check if training is running by looking for results.csv
            results_path = "runs/train/results.csv"
            if not os.path.exists(results_path):
                return ("Training not started", 0, 0.0, 0.0, 0.0, None)
            
            # Try to get latest metrics from results.csv
            metrics = self._get_training_metrics()
            
            # Create updated chart
            chart = self._create_training_chart()
            
            # Get current epoch from results
            current_epoch = 0
            try:
                import pandas as pd
                df = pd.read_csv(results_path)
                if not df.empty:
                    current_epoch = int(df.iloc[-1]['epoch'])
            except:
                pass
            
            # Check if training is still active
            if current_epoch > 0 and current_epoch < 50:
                status = f"🔄 Training Epoch {current_epoch}/50 - Loss: {metrics.get('loss', 0.0):.4f}"
            elif current_epoch >= 50:
                status = f"✅ Training Completed - Final Loss: {metrics.get('loss', 0.0):.4f}"
                self.training_active = False
            else:
                status = "⏳ Training starting..."
            
            return (status, current_epoch, metrics.get('loss', 0.0), 
                    metrics.get('map50', 0.0), metrics.get('lr', 0.0), chart)
                    
        except Exception as e:
            return (f"❌ Error: {str(e)}", 0, 0.0, 0.0, 0.0, None)
    
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
    
    def _fix_data_yaml_paths(self, data_yaml_path, dataset_path):
        """Fix data.yaml to use absolute paths and validate dataset"""
        try:
            import yaml
            
            # Read current data.yaml
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # Update paths to absolute
            data_config['path'] = os.path.abspath(dataset_path)
            data_config['train'] = 'train/images'
            data_config['val'] = 'valid/images'
            data_config['test'] = 'test/images'
            
            # Validate and fix labels
            self._validate_and_fix_labels(dataset_path)
            
            # Write back to data.yaml
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
                
        except Exception as e:
            print(f"Warning: Could not fix data.yaml paths: {e}")
    
    def _validate_and_fix_labels(self, dataset_path):
        """Validate and fix label files to ensure correct class indices"""
        try:
            import glob
            
            # Find all label files
            train_labels_path = os.path.join(dataset_path, "train", "labels")
            val_labels_path = os.path.join(dataset_path, "valid", "labels")
            
            label_paths = []
            if os.path.exists(train_labels_path):
                label_paths.extend(glob.glob(os.path.join(train_labels_path, "*.txt")))
            if os.path.exists(val_labels_path):
                label_paths.extend(glob.glob(os.path.join(val_labels_path, "*.txt")))
            
            print(f"🔍 Checking {len(label_paths)} label files...")
            
            fixed_count = 0
            for label_path in label_paths:
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    fixed_lines = []
                    file_changed = False
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Valid YOLO format
                            class_id = int(parts[0])
                            # Fix class 3 to class 1 (assuming class 3 is person-noseatbelt)
                            if class_id == 3:
                                parts[0] = '1'  # Map to person-noseatbelt
                                file_changed = True
                            elif class_id > 1:
                                # Skip invalid classes
                                continue
                            
                            fixed_lines.append(' '.join(parts) + '\n')
                        else:
                            # Skip invalid lines
                            continue
                    
                    if file_changed:
                        with open(label_path, 'w') as f:
                            f.writelines(fixed_lines)
                        fixed_count += 1
                        
                except Exception as e:
                    print(f"Warning: Could not process {label_path}: {e}")
                    continue
            
            if fixed_count > 0:
                print(f"✅ Fixed {fixed_count} label files with invalid class indices")
            else:
                print("✅ All label files are valid")
                
        except Exception as e:
            print(f"Warning: Could not validate labels: {e}")
    
    def _get_training_metrics(self):
        """Get training metrics from results"""
        try:
            # Look for results.csv in runs/train
            results_path = "runs/train/results.csv"
            if os.path.exists(results_path):
                import pandas as pd
                df = pd.read_csv(results_path)
                if not df.empty:
                    last_row = df.iloc[-1]
                    return {
                        'loss': last_row.get('train/box_loss', 0.0),
                        'map50': last_row.get('metrics/mAP50(B)', 0.0),
                        'map50_95': last_row.get('metrics/mAP50-95(B)', 0.0),
                        'precision': last_row.get('metrics/precision(B)', 0.0),
                        'recall': last_row.get('metrics/recall(B)', 0.0)
                    }
            return {'loss': 0.0, 'map50': 0.0, 'map50_95': 0.0, 'precision': 0.0, 'recall': 0.0}
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return {'loss': 0.0, 'map50': 0.0, 'map50_95': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    def _create_training_chart(self):
        """Create training progress chart"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            results_path = "runs/train/results.csv"
            if not os.path.exists(results_path):
                return None
            
            df = pd.read_csv(results_path)
            if df.empty:
                return None
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
            
            # Plot loss
            if 'train/box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Box Loss')
                ax1.plot(df['epoch'], df['val/box_loss'], 'r-', label='Val Box Loss')
                ax1.set_title('Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
            
            # Plot mAP
            if 'metrics/mAP50(B)' in df.columns:
                ax2.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', label='mAP@0.5')
                ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'orange', label='mAP@0.5:0.95')
                ax2.set_title('mAP')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('mAP')
                ax2.legend()
                ax2.grid(True)
            
            # Plot Precision/Recall
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                ax3.plot(df['epoch'], df['metrics/precision(B)'], 'purple', label='Precision')
                ax3.plot(df['epoch'], df['metrics/recall(B)'], 'brown', label='Recall')
                ax3.set_title('Precision & Recall')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Score')
                ax3.legend()
                ax3.grid(True)
            
            # Plot Learning Rate
            if 'lr/pg0' in df.columns:
                ax4.plot(df['epoch'], df['lr/pg0'], 'red', label='Learning Rate')
                ax4.set_title('Learning Rate')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('LR')
                ax4.legend()
                ax4.grid(True)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
    

            
        except Exception as e:
            print(f"Error creating empty chart: {e}")
            return None
    
    def refresh_training_status(self):
        """Refresh training status and metrics"""
        try:
            metrics = self._get_training_metrics()
            chart = self._create_training_chart()
            
            status = "✅ Training completed" if metrics['loss'] > 0 else "Chưa bắt đầu training"
            
            return (status, 
                    int(metrics.get('epoch', 0)),
                    float(metrics.get('loss', 0.0)),
                    float(metrics.get('map50', 0.0)),
                    float(metrics.get('lr', 0.0)),
                    chart)
        except Exception as e:
            return (f"❌ Error: {str(e)}", 0, 0.0, 0.0, 0.0, None)
    
    def stop_training(self):
        """Stop current training"""
        try:
            if self.training_active:
                self.training_stopped = True
                self.training_active = False
                self.training_paused = False
                
                # Try to stop the training process if it exists
                if self.current_training_process:
                    self.current_training_process.terminate()
                    self.current_training_process = None
                
                return ("⏹️ Training đã được dừng thành công!", 
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            else:
                return ("⚠️ Không có training nào đang chạy để dừng.",
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        except Exception as e:
            return (f"❌ Lỗi dừng training: {str(e)}",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    
    def pause_training(self):
        """Pause current training"""
        try:
            if self.training_active and not self.training_paused:
                self.training_paused = True
                return ("⏸️ Training đã được tạm dừng. Sử dụng 'Resume' để tiếp tục.",
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
            elif self.training_paused:
                return ("⚠️ Training đã được tạm dừng rồi.",
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
            else:
                return ("⚠️ Không có training nào đang chạy để tạm dừng.",
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        except Exception as e:
            return (f"❌ Lỗi tạm dừng training: {str(e)}",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    
    def resume_training(self):
        """Resume paused training"""
        try:
            if self.training_paused:
                self.training_paused = False
                return ("▶️ Training đã được tiếp tục!",
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))
            elif self.training_active:
                return ("⚠️ Training đang chạy bình thường.",
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))
            else:
                return ("⚠️ Không có training nào để tiếp tục.",
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        except Exception as e:
            return (f"❌ Lỗi tiếp tục training: {str(e)}",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    
    def get_training_control_state(self):
        """Get current training control button states"""
        if self.training_active:
            if self.training_paused:
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
            else:
                return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
        else:
            return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    
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
                    with gr.Column(scale=1) as left_column:
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
                        with gr.Row():
                            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                            stop_btn = gr.Button("⏹️ Stop Training", variant="stop", size="lg", visible=False)
                        with gr.Row():
                            resume_btn = gr.Button("▶️ Resume Training", variant="secondary", size="lg", visible=False)
                            pause_btn = gr.Button("⏸️ Pause Training", variant="secondary", size="lg", visible=False)
                        
                        # Model Status
                        gr.Markdown("### 📊 Model Status")
                        model_status = gr.Textbox(
                            label="Current Model",
                            value="Chưa có model nào được train",
                            interactive=False,
                            lines=2
                        )
                        
                        # Quick Actions
                        gr.Markdown("### ⚡ Quick Actions")
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Output", variant="secondary", size="sm")
                            open_results_btn = gr.Button("📁 Open Results", variant="secondary", size="sm")
                            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                    
                    with gr.Column(scale=2) as right_column:
                        # Training Progress Section (hidden initially)
                        with gr.Group(visible=False) as training_progress_group:
                            gr.Markdown("### 📈 Training Progress")
                            training_progress = gr.Progress()
                            
                            # Real-time Training Status
                            training_status = gr.Textbox(
                                label="Current Status",
                                value="Chưa bắt đầu training",
                                interactive=False,
                                lines=2
                            )
                            
                            # Training Metrics
                            gr.Markdown("### 📊 Training Metrics")
                            with gr.Row():
                                current_epoch = gr.Number(label="Epoch", value=0, interactive=False)
                                current_loss = gr.Number(label="Loss", value=0.0, interactive=False, precision=4)
                            with gr.Row():
                                current_map = gr.Number(label="mAP@0.5", value=0.0, interactive=False, precision=4)
                                current_lr = gr.Number(label="Learning Rate", value=0.0, interactive=False, precision=6)
                            
                            # Training Charts
                            gr.Markdown("### 📈 Training Charts")
                            training_chart = gr.Plot(label="Training Progress Chart")
                        
                        # Training Output (only when not training)
                        with gr.Group(visible=True) as training_output_group:
                            training_output = gr.Textbox(
                                label="Training Output",
                                lines=15,
                                interactive=False,
                                placeholder="Kết quả training sẽ hiển thị ở đây..."
                            )
                
                # Training event handlers
                train_btn.click(
                    app.train_model,
                    inputs=[data_folder, model_size, epochs, imgsz, batch, lr, device],
                    outputs=[training_output, model_status, training_status, current_epoch, current_loss, current_map, current_lr, training_chart, train_btn, stop_btn, resume_btn, pause_btn, training_progress_group, training_output_group, left_column]
                )
                
                stop_btn.click(
                    app.stop_training,
                    outputs=[training_output, train_btn, stop_btn, resume_btn, pause_btn]
                )
                
                pause_btn.click(
                    app.pause_training,
                    outputs=[training_output, train_btn, stop_btn, resume_btn, pause_btn]
                )
                
                resume_btn.click(
                    app.resume_training,
                    outputs=[training_output, train_btn, stop_btn, resume_btn, pause_btn]
                )
                
                # Folder selection
                folder_btn.click(
                    app.select_folder,
                    outputs=data_folder
                )
                
                # Quick actions
                clear_btn.click(
                    lambda: ("", "Chưa có model nào được train", "Chưa bắt đầu training", 0, 0.0, 0.0, 0.0, None, gr.update(visible=False), gr.update(visible=True), gr.update(scale=1)),
                    outputs=[training_output, model_status, training_status, current_epoch, current_loss, current_map, current_lr, training_chart, training_progress_group, training_output_group, left_column]
                )
                
                open_results_btn.click(
                    app.open_training_results,
                    outputs=training_output
                )
                
                # Manual refresh button for real-time updates
                def manual_refresh():
                    if app.training_active:
                        return app.get_training_progress()
                    else:
                        return app.refresh_training_status()
                
                refresh_btn.click(
                    manual_refresh,
                    outputs=[training_status, current_epoch, current_loss, current_map, current_lr, training_chart]
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
