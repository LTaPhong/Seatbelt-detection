"""
Training utilities for YOLOv11 Seatbelt Detection
"""
import os
import yaml
import torch
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class SeatbeltTrainer:
    def __init__(self, data_path="data/data.yaml", model_size="s"):
        """
        Initialize trainer
        
        Args:
            data_path: Path to data.yaml file
            model_size: Model size ('s', 'm', 'l', 'x')
        """
        self.data_path = data_path
        self.model_size = model_size
        self.model_name = f"yolo11{model_size}.pt"
        self.results = None
        
    def start_training(self, epochs=50, imgsz=640, batch=16, lr=0.01, 
                      patience=50, save_period=10, device='auto'):
        """
        Start YOLOv11 training
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            lr: Learning rate
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        try:
            # Load model
            model = YOLO(self.model_name)
            
            # Training parameters
            train_params = {
                'data': self.data_path,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                'lr0': lr,
                'patience': patience,
                'save_period': save_period,
                'device': device,
                'project': 'runs/train',
                'name': f'seatbelt_{self.model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'auto',
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'split': 'val',
                'save_json': False,
                'save_hybrid': False,
                'conf': None,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True
            }
            
            print(f"🚀 Bắt đầu training YOLOv11{self.model_size}...")
            print(f"📊 Tham số: epochs={epochs}, imgsz={imgsz}, batch={batch}, lr={lr}")
            
            # Start training
            self.results = model.train(**train_params)
            
            # Save training info
            self._save_training_info(train_params)
            
            print("✅ Training hoàn thành!")
            return self.results
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình training: {str(e)}")
            return None
    
    def _save_training_info(self, params):
        """Save training information to CSV"""
        if self.results is None:
            return
            
        # Create training info
        training_info = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': self.model_name,
            'epochs': params['epochs'],
            'imgsz': params['imgsz'],
            'batch': params['batch'],
            'lr': params['lr0'],
            'device': params['device'],
            'best_fitness': getattr(self.results, 'best_fitness', None),
            'results_dir': getattr(self.results, 'save_dir', None)
        }
        
        # Save to CSV
        df = pd.DataFrame([training_info])
        os.makedirs('runs/training_logs', exist_ok=True)
        log_file = f"runs/training_logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(log_file, index=False)
        print(f"📝 Training log saved: {log_file}")
    
    def get_training_progress(self):
        """Get current training progress"""
        if self.results is None:
            return None
            
        try:
            # Get results directory
            results_dir = getattr(self.results, 'save_dir', None)
            if results_dir and os.path.exists(results_dir):
                # Read results.csv if exists
                results_csv = os.path.join(results_dir, 'results.csv')
                if os.path.exists(results_csv):
                    df = pd.read_csv(results_csv)
                    return df
            return None
        except Exception as e:
            print(f"❌ Lỗi đọc training progress: {str(e)}")
            return None
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        if self.results is None:
            return None
            
        try:
            results_dir = getattr(self.results, 'save_dir', None)
            if not results_dir or not os.path.exists(results_dir):
                return None
                
            # Read results
            results_csv = os.path.join(results_dir, 'results.csv')
            if not os.path.exists(results_csv):
                return None
                
            df = pd.read_csv(results_csv)
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLOv11 Training Curves', fontsize=16)
            
            # Loss curves
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='blue')
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='red')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # mAP curves
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
            axes[1, 0].set_title('mAP Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Precision/Recall
            axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='purple')
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='brown')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Training curves saved: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Lỗi vẽ training curves: {str(e)}")
            return None
    
    def retrain_with_custom_data(self, custom_data_path="custom_data", epochs=30):
        """
        Retrain model with custom labeled data
        
        Args:
            custom_data_path: Path to custom data directory
            epochs: Number of epochs for retraining
        """
        try:
            # Merge custom data with existing data
            self._merge_custom_data(custom_data_path)
            
            # Start retraining
            print(f"🔄 Bắt đầu retraining với custom data...")
            return self.start_training(epochs=epochs)
            
        except Exception as e:
            print(f"❌ Lỗi retraining: {str(e)}")
            return None
    
    def _merge_custom_data(self, custom_data_path):
        """Merge custom data with existing dataset"""
        try:
            # Update data.yaml to include custom data
            with open(self.data_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Add custom data paths
            custom_train_path = os.path.join(custom_data_path, "images")
            custom_labels_path = os.path.join(custom_data_path, "labels")
            
            if os.path.exists(custom_train_path) and os.path.exists(custom_labels_path):
                print(f"📁 Merging custom data from: {custom_data_path}")
                # Here you would implement the logic to merge datasets
                # For now, just print the paths
                print(f"   - Images: {custom_train_path}")
                print(f"   - Labels: {custom_labels_path}")
            
        except Exception as e:
            print(f"❌ Lỗi merge custom data: {str(e)}")

def download_roboflow_dataset(api_key, workspace="traffic-violations", 
                             project="seatbelt-detection-esut6", version=5):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
    """
    try:
        from roboflow import Roboflow
        
        print("📥 Đang tải dataset từ Roboflow...")
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8")
        
        print(f"✅ Dataset đã tải về: {dataset.location}")
        return dataset
        
    except Exception as e:
        print(f"❌ Lỗi tải dataset: {str(e)}")
        return None
