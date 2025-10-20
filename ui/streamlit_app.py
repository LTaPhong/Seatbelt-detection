"""
Seatbelt Detection - YOLOv11 Streamlit App
Real-time training progress with Streamlit
"""
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import threading
import cv2
import numpy as np
from pathlib import Path

# Import our custom modules
try:
    from train_utils import SeatbeltTrainer, download_roboflow_dataset
    from test_utils import SeatbeltTester
    from visual_utils import SeatbeltVisualizer
    from label_tool import SeatbeltLabeler
except ImportError:
    from .train_utils import SeatbeltTrainer, download_roboflow_dataset
    from .test_utils import SeatbeltTester
    from .visual_utils import SeatbeltVisualizer
    from .label_tool import SeatbeltLabeler

# Page config
st.set_page_config(
    page_title="Seatbelt Detection - YOLOv11",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .training-status {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-running {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-completed {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'current_metrics' not in st.session_state:
    st.session_state.current_metrics = {}

class SeatbeltStreamlitApp:
    def __init__(self):
        self.trainer = SeatbeltTrainer()
        self.tester = SeatbeltTester()
        self.visualizer = SeatbeltVisualizer()
        self.labeler = SeatbeltLabeler()
    
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
            
            # Create data.yaml - preserve existing structure if available
            data_yaml_path = os.path.join(folder_path, "data.yaml")
            if os.path.exists(data_yaml_path):
                # Keep existing data.yaml structure
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                data_config['path'] = folder_path
                data_config['train'] = 'train/images'
                data_config['val'] = 'valid/images'
            else:
                # Create new data.yaml with 4 classes
                data_config = {
                    'path': folder_path,
                    'train': 'train/images',
                    'val': 'valid/images',
                    'nc': 4,
                    'names': ['person-noseatbelt', 'person-seatbelt', 'seatbelt', 'windshield']
                }
            
            import yaml
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            return f"✅ Đã tạo cấu trúc YOLO cho dataset:\n📁 {folder_path}\n📊 Training: {len(train_files)} ảnh\n📊 Validation: {len(valid_files)} ảnh\n📄 data.yaml đã được tạo"
            
        except Exception as e:
            return f"❌ Lỗi chuẩn bị dataset: {str(e)}"
    
    def get_training_metrics(self):
        """Get current training metrics from results.csv"""
        try:
            results_path = "runs/train/results.csv"
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                if not df.empty:
                    last_row = df.iloc[-1]
                    return {
                        'epoch': int(last_row.get('epoch', 0)),
                        'loss': last_row.get('train/box_loss', 0.0),
                        'map50': last_row.get('metrics/mAP50(B)', 0.0),
                        'map50_95': last_row.get('metrics/mAP50-95(B)', 0.0),
                        'precision': last_row.get('metrics/precision(B)', 0.0),
                        'recall': last_row.get('metrics/recall(B)', 0.0),
                        'lr': last_row.get('lr/pg0', 0.0)
                    }
            return {'epoch': 0, 'loss': 0.0, 'map50': 0.0, 'map50_95': 0.0, 'precision': 0.0, 'recall': 0.0, 'lr': 0.0}
        except Exception as e:
            st.error(f"Error getting metrics: {e}")
            return {'epoch': 0, 'loss': 0.0, 'map50': 0.0, 'map50_95': 0.0, 'precision': 0.0, 'recall': 0.0, 'lr': 0.0}
    
    def create_training_chart(self):
        """Create training progress chart"""
        try:
            results_path = "runs/train/results.csv"
            if not os.path.exists(results_path):
                return None
            
            df = pd.read_csv(results_path)
            if df.empty:
                return None
            
            # Create subplots
            fig = go.Figure()
            
            # Add traces if columns exist
            if 'train/box_loss' in df.columns:
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['train/box_loss'], 
                                       mode='lines+markers', name='Box Loss', line=dict(color='blue')))
            if 'val/box_loss' in df.columns:
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['val/box_loss'], 
                                       mode='lines+markers', name='Val Box Loss', line=dict(color='red')))
            
            fig.update_layout(
                title='Training Loss',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                height=300
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            return None
    
    def train_model_thread(self, data_folder, model_size, epochs, imgsz, batch, lr, device):
        """Train model in background thread"""
        try:
            # Extract actual path from data_folder
            actual_path = data_folder
            if "✅" in data_folder and ":" in data_folder:
                parts = data_folder.split(":")
                if len(parts) > 1:
                    actual_path = parts[-1].strip()
            
            # Prepare dataset
            prepare_result = self.check_and_prepare_dataset(actual_path)
            if "❌" in prepare_result:
                st.session_state.training_error = prepare_result
                st.session_state.training_active = False
                return
            
            # Create data.yaml path
            data_yaml_path = os.path.join(actual_path, "data.yaml")
            
            # Fix device selection
            if device == "auto":
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            
            # Update trainer
            self.trainer = SeatbeltTrainer(data_path=data_yaml_path, model_size=model_size)
            
            # Start training
            results = self.trainer.start_training(
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                lr=float(lr),
                device=device
            )
            
            st.session_state.training_results = results
            st.session_state.training_active = False
            
        except Exception as e:
            st.session_state.training_error = f"❌ Lỗi training: {str(e)}"
            st.session_state.training_active = False

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Seatbelt Detection - YOLOv11</h1>', unsafe_allow_html=True)
    st.markdown("### Công cụ hoàn chỉnh cho training, testing, visualization và labeling")
    
    # Initialize app
    app = SeatbeltStreamlitApp()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Dataset selection
        st.subheader("📁 Dataset")
        data_folder = st.text_input(
            "Dataset Folder Path",
            value="E:\\Projects\\Seatbelt_detection\\data",
            help="Đường dẫn đến folder chứa dataset"
        )
        
        if st.button("📁 Check Dataset"):
            result = app.check_and_prepare_dataset(data_folder)
            st.info(result)
        
        # Model configuration
        st.subheader("⚙️ Model Configuration")
        model_size = st.selectbox(
            "Model Size",
            options=["s", "m", "l", "x"],
            index=0,
            help="s=small, m=medium, l=large, x=xlarge"
        )
        
        # Training parameters
        st.subheader("🎯 Training Parameters")
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
        imgsz = st.number_input("Image Size", min_value=32, max_value=2048, value=640)
        batch = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
        lr = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001)
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        
        # Training controls
        st.subheader("🚀 Training Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Training", type="primary", disabled=st.session_state.training_active):
                st.session_state.training_active = True
                st.session_state.training_error = None
                
                # Start training in background thread
                thread = threading.Thread(
                    target=app.train_model_thread,
                    args=(data_folder, model_size, epochs, imgsz, batch, lr, device)
                )
                thread.daemon = True
                thread.start()
                st.success("Training started!")
        
        with col2:
            if st.button("⏹️ Stop Training", disabled=not st.session_state.training_active):
                st.session_state.training_active = False
                st.warning("Training stopped!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Training", "🔍 Test / Visualize", "🏷️ Label Tool", "🔄 Combined Training"])
    
    with tab1:
        st.header("🧠 Training Progress")
        
        # Training status
        if st.session_state.training_active:
            st.markdown('<div class="training-status status-running">🔄 Training đang chạy...</div>', unsafe_allow_html=True)
        elif hasattr(st.session_state, 'training_error') and st.session_state.training_error:
            st.markdown(f'<div class="training-status status-error">{st.session_state.training_error}</div>', unsafe_allow_html=True)
        elif st.session_state.training_results:
            st.markdown('<div class="training-status status-completed">✅ Training hoàn thành!</div>', unsafe_allow_html=True)
        else:
            st.info("Chưa có training nào được thực hiện")
        
        # Real-time metrics
        if st.session_state.training_active or st.session_state.training_results:
            metrics = app.get_training_metrics()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Epoch", f"{metrics['epoch']}/{epochs}")
            with col2:
                st.metric("Loss", f"{metrics['loss']:.4f}")
            with col3:
                st.metric("mAP@0.5", f"{metrics['map50']:.4f}")
            with col4:
                st.metric("Learning Rate", f"{metrics['lr']:.6f}")
            
            # Training chart
            chart = app.create_training_chart()
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Auto-refresh when training is active
            if st.session_state.training_active:
                time.sleep(2)
                st.rerun()
    
    with tab2:
        st.header("🔍 Test & Visualization")
        st.info("Test functionality sẽ được implement sau")
    
    with tab3:
        st.header("🏷️ Label Tool")
        st.info("Label tool sẽ được implement sau")
    
    with tab4:
        st.header("🔄 Combined Training")
        st.info("Combined training sẽ được implement sau")

if __name__ == "__main__":
    main()
