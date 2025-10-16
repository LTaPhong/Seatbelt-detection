"""
Visualization utilities for YOLOv11 Seatbelt Detection
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime
import os

class SeatbeltVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'person-seatbelt': (0, 255, 0),      # Green
            'person-noseatbelt': (0, 0, 255),    # Red
            'default': (255, 0, 0)               # Blue
        }
        
    def draw_detection(self, image, detections, conf_threshold=0.25, 
                      show_labels=True, show_conf=True, line_thickness=2):
        """
        Draw detections on image
        
        Args:
            image: Input image (numpy array)
            detections: List of detection dictionaries
            conf_threshold: Confidence threshold
            show_labels: Show class labels
            show_conf: Show confidence scores
            line_thickness: Line thickness for boxes
        """
        try:
            # Create copy of image
            result_image = image.copy()
            
            for detection in detections:
                if detection['confidence'] < conf_threshold:
                    continue
                    
                # Get bbox coordinates
                x1, y1, x2, y2 = detection['bbox']['x1'], detection['bbox']['y1'], \
                                detection['bbox']['x2'], detection['bbox']['y2']
                
                # Get class info
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Get color
                color = self.colors.get(class_name, self.colors['default'])
                
                # Draw bounding box
                cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                            color, line_thickness)
                
                # Draw label
                if show_labels or show_conf:
                    label = ""
                    if show_labels:
                        label += class_name
                    if show_conf:
                        if label:
                            label += f" {confidence:.2f}"
                        else:
                            label = f"{confidence:.2f}"
                    
                    if label:
                        # Get text size
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 1
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(result_image, 
                                    (int(x1), int(y1) - text_height - baseline),
                                    (int(x1) + text_width, int(y1)),
                                    color, -1)
                        
                        # Draw text
                        cv2.putText(result_image, label,
                                  (int(x1), int(y1) - baseline),
                                  font, font_scale, (255, 255, 255), thickness)
            
            return result_image
            
        except Exception as e:
            print(f"❌ Lỗi vẽ detection: {str(e)}")
            return image
    
    def create_detection_summary(self, results, save_path=None):
        """
        Create detection summary visualization
        
        Args:
            results: List of test results
            save_path: Path to save summary image
        """
        try:
            if not results:
                return None
                
            # Extract data for visualization
            all_detections = []
            for result in results:
                for detection in result.get('detections', []):
                    all_detections.append(detection)
            
            if not all_detections:
                print("❌ Không có detection nào để visualize")
                return None
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Seatbelt Detection Summary', fontsize=16, fontweight='bold')
            
            # 1. Class distribution
            class_names = [d['class_name'] for d in all_detections]
            class_counts = pd.Series(class_names).value_counts()
            
            axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                          colors=['green', 'red'][:len(class_counts)])
            axes[0, 0].set_title('Class Distribution')
            
            # 2. Confidence distribution
            confidences = [d['confidence'] for d in all_detections]
            axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Detections per image
            detections_per_image = [len(result.get('detections', [])) for result in results]
            axes[1, 0].hist(detections_per_image, bins=range(max(detections_per_image) + 2), 
                           alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].set_title('Detections per Image')
            axes[1, 0].set_xlabel('Number of Detections')
            axes[1, 0].set_ylabel('Number of Images')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Confidence by class
            df = pd.DataFrame(all_detections)
            if not df.empty:
                sns.boxplot(data=df, x='class_name', y='confidence', ax=axes[1, 1])
                axes[1, 1].set_title('Confidence by Class')
                axes[1, 1].set_xlabel('Class')
                axes[1, 1].set_ylabel('Confidence Score')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Summary saved: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Lỗi tạo summary: {str(e)}")
            return None
    
    def create_training_curves(self, results_csv_path, save_path=None):
        """
        Create training curves from results.csv
        
        Args:
            results_csv_path: Path to results.csv file
            save_path: Path to save curves
        """
        try:
            if not os.path.exists(results_csv_path):
                print(f"❌ File không tồn tại: {results_csv_path}")
                return None
                
            # Read results
            df = pd.read_csv(results_csv_path)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLOv11 Training Curves', fontsize=16, fontweight='bold')
            
            # Loss curves
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', 
                           color='blue', linewidth=2)
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', 
                           color='red', linewidth=2)
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', 
                           color='blue', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', 
                           color='red', linewidth=2)
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # mAP curves
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', 
                           color='green', linewidth=2)
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', 
                           color='orange', linewidth=2)
            axes[1, 0].set_title('mAP Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Precision/Recall
            axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', 
                           color='purple', linewidth=2)
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', 
                           color='brown', linewidth=2)
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Training curves saved: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Lỗi tạo training curves: {str(e)}")
            return None
    
    def create_metrics_table(self, metrics, save_path=None):
        """
        Create metrics table visualization
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save table
        """
        try:
            if not metrics:
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for table
            table_data = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    table_data.append([key, f"{value:.3f}"])
                else:
                    table_data.append([key, str(value)])
            
            # Create table
            table = ax.table(cellText=table_data,
                           colLabels=['Metric', 'Value'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Color header
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Metrics table saved: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Lỗi tạo metrics table: {str(e)}")
            return None
    
    def create_comparison_grid(self, images_with_results, save_path=None, max_images=9):
        """
        Create comparison grid of images with detections
        
        Args:
            images_with_results: List of (image_path, result) tuples
            save_path: Path to save grid
            max_images: Maximum number of images to show
        """
        try:
            if not images_with_results:
                return None
                
            # Limit number of images
            images_to_show = images_with_results[:max_images]
            n_images = len(images_to_show)
            
            # Calculate grid size
            cols = min(3, n_images)
            rows = (n_images + cols - 1) // cols
            
            # Create figure
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, (image_path, result) in enumerate(images_to_show):
                if i >= max_images:
                    break
                    
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Draw detections
                if result and 'detections' in result:
                    image = self.draw_detection(image, result['detections'])
                
                # Show image
                axes[i].imshow(image)
                axes[i].set_title(f"{os.path.basename(image_path)}", fontsize=10)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Comparison grid saved: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Lỗi tạo comparison grid: {str(e)}")
            return None
