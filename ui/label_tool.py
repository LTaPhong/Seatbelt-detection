"""
Labeling tool for YOLOv11 Seatbelt Detection
"""
import cv2
import numpy as np
import os
import json
from pathlib import Path
import yaml
from datetime import datetime

class SeatbeltLabeler:
    def __init__(self, custom_data_path="custom_data"):
        """
        Initialize labeler
        
        Args:
            custom_data_path: Path to custom data directory
        """
        self.custom_data_path = custom_data_path
        self.images_path = os.path.join(custom_data_path, "images")
        self.labels_path = os.path.join(custom_data_path, "labels")
        self.classes = ['person-seatbelt', 'person-noseatbelt']
        self.current_class = 0
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Create directories
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)
    
    def load_image(self, image_path):
        """
        Load image for labeling
        
        Args:
            image_path: Path to image file
        """
        try:
            if not os.path.exists(image_path):
                print(f"❌ File không tồn tại: {image_path}")
                return False
                
            self.current_image_path = image_path
            self.current_image = cv2.imread(image_path)
            
            if self.current_image is None:
                print(f"❌ Không thể đọc ảnh: {image_path}")
                return False
            
            # Reset annotations for new image
            self.annotations = []
            
            # Load existing annotations if any
            self._load_existing_annotations()
            
            print(f"✅ Đã load ảnh: {os.path.basename(image_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi load ảnh: {str(e)}")
            return False
    
    def _load_existing_annotations(self):
        """Load existing annotations for current image"""
        try:
            if not self.current_image_path:
                return
                
            # Get corresponding label file
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            label_file = os.path.join(self.labels_path, f"{image_name}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                self.annotations = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to absolute coordinates
                        img_h, img_w = self.current_image.shape[:2]
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h
                        
                        annotation = {
                            'class_id': class_id,
                            'class_name': self.classes[class_id] if class_id < len(self.classes) else 'unknown',
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        }
                        self.annotations.append(annotation)
                
                print(f"📝 Đã load {len(self.annotations)} annotations")
            
        except Exception as e:
            print(f"❌ Lỗi load annotations: {str(e)}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for drawing bounding boxes
        
        Args:
            event: Mouse event
            x, y: Mouse coordinates
            flags: Mouse flags
            param: Additional parameters
        """
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
                self.end_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.end_point = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.end_point = (x, y)
                
                # Add annotation
                if self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    
                    # Ensure proper coordinates
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Check if box is large enough
                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                        annotation = {
                            'class_id': self.current_class,
                            'class_name': self.classes[self.current_class],
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        }
                        self.annotations.append(annotation)
                        print(f"✅ Đã thêm annotation: {self.classes[self.current_class]}")
            
        except Exception as e:
            print(f"❌ Lỗi mouse callback: {str(e)}")
    
    def draw_annotations(self, image):
        """
        Draw annotations on image
        
        Args:
            image: Input image
        """
        try:
            result_image = image.copy()
            
            # Draw existing annotations
            for i, annotation in enumerate(self.annotations):
                bbox = annotation['bbox']
                class_name = annotation['class_name']
                class_id = annotation['class_id']
                
                # Get color
                if class_name == 'person-seatbelt':
                    color = (0, 255, 0)  # Green
                elif class_name == 'person-noseatbelt':
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 0, 0)  # Blue
                
                # Draw bounding box
                cv2.rectangle(result_image, 
                            (int(bbox['x1']), int(bbox['y1'])),
                            (int(bbox['x2']), int(bbox['y2'])),
                            color, 2)
                
                # Draw label
                label = f"{class_name} ({i})"
                cv2.putText(result_image, label,
                           (int(bbox['x1']), int(bbox['y1']) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw current drawing box
            if self.drawing and self.start_point and self.end_point:
                color = (0, 255, 255)  # Yellow for current drawing
                cv2.rectangle(result_image, self.start_point, self.end_point, color, 2)
            
            return result_image
            
        except Exception as e:
            print(f"❌ Lỗi vẽ annotations: {str(e)}")
            return image
    
    def save_annotations(self):
        """Save annotations to YOLO format"""
        try:
            if not self.current_image_path or not self.annotations:
                print("❌ Không có annotations để save")
                return False
            
            # Get image name
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            label_file = os.path.join(self.labels_path, f"{image_name}.txt")
            
            # Convert to YOLO format
            img_h, img_w = self.current_image.shape[:2]
            yolo_lines = []
            
            for annotation in self.annotations:
                bbox = annotation['bbox']
                class_id = annotation['class_id']
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (bbox['x1'] + bbox['x2']) / 2 / img_w
                y_center = (bbox['y1'] + bbox['y2']) / 2 / img_h
                width = (bbox['x2'] - bbox['x1']) / img_w
                height = (bbox['y2'] - bbox['y1']) / img_h
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save to file
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            print(f"💾 Đã save annotations: {label_file}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi save annotations: {str(e)}")
            return False
    
    def delete_annotation(self, index):
        """
        Delete annotation by index
        
        Args:
            index: Index of annotation to delete
        """
        try:
            if 0 <= index < len(self.annotations):
                deleted = self.annotations.pop(index)
                print(f"🗑️ Đã xóa annotation: {deleted['class_name']}")
                return True
            else:
                print(f"❌ Index không hợp lệ: {index}")
                return False
        except Exception as e:
            print(f"❌ Lỗi xóa annotation: {str(e)}")
            return False
    
    def set_class(self, class_id):
        """
        Set current class for labeling
        
        Args:
            class_id: Class ID (0 or 1)
        """
        try:
            if 0 <= class_id < len(self.classes):
                self.current_class = class_id
                print(f"🏷️ Đã chọn class: {self.classes[class_id]}")
                return True
            else:
                print(f"❌ Class ID không hợp lệ: {class_id}")
                return False
        except Exception as e:
            print(f"❌ Lỗi set class: {str(e)}")
            return False
    
    def copy_image_to_custom_data(self, source_path):
        """
        Copy image to custom data directory
        
        Args:
            source_path: Source image path
        """
        try:
            if not os.path.exists(source_path):
                print(f"❌ File không tồn tại: {source_path}")
                return False
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_{timestamp}_{os.path.basename(source_path)}"
            dest_path = os.path.join(self.images_path, filename)
            
            # Copy file
            import shutil
            shutil.copy2(source_path, dest_path)
            
            print(f"📁 Đã copy ảnh: {dest_path}")
            return dest_path
            
        except Exception as e:
            print(f"❌ Lỗi copy ảnh: {str(e)}")
            return None
    
    def update_data_yaml(self):
        """Update data.yaml to include custom data"""
        try:
            # Check if custom data exists
            if not os.path.exists(self.images_path) or not os.listdir(self.images_path):
                print("❌ Không có custom data để update")
                return False
            
            # Update data.yaml
            data_yaml_path = "data/data.yaml"
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r') as f:
                    data_config = yaml.safe_load(f)
            else:
                data_config = {
                    'path': './data',
                    'train': 'train/images',
                    'val': 'valid/images',
                    'test': 'test/images',
                    'nc': len(self.classes),
                    'names': self.classes
                }
            
            # Add custom data info
            data_config['custom_data'] = {
                'path': self.custom_data_path,
                'images': 'images',
                'labels': 'labels'
            }
            
            # Save updated config
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            print(f"📝 Đã update data.yaml với custom data")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi update data.yaml: {str(e)}")
            return False
    
    def get_annotation_summary(self):
        """Get summary of annotations"""
        try:
            if not self.current_image_path:
                return None
            
            summary = {
                'image_path': self.current_image_path,
                'total_annotations': len(self.annotations),
                'class_counts': {},
                'annotations': self.annotations
            }
            
            # Count by class
            for annotation in self.annotations:
                class_name = annotation['class_name']
                if class_name not in summary['class_counts']:
                    summary['class_counts'][class_name] = 0
                summary['class_counts'][class_name] += 1
            
            return summary
            
        except Exception as e:
            print(f"❌ Lỗi get summary: {str(e)}")
            return None
    
    def launch_labeling_session(self, image_path):
        """
        Launch interactive labeling session
        
        Args:
            image_path: Path to image to label
        """
        try:
            if not self.load_image(image_path):
                return False
            
            print("\n🎯 LABELING SESSION")
            print("=" * 50)
            print("Hướng dẫn:")
            print("- Click và drag để vẽ bounding box")
            print("- Nhấn 's' để save annotations")
            print("- Nhấn 'd' để delete annotation cuối")
            print("- Nhấn '1' để chọn class: person-seatbelt")
            print("- Nhấn '2' để chọn class: person-noseatbelt")
            print("- Nhấn 'q' để quit")
            print("=" * 50)
            
            cv2.namedWindow('Seatbelt Labeling Tool', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Seatbelt Labeling Tool', self.mouse_callback)
            
            while True:
                # Draw annotations
                display_image = self.draw_annotations(self.current_image)
                
                # Add instructions
                cv2.putText(display_image, f"Class: {self.classes[self.current_class]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_image, f"Annotations: {len(self.annotations)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Seatbelt Labeling Tool', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_annotations()
                elif key == ord('d'):
                    if self.annotations:
                        self.delete_annotation(-1)
                elif key == ord('1'):
                    self.set_class(0)
                elif key == ord('2'):
                    self.set_class(1)
            
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"❌ Lỗi labeling session: {str(e)}")
            return False
