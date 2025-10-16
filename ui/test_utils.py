"""
Testing utilities for YOLOv11 Seatbelt Detection
"""
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path

class SeatbeltTester:
    def __init__(self, model_path="models/best.pt"):
        """
        Initialize tester
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.results = None
        
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Model loaded: {self.model_path}")
                return True
            else:
                print(f"❌ Model not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Lỗi load model: {str(e)}")
            return False
    
    def test_single_image(self, image_path, conf_threshold=0.25, save_result=True):
        """
        Test single image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            save_result: Save result image
        """
        if self.model is None:
            if not self.load_model():
                return None
                
        try:
            # Run inference
            results = self.model(image_path, conf=conf_threshold)
            
            # Get result
            result = results[0]
            
            # Create result info
            result_info = {
                'image_path': image_path,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'conf_threshold': conf_threshold,
                'detections': []
            }
            
            # Process detections
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    detection = {
                        'id': i,
                        'class_id': int(cls_id),
                        'class_name': self.model.names[int(cls_id)],
                        'confidence': float(conf),
                        'bbox': {
                            'x1': float(box[0]),
                            'y1': float(box[1]),
                            'x2': float(box[2]),
                            'y2': float(box[3])
                        }
                    }
                    result_info['detections'].append(detection)
            
            # Save result image if requested
            if save_result:
                result_image = result.plot()
                os.makedirs('runs/test_results', exist_ok=True)
                result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                result_path = os.path.join('runs/test_results', result_filename)
                cv2.imwrite(result_path, result_image)
                result_info['result_image'] = result_path
                print(f"📸 Result saved: {result_path}")
            
            self.results = result_info
            return result_info
            
        except Exception as e:
            print(f"❌ Lỗi test single image: {str(e)}")
            return None
    
    def test_folder(self, folder_path, conf_threshold=0.25, save_results=True):
        """
        Test all images in folder
        
        Args:
            folder_path: Path to folder containing images
            conf_threshold: Confidence threshold
            save_results: Save result images
        """
        if self.model is None:
            if not self.load_model():
                return None
                
        try:
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))
                image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            if not image_files:
                print(f"❌ Không tìm thấy ảnh trong folder: {folder_path}")
                return None
            
            print(f"🔍 Tìm thấy {len(image_files)} ảnh trong folder")
            
            # Test each image
            all_results = []
            for i, image_path in enumerate(image_files):
                print(f"📸 Testing {i+1}/{len(image_files)}: {image_path.name}")
                result = self.test_single_image(str(image_path), conf_threshold, save_results)
                if result:
                    all_results.append(result)
            
            # Calculate overall metrics
            metrics = self._calculate_metrics(all_results)
            
            # Save summary
            summary = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'folder_path': folder_path,
                'total_images': len(image_files),
                'processed_images': len(all_results),
                'conf_threshold': conf_threshold,
                'metrics': metrics,
                'results': all_results
            }
            
            # Save to JSON
            os.makedirs('runs/test_results', exist_ok=True)
            summary_file = f"runs/test_results/folder_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Folder test completed. Summary saved: {summary_file}")
            return summary
            
        except Exception as e:
            print(f"❌ Lỗi test folder: {str(e)}")
            return None
    
    def test_random_sample(self, data_path="data", num_samples=10, conf_threshold=0.25):
        """
        Test random sample of images
        
        Args:
            data_path: Path to data directory
            num_samples: Number of random samples
            conf_threshold: Confidence threshold
        """
        try:
            import random
            
            # Get all images from train/valid/test
            all_images = []
            for split in ['train', 'valid', 'test']:
                split_path = os.path.join(data_path, split, 'images')
                if os.path.exists(split_path):
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        all_images.extend(Path(split_path).glob(f"*{ext}"))
                        all_images.extend(Path(split_path).glob(f"*{ext.upper()}"))
            
            if not all_images:
                print(f"❌ Không tìm thấy ảnh trong {data_path}")
                return None
            
            # Random sample
            sample_images = random.sample(all_images, min(num_samples, len(all_images)))
            print(f"🎲 Random sample: {len(sample_images)} ảnh")
            
            # Test sample
            all_results = []
            for i, image_path in enumerate(sample_images):
                print(f"📸 Testing {i+1}/{len(sample_images)}: {image_path.name}")
                result = self.test_single_image(str(image_path), conf_threshold, True)
                if result:
                    all_results.append(result)
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_results)
            
            # Save summary
            summary = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_path': data_path,
                'num_samples': num_samples,
                'actual_samples': len(sample_images),
                'conf_threshold': conf_threshold,
                'metrics': metrics,
                'results': all_results
            }
            
            # Save to JSON
            os.makedirs('runs/test_results', exist_ok=True)
            summary_file = f"runs/test_results/random_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Random test completed. Summary saved: {summary_file}")
            return summary
            
        except Exception as e:
            print(f"❌ Lỗi random test: {str(e)}")
            return None
    
    def _calculate_metrics(self, results):
        """Calculate test metrics"""
        try:
            if not results:
                return {}
            
            total_detections = 0
            class_counts = {}
            confidence_scores = []
            
            for result in results:
                for detection in result['detections']:
                    total_detections += 1
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    confidence_scores.append(confidence)
            
            metrics = {
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / len(results) if results else 0,
                'class_distribution': class_counts,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
                'max_confidence': np.max(confidence_scores) if confidence_scores else 0
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Lỗi tính metrics: {str(e)}")
            return {}
    
    def validate_model(self, val_data_path="data/valid", conf_threshold=0.25):
        """
        Validate model on validation set
        
        Args:
            val_data_path: Path to validation data
            conf_threshold: Confidence threshold
        """
        if self.model is None:
            if not self.load_model():
                return None
                
        try:
            print(f"🔍 Validating model on: {val_data_path}")
            
            # Run validation
            results = self.model.val(
                data=val_data_path,
                conf=conf_threshold,
                save_json=True,
                save_hybrid=False,
                plots=True,
                project='runs/val',
                name=f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            
            # Get metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP50_95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
            }
            
            print(f"📊 Validation metrics:")
            print(f"   - mAP@0.5: {metrics['mAP50']:.3f}")
            print(f"   - mAP@0.5:0.95: {metrics['mAP50_95']:.3f}")
            print(f"   - Precision: {metrics['precision']:.3f}")
            print(f"   - Recall: {metrics['recall']:.3f}")
            print(f"   - F1: {metrics['f1']:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Lỗi validation: {str(e)}")
            return None
