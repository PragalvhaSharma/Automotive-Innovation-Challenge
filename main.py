import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
from dataset import ThermalDataset, collate_fn

class LightweightDetector(nn.Module):
    def __init__(self, num_classes):
        super(LightweightDetector, self).__init__()
        # Use MobileNetV3 as backbone (very efficient for embedded systems)
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        
        # Remove the last layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Detection head
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4 + num_classes)  # 4 bbox coords + class scores
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.detector(features)
        # Split output into bbox and class predictions
        bbox = output[:, :4]  # First 4 values are bbox coordinates [x_min, x_max, y_min, y_max]
        classes = output[:, 4:]  # Remaining values are class scores
        return bbox, classes

class ThermalObjectDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Update class mapping to match dataset
        self.classes = {
            1: 'person',
            2: 'bicycle',
            3: 'vehicle'
        }
        self.num_classes = len(self.classes)
        
        # Initialize model
        self.model = LightweightDetector(self.num_classes).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
    def preprocess_image(self, image):
        # Resize to a standard size
        image = cv2.resize(image, (224, 224))
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
        
    def detect_objects(self, image):
        with torch.no_grad():
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get predictions
            bbox_pred, class_pred = self.model(processed_image)
            
            # Get class probabilities
            class_probs = torch.softmax(class_pred, dim=1)
            
            # Get predicted class and confidence
            confidence, class_idx = class_probs.max(1)
            class_idx = class_idx.item() + 1  # Add 1 because our classes start from 1
            confidence = confidence.item()
            
            if confidence > 0.5:  # Confidence threshold
                # Convert relative coordinates to absolute
                h, w = image.shape[:2]
                bbox = bbox_pred[0].cpu().numpy()
                x_min = int(bbox[0] * w)  # x_min
                x_max = int(bbox[1] * w)  # x_max
                y_min = int(bbox[2] * h)  # y_min
                y_max = int(bbox[3] * h)  # y_max
                
                return [(class_idx, confidence, (x_min, y_min, x_max, y_max))]
            
            return []
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Convert to [x1, y1, x2, y2] format
        box1 = [box1[0], box1[2], box1[1], box1[3]]  # [x_min, y_min, x_max, y_max]
        box2 = [box2[0], box2[2], box2[1], box2[3]]  # [x_min, y_min, x_max, y_max]
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def evaluate(self, val_loader, iou_threshold=0.5):
        """Evaluate model performance"""
        self.model.eval()
        total_correct = 0
        total_predictions = 0
        total_ground_truth = 0
        class_metrics = {cls_id: {'tp': 0, 'fp': 0, 'fn': 0} for cls_id in self.classes}
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                target_boxes = batch['boxes']
                target_classes = batch['classes']
                
                # Get predictions
                pred_boxes, pred_classes = self.model(images)
                pred_boxes = pred_boxes.cpu()
                pred_classes = torch.softmax(pred_classes, dim=1).cpu()
                
                # Process each image in batch
                for i in range(len(images)):
                    pred_conf, pred_cls = pred_classes[i].max(0)
                    pred_cls = pred_cls.item() + 1  # Add 1 as classes start from 1
                    pred_box = pred_boxes[i]
                    target_box = target_boxes[i]
                    target_cls = target_classes[i].item()
                    
                    if pred_conf > 0.5:  # Confidence threshold
                        iou = self.calculate_iou(pred_box, target_box)
                        
                        if iou >= iou_threshold and pred_cls == target_cls:
                            class_metrics[target_cls]['tp'] += 1
                            total_correct += 1
                        else:
                            class_metrics[pred_cls]['fp'] += 1
                    else:
                        class_metrics[target_cls]['fn'] += 1
                    
                    total_predictions += 1
                    total_ground_truth += 1
        
        # Calculate metrics
        metrics = {}
        overall_precision = total_correct / total_predictions if total_predictions > 0 else 0
        overall_recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        }
        
        # Per-class metrics
        for cls_id in self.classes:
            tp = class_metrics[cls_id]['tp']
            fp = class_metrics[cls_id]['fp']
            fn = class_metrics[cls_id]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[self.classes[cls_id]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics

    def train(self, train_dataset, val_dataset, num_epochs=10, batch_size=32, learning_rate=0.001, target_accuracy=0.98):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, collate_fn=collate_fn)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                             factor=0.5, patience=5, verbose=True)
        bbox_criterion = nn.MSELoss()
        class_criterion = nn.CrossEntropyLoss()
        
        best_val_metrics = None
        best_model_state = None
        epochs_without_improvement = 0
        max_epochs_without_improvement = 10
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                images = batch['images'].to(self.device)
                target_boxes = batch['boxes'].to(self.device)
                target_classes = batch['classes'].to(self.device) - 1
                
                optimizer.zero_grad()
                pred_boxes, pred_classes = self.model(images)
                
                bbox_loss = bbox_criterion(pred_boxes, target_boxes)
                class_loss = class_criterion(pred_classes, target_classes)
                loss = bbox_loss + class_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            metrics = self.evaluate(val_loader)
            overall_accuracy = metrics['overall']['precision']
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Overall Accuracy: {overall_accuracy:.4f}")
            
            # Print per-class metrics
            for cls_name, cls_metrics in metrics.items():
                if cls_name != 'overall':
                    print(f"\n{cls_name}:")
                    print(f"Precision: {cls_metrics['precision']:.4f}")
                    print(f"Recall: {cls_metrics['recall']:.4f}")
                    print(f"F1 Score: {cls_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(overall_accuracy)
            
            # Save best model and check for early stopping
            if best_val_metrics is None or overall_accuracy > best_val_metrics['overall']['precision']:
                best_val_metrics = metrics
                best_model_state = self.model.state_dict()
                epochs_without_improvement = 0
                print("\nNew best model saved!")
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            if epochs_without_improvement >= max_epochs_without_improvement:
                print("\nEarly stopping triggered!")
                break
            
            # Check if target accuracy is reached
            if overall_accuracy >= target_accuracy:
                print(f"\nReached target accuracy of {target_accuracy}!")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            return best_val_metrics
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

class ThermalCamera:
    def __init__(self, url):
        self.url = url
        
    def get_frame(self):
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                # Convert string of image data to uint8
                nparr = np.frombuffer(response.content, np.uint8)
                # Decode image
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                return image
            else:
                print(f"Failed to get image: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting frame: {str(e)}")
            return None

def main():
    # Load environment variables
    load_dotenv()
    
    # Get camera URL from environment variable
    camera_url = os.getenv('THERMAL_CAMERA_URL')
    if not camera_url:
        print("Error: THERMAL_CAMERA_URL not set in environment variables")
        return
    
    # Initialize camera and detector
    camera = ThermalCamera(camera_url)
    detector = ThermalObjectDetector()
    
    print("Starting thermal object detection...")
    while True:
        # Get frame from camera
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Process detections
        for class_id, confidence, bbox in detections:
            x_min, y_min, x_max, y_max = bbox
            class_name = detector.classes[class_id]
            print(f"Detected {class_name} with confidence {confidence:.2f} at {bbox}")
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                       (int(x_min), int(y_min)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame (for debugging)
        cv2.imshow('Thermal Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
