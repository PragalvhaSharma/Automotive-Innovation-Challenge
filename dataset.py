import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ThermalDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, image_size=(224, 224)):
        """
        Args:
            img_dir (string): Directory with all the images
            label_file (string): Path to the label file
            transform (callable, optional): Optional transform to be applied on a sample
            image_size (tuple): Target size for image resizing (width, height)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_size = image_size
        self.class_names = {1: "Person", 2: "Bicycle", 3: "Vehicle"}
        self.class_colors = {1: 'red', 2: 'blue', 3: 'green'}
        
        # Group labels by image
        self.image_labels = {}
        invalid_entries = []
        fixed_entries = []
        
        # Read labels file
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 6:
                    image_name = parts[0]
                    try:
                        class_id = int(parts[1])
                        # Read coordinates in x_min, y_min, x_max, y_max order
                        x_min = float(parts[2])
                        y_min = float(parts[3])
                        x_max = float(parts[4])
                        y_max = float(parts[5])
                        
                        # Validate class
                        if class_id not in self.class_names:
                            invalid_entries.append((line_num, f"Invalid class ID: {class_id}"))
                            continue
                        
                        # Fix swapped coordinates if needed
                        if x_min > x_max:
                            x_min, x_max = x_max, x_min
                            fixed_entries.append((line_num, f"Fixed swapped x coordinates"))
                        if y_min > y_max:
                            y_min, y_max = y_max, y_min
                            fixed_entries.append((line_num, f"Fixed swapped y coordinates"))
                        
                        # Ensure coordinates are valid
                        if x_min < 0 or y_min < 0:
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            fixed_entries.append((line_num, f"Fixed negative coordinates to 0"))
                        
                        # Group by image
                        if image_name not in self.image_labels:
                            self.image_labels[image_name] = []
                            
                        self.image_labels[image_name].append({
                            'box': [x_min, y_min, x_max, y_max],  # Store as [x_min, y_min, x_max, y_max]
                            'class': class_id
                        })
                        
                    except ValueError:
                        invalid_entries.append((line_num, "Invalid numeric values"))
                else:
                    invalid_entries.append((line_num, f"Wrong number of values: expected 6, got {len(parts)}"))
        
        # Convert to list format for indexing
        self.images = list(self.image_labels.keys())
        
        # Print dataset statistics
        total_objects = sum(len(labels) for labels in self.image_labels.values())
        print(f"\nDataset Statistics:")
        print(f"Total images: {len(self.images)}")
        print(f"Total objects: {total_objects}")
        print(f"Average objects per image: {total_objects/len(self.images):.2f}")
        
        # Print class distribution
        class_counts = Counter()
        for labels in self.image_labels.values():
            class_counts.update(obj['class'] for obj in labels)
            
        print("\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / total_objects) * 100
            print(f"Class {class_id} ({self.class_names[class_id]}): {count} objects ({percentage:.1f}%)")
        
        if fixed_entries:
            print(f"\nFixed {len(fixed_entries)} coordinate issues")
            
        if invalid_entries:
            print(f"\nFound {len(invalid_entries)} invalid entries")
            
        # Verify images exist
        missing_images = [img for img in self.images if not os.path.exists(os.path.join(img_dir, img))]
        if missing_images:
            print(f"\nWarning: {len(missing_images)} images not found in {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name = self.images[idx]
        img_path = os.path.join(self.img_dir, image_name)
        
        # Load original image without modifications
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Get all objects for this image
        labels = self.image_labels[image_name]
        
        # Keep original boxes and classes
        boxes = [label['box'] for label in labels]
        classes = [label['class'] for label in labels]
        
        # Convert to tensors without normalization
        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)
        image = torch.from_numpy(image)
        
        sample = {
            'image': image,
            'boxes': boxes,
            'classes': classes,
            'image_name': image_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def collate_fn(batch):
    """Custom collate function for DataLoader that handles multiple objects per image"""
    images = []
    boxes = []
    classes = []
    image_names = []
    
    for item in batch:
        # Handle grayscale images
        if len(item['image'].shape) == 2:
            image = item['image'].unsqueeze(0)  # Add channel dimension
        else:
            image = item['image'].permute(2, 0, 1)  # CHW format
            
        images.append(image)
        boxes.append(item['boxes'])  # [x_min, x_max, y_min, y_max] format
        classes.append(item['classes'])
        image_names.append(item['image_name'])
    
    return {
        'images': torch.stack(images),
        'boxes': boxes,  # Keep as list since each image can have different number of boxes
        'classes': classes,  # Keep as list since each image can have different number of classes
        'image_names': image_names
    }

if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Test parameters
    img_dir = "/Users/pragalvhasharma/Downloads/thermal_dataset/8_bit_dataset/8_bit_dataset/train_images_8_bit"
    label_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/test/train_labels_8_bit.txt"
    
    try:
        # Create dataset instance
        dataset = ThermalDataset(img_dir, label_file)
        
        # Test loading a random sample
        if len(dataset) > 0:
            # Select a random image
            random_image = random.choice(dataset.images)
            print(f"\nSelected image: {random_image}")
            
            # Print raw label data for this image
            print("\nRaw label data for this image:")
            image_labels = dataset.image_labels[random_image]
            for idx, label in enumerate(image_labels, 1):
                print(f"Object {idx}:")
                print(f"  Class: {label['class']} ({dataset.class_names[label['class']]})")
                print(f"  Box [x_min, x_max, y_min, y_max]: {label['box']}")
            
            # Load image
            img_path = os.path.join(img_dir, random_image)
            print(f"\nLoading image from: {img_path}")
            
            # Load original image
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            print(f"Original image shape: {image.shape}")
            
            # Create figure with original image size
            dpi = 100
            height, width = image.shape[:2]
            plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            
            # Plot original image
            plt.imshow(image, cmap='gray')
            plt.title(f"Original Image: {width}x{height}")
            
            # Draw boxes on original image
            for label in image_labels:
                x_min, y_min, x_max, y_max = label['box']
                class_id = label['class']
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min,
                    y_max - y_min, 
                    linewidth=1, 
                    edgecolor=dataset.class_colors[class_id], 
                    facecolor='none'
                )
                plt.gca().add_patch(rect)
                
                # Add label with coordinates
                label_text = f"{dataset.class_names[class_id]} ({x_min:.1f},{y_min:.1f})-({x_max:.1f},{y_max:.1f})"
                plt.text(
                    x_min, y_min - 5,
                    label_text,
                    color=dataset.class_colors[class_id],
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            
            plt.axis('on')
            plt.grid(False)
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. The image directory exists and contains thermal images")
        print("2. The label file exists and follows the correct format")
        print("3. All paths are correctly specified") 