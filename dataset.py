import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

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
        self.labels = []
        
        # Read labels file
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:  # image_file class_id x1 y1 x2 y2
                    image_name = parts[0]
                    class_id = int(parts[1])  # 1=person, 2=bicycle, 3=vehicle
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    x2 = float(parts[4])
                    y2 = float(parts[5])
                    
                    self.labels.append({
                        'image': image_name,
                        'box': [x1, x2, y1, y2],  # Store as [x_min, x_max, y_min, y_max]
                        'class': class_id
                    })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image
        img_name = os.path.join(self.img_dir, self.labels[idx]['image'])
        image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        
        # Ensure image is loaded correctly
        if image is None:
            raise ValueError(f"Failed to load image: {img_name}")
        
        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Handle grayscale images (thermal images are typically single channel)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Get bounding box and class
        box = np.array(self.labels[idx]['box'], dtype=np.float32)  # [x_min, x_max, y_min, y_max]
        class_id = self.labels[idx]['class']
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Scale bounding box to new image size
        scale_x = self.image_size[0] / orig_width
        scale_y = self.image_size[1] / orig_height
        
        # Scale coordinates (maintaining x_min, x_max, y_min, y_max order)
        box[0] *= scale_x  # x_min
        box[1] *= scale_x  # x_max
        box[2] *= scale_y  # y_min
        box[3] *= scale_y  # y_max
        
        # Normalize coordinates to [0, 1]
        box[0] /= self.image_size[0]  # x_min
        box[1] /= self.image_size[0]  # x_max
        box[2] /= self.image_size[1]  # y_min
        box[3] /= self.image_size[1]  # y_max
        
        # Ensure box coordinates are within [0, 1] range
        box = np.clip(box, 0, 1)
        
        # Convert image to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        sample = {
            'image': image,
            'box': torch.from_numpy(box),
            'class': torch.tensor(class_id, dtype=torch.long)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = []
    boxes = []
    classes = []
    
    for item in batch:
        images.append(torch.from_numpy(item['image']).permute(2, 0, 1))
        boxes.append(item['box'])
        classes.append(item['class'])
    
    return {
        'images': torch.stack(images),
        'boxes': torch.stack(boxes),
        'classes': torch.stack(classes)
    } 

if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    
    # Test parameters
    img_dir = "/Users/pragalvhasharma/Downloads/thermal_dataset/8_bit_dataset/8_bit_dataset/train_images_8_bit"  # Replace with your image directory
    label_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/test/train_labels_8_bit.txt"   # Replace with your label file
    
    try:
        # Create dataset instance
        dataset = ThermalDataset(img_dir, label_file)
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a single sample
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample contents:")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Box coordinates: {sample['box']}")
            print(f"Class ID: {sample['class']}")
            
            # Visualize the first image with its bounding box
            img = sample['image']
            box = sample['box'].numpy()
            
            # Convert normalized coordinates back to pixel coordinates
            h, w = img.shape[:2]
            x_min = int(box[0] * w)
            x_max = int(box[1] * w)
            y_min = int(box[2] * h)
            y_max = int(box[3] * h)
            
            # Draw bounding box
            img_display = img.copy()
            cv2.rectangle(img_display, 
                         (x_min, y_min), 
                         (x_max, y_max), 
                         (0, 255, 0), 2)
            
            # Display image
            plt.figure(figsize=(8, 8))
            plt.imshow(img_display)
            plt.title(f"Sample Image with Bounding Box (Class {sample['class']})")
            plt.axis('off')
            plt.show()
            
            # Test DataLoader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, 
                                  batch_size=4, 
                                  shuffle=True, 
                                  collate_fn=collate_fn)
            
            # Get a batch
            batch = next(iter(dataloader))
            print("\nBatch contents:")
            print(f"Batch image shape: {batch['images'].shape}")
            print(f"Batch boxes shape: {batch['boxes'].shape}")
            print(f"Batch classes shape: {batch['classes'].shape}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("\nPlease ensure:")
        print("1. The image directory exists and contains thermal images")
        print("2. The label file exists and follows the correct format")
        print("3. All paths are correctly specified") 