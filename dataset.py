import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ThermalTransform:
    def __init__(self, normalize_method='adaptive', target_size=(224, 224)):
        """
        Args:
            normalize_method (str): Method to normalize thermal images
                - 'adaptive': Adaptive histogram equalization (default)
                - 'min_max': Simple [0,1] scaling
                - 'percentile': Percentile-based normalization
                - 'thermal': Thermal-specific normalization
            target_size (tuple): Target size for resizing (width, height)
        """
        self.normalize_method = normalize_method
        self.target_size = target_size
        
    def apply_min_max_normalization(self, image):
        """Simple min-max scaling to [0,1] range"""
        return (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    def apply_percentile_normalization(self, image):
        """Percentile-based normalization to handle outliers"""
        lower_percentile = np.percentile(image, 2)
        upper_percentile = np.percentile(image, 98)
        image = np.clip(image, lower_percentile, upper_percentile)
        return (image - lower_percentile) / (upper_percentile - lower_percentile + 1e-8)
    
    def apply_adaptive_normalization(self, image):
        """
        Enhanced Contrast Limited Adaptive Histogram Equalization for thermal images.
        Uses optimal parameters for thermal road scenes and scales to [-1, 1] range.
        """
        # Convert to 8-bit if not already
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE with optimized parameters
        clahe = cv2.createCLAHE(
            clipLimit=2.0,      # Increased slightly for better contrast
            tileGridSize=(8,8)  # Smaller tiles for more local enhancement
        )
        normalized = clahe.apply(image)
        
        # Convert to float and scale to [-1, 1] range
        normalized = normalized.astype(np.float32)
        normalized = (normalized / 127.5) - 1.0  # Scale from [0,255] to [-1,1]
        
        return normalized
    
    def apply_thermal_normalization(self, image):
        """Thermal-specific normalization with gamma correction and outlier removal"""
        # 1. Convert to 0-1 range
        image = image / 255.0
        
        # 2. Apply mild contrast enhancement
        image = np.power(image, 0.9)  # Slight gamma correction
        
        # 3. Remove extreme outliers (keep 99% of data)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # 4. Rescale to [0,1]
        return (image - p1) / (p99 - p1)
    
    def apply_thermal_road_normalization(self, image):
        """
        Specialized normalization for thermal road images:
        1. Bilateral filtering to preserve thermal edges while reducing noise
        2. Local contrast enhancement to highlight objects
        3. Dynamic range adjustment based on scene temperature distribution
        """
        # Convert to float32 and 0-1 range
        image = image.astype(np.float32) / 255.0
        
        # 1. Apply bilateral filter to preserve thermal edges while reducing noise
        # Parameters tuned for thermal road images
        filtered = cv2.bilateralFilter(image, d=5, sigmaColor=0.1, sigmaSpace=5)
        
        # 2. Local contrast enhancement
        # Create a Gaussian-weighted local mean
        kernel_size = (15, 15)
        local_mean = cv2.GaussianBlur(filtered, kernel_size, 0)
        
        # Enhance local contrast while preserving thermal signatures
        enhanced = filtered + 0.5 * (filtered - local_mean)
        
        # 3. Dynamic range adjustment
        # Calculate temperature distribution
        temp_mean = np.mean(enhanced)
        temp_std = np.std(enhanced)
        
        # Adjust range based on scene statistics
        lower_bound = max(0, temp_mean - 2 * temp_std)
        upper_bound = min(1, temp_mean + 2 * temp_std)
        
        # Clip and rescale
        normalized = np.clip(enhanced, lower_bound, upper_bound)
        normalized = (normalized - lower_bound) / (upper_bound - lower_bound)
        
        return normalized
    
    def resize_image(self, image, boxes):
        """Resize image and adjust bounding boxes accordingly"""
        if self.target_size != image.shape[:2]:
            h, w = image.shape[:2]
            scale_x = self.target_size[0] / w
            scale_y = self.target_size[1] / h
            
            # Resize image
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Scale bounding boxes
            if len(boxes) > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                
        return image, boxes
        
    def __call__(self, sample):
        """Apply the selected normalization method to the image"""
        image = sample['image']
        boxes = sample['boxes']
        classes = sample['classes']
        
        # Convert to float32 if not already
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.astype(np.float32)
        
        # Apply selected normalization method
        if self.normalize_method == 'min_max':
            image = self.apply_min_max_normalization(image)
        elif self.normalize_method == 'percentile':
            image = self.apply_percentile_normalization(image)
        elif self.normalize_method == 'adaptive':
            image = self.apply_adaptive_normalization(image)
        elif self.normalize_method == 'thermal':
            image = self.apply_thermal_normalization(image)
        elif self.normalize_method == 'thermal_road':
            image = self.apply_thermal_road_normalization(image)
        
        # Resize image and adjust boxes
        image, boxes = self.resize_image(image, boxes)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Convert to single-channel format (needed for grayscale thermal images)
        if len(image.shape) == 2:
            # Add channel dimension for 2D grayscale images (C,H,W format)
            image = image.unsqueeze(0)
        
        return {
            'image': image,
            'boxes': boxes,
            'classes': classes,
            'image_name': sample.get('image_name', '')
        }

    def visualize_normalization(self, image):
        """
        Visualize the original and adaptively normalized image side by side
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy
            if image.dim() == 3 and image.shape[0] == 3:
                # Convert CHW to HWC for visualization
                image = image.permute(1, 2, 0).numpy()
            else:
                image = image.numpy()
        
        # For 3-channel images, convert to grayscale for visualization
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
            
        image_gray = image_gray.astype(np.float32)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original
        ax1.imshow(image_gray, cmap='gray')
        ax1.set_title('Original')
        
        # Adaptive normalization
        norm_adaptive = self.apply_adaptive_normalization(image_gray.copy())
        # Rescale to [0,1] for visualization only
        norm_adaptive_viz = (norm_adaptive + 1.0) / 2.0
        ax2.imshow(norm_adaptive_viz, cmap='gray')
        ax2.set_title('Adaptive Normalized [-1,1]')
        
        # Add value ranges to titles
        ax1.set_title(f'Original [{image_gray.min():.1f}, {image_gray.max():.1f}]')
        ax2.set_title(f'Adaptive Normalized [{norm_adaptive.min():.1f}, {norm_adaptive.max():.1f}]')
        
        plt.tight_layout()
        plt.show()

class ThermalDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, image_size=(224, 224), include_empty=False):
        """
        Args:
            img_dir (string): Directory with all the images
            label_file (string): Path to the label file
            transform (callable, optional): Optional transform to be applied on a sample
            image_size (tuple): Target size for image resizing (width, height)
            include_empty (bool): Whether to include images without any detections
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_size = image_size
        self.class_names = {1: "Person", 2: "Bicycle", 3: "Vehicle"}
        self.class_colors = {1: 'red', 2: 'blue', 3: 'green'}
        self.include_empty = include_empty
        
        # Get all images in directory
        all_images = set([f for f in os.listdir(img_dir) if f.endswith('.jpeg')])
        print("\nChecking image files:")
        print(f"First image in directory: {min(all_images)}")
        print(f"Last image in directory: {max(all_images)}")
        
        # Group labels by image
        self.image_labels = {}
        invalid_entries = []
        fixed_entries = []
        
        # Read labels file
        labeled_images = set()
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 6:
                    image_name = parts[0]
                    labeled_images.add(image_name)
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
        if not include_empty:
            self.images = list(self.image_labels.keys())  # Only images with labels
        else:
            self.images = list(all_images)  # All images including ones without labels
            # Add empty label lists for images without detections
            for img in all_images:
                if img not in self.image_labels:
                    self.image_labels[img] = []
        
        # Print dataset statistics
        total_objects = sum(len(labels) for labels in self.image_labels.values())
        empty_images = sum(1 for labels in self.image_labels.values() if len(labels) == 0)
        print(f"\nDataset Statistics:")
        print(f"Total images in directory: {len(all_images)}")
        print(f"Images with labels: {len(labeled_images)}")
        print(f"Images without labels: {len(all_images - labeled_images)}")
        print(f"Images included in dataset: {len(self.images)}")
        if include_empty:
            print(f"Empty images included: {empty_images}")
        
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
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # Ensure image is in float32 format
        image = image.astype(np.float32)
        
        # Get labels for this image
        labels = self.image_labels[image_name]
        boxes = torch.tensor([label['box'] for label in labels], dtype=torch.float32)
        classes = torch.tensor([label['class'] for label in labels], dtype=torch.long)
        
        # Create sample dictionary
        sample = {
            'image': image,
            'boxes': boxes,
            'classes': classes,
            'image_name': image_name
        }
        
        # Apply transforms if any
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
        # Images should already be in CHW format with 3 channels from the transform
        images.append(item['image'])
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
    label_file = "train_labels_8_bit.txt"
    
    try:
        # Create transform instance
        transform = ThermalTransform(normalize_method='adaptive', target_size=(224, 224))
        
        print("\n=== Dataset WITHOUT empty images (recommended for training) ===")
        dataset_no_empty = ThermalDataset(img_dir, label_file, transform=transform, include_empty=False)
        
        print("\n=== Dataset WITH empty images ===")
        dataset_with_empty = ThermalDataset(img_dir, label_file, transform=transform, include_empty=True)
        
        # Test loading a random sample from dataset without empty images
        if len(dataset_no_empty) > 0:
            # Select a random image
            random_image = random.choice(dataset_no_empty.images)
            print(f"\nSelected image: {random_image}")
            
            # Load original image for visualization
            img_path = os.path.join(img_dir, random_image)
            original_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if original_image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Get normalized sample
            idx = dataset_no_empty.images.index(random_image)
            normalized_sample = dataset_no_empty[idx]
            
            # Print raw label data
            print("\nRaw label data for this image:")
            image_labels = dataset_no_empty.image_labels[random_image]
            for idx, label in enumerate(image_labels, 1):
                print(f"Object {idx}:")
                print(f"  Class: {label['class']} ({dataset_no_empty.class_names[label['class']]})")
                print(f"  Box: {label['box']}")
            
            # Create figure with original and normalized images side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot original image
            ax1.imshow(original_image, cmap='gray')
            ax1.set_title("Original Image")
            
            # Plot normalized image with boxes
            normalized_image = normalized_sample['image']
            
            # Handle 3-channel images for visualization
            if isinstance(normalized_image, torch.Tensor):
                if normalized_image.dim() == 3 and normalized_image.shape[0] == 3:
                    # Convert from CHW to HWC for visualization
                    normalized_image = normalized_image.permute(1, 2, 0).numpy()
                else:
                    normalized_image = normalized_image.squeeze().numpy()
            
            # For 3-channel images, convert to grayscale for visualization
            if len(normalized_image.shape) == 3 and (normalized_image.shape[2] == 3 or normalized_image.shape[0] == 3):
                if normalized_image.shape[0] == 3:  # CHW format
                    normalized_image = normalized_image.transpose(1, 2, 0)
                normalized_image_gray = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2GRAY)
                ax2.imshow(normalized_image_gray, cmap='gray')
            else:
                ax2.imshow(normalized_image, cmap='gray')
                
            ax2.set_title(f"Normalized Image ({transform.normalize_method})")
            
            # Draw boxes on normalized image
            for label in image_labels:
                x_min, y_min, x_max, y_max = label['box']
                class_id = label['class']
                
                # Scale coordinates if needed
                if transform.target_size != original_image.shape[:2]:
                    h, w = original_image.shape[:2]
                    scale_x = transform.target_size[0] / w
                    scale_y = transform.target_size[1] / h
                    x_min, x_max = x_min * scale_x, x_max * scale_x
                    y_min, y_max = y_min * scale_y, y_max * scale_y
                
                rect = patches.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min,
                    y_max - y_min, 
                    linewidth=2, 
                    edgecolor=dataset_no_empty.class_colors[class_id], 
                    facecolor='none'
                )
                ax2.add_patch(rect)
                
                label_text = f"{dataset_no_empty.class_names[class_id]}"
                ax2.text(
                    x_min, y_min - 5,
                    label_text,
                    color=dataset_no_empty.class_colors[class_id],
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            
            plt.tight_layout()
            plt.show()
            
            # Also show the normalization comparison
            transform.visualize_normalization(original_image)
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. The image directory exists and contains thermal images")
        print("2. The label file exists and follows the correct format")
        print("3. All paths are correctly specified") 