import os
from main import ThermalObjectDetector
from dataset import ThermalDataset
import torch
import json
from datetime import datetime

def main():
    # Dataset paths
    train_img_dir = "/Users/pragalvhasharma/Downloads/thermal_dataset/8_bit_dataset/8_bit_dataset/train_images_8_bit"
    train_label_file = "/Users/pragalvhasharma/Downloads/thermal_dataset/8_bit_dataset/8_bit_dataset/train_labels_8_bit.txt"
    val_img_dir = "/Users/pragalvhasharma/Downloads/thermal_dataset/8_bit_dataset/8_bit_dataset/val_images_8_bit"
    val_label_file = "/Users/pragalvhasharma/Downloads/thermal_dataset/8_bit_dataset/8_bit_dataset/val_labels_8_bit.txt"
    
    # Create training and validation datasets
    train_dataset = ThermalDataset(train_img_dir, train_label_file)
    val_dataset = ThermalDataset(val_img_dir, val_label_file)
    
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    
    # Initialize detector
    detector = ThermalObjectDetector()
    
    # Training parameters
    num_epochs = 100  # Increased epochs for better accuracy
    batch_size = 16
    learning_rate = 0.0001
    target_accuracy = 0.98  # 98% accuracy target
    
    print("\nStarting training with target accuracy: 98%")
    print("Training will continue until target accuracy is reached or early stopping is triggered")
    
    # Train the model
    final_metrics = detector.train(
        train_dataset, 
        val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        target_accuracy=target_accuracy
    )
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the trained model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'results/thermal_detector_{timestamp}.pth'
    detector.save_model(model_path)
    
    # Save metrics
    metrics_path = f'results/metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nFinal Metrics:")
    print(f"Overall Accuracy: {final_metrics['overall']['precision']:.4f}")
    print(f"Overall F1 Score: {final_metrics['overall']['f1']:.4f}")
    
    # Print per-class metrics
    for cls_name, metrics in final_metrics.items():
        if cls_name != 'overall':
            print(f"\n{cls_name}:")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
    
    if final_metrics['overall']['precision'] >= target_accuracy:
        print(f"\nSuccess! Reached target accuracy of {target_accuracy:.2%}")
    else:
        print(f"\nNote: Target accuracy of {target_accuracy:.2%} not reached.")
        print("Consider:")
        print("1. Training for more epochs")
        print("2. Adjusting the learning rate")
        print("3. Increasing the model capacity")
        print("4. Adding more training data")

if __name__ == "__main__":
    main() 