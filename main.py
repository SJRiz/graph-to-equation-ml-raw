import os

from data.dataset import EquationDataset
from training.train import load_or_train_model
from training.evaluate import evaluate_model, predict_from_image
from config.config import VALIDATION_SAMPLE_SIZE, TRAINING_IMAGE_SIZE

def main():
    print("=== Graph-to-Equation Model ===")
    
    # Load or train model
    model = load_or_train_model()
    
    # Interactive mode
    while True:
        print(f"\n" + "="*50)
        print("Choose an option:")
        print("1. Test model on validation dataset")
        print("2. Predict from your own image file")
        print("3. Train new model (will overwrite existing)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':

            # Create small validation dataset for testing
            print("Creating test dataset...")
            test_dataset = EquationDataset(num_samples=VALIDATION_SAMPLE_SIZE, image_size=TRAINING_IMAGE_SIZE, split='val')
            evaluate_model(model, test_dataset)
            
        elif choice == '2':
            image_path = input("Enter path to your image file: ").strip().strip('"')
            result = predict_from_image(model, image_path)
            
            if result:
                print(f"\n🎯 Prediction Result:")
                print(f"Function Type: {result['type']}")
                print(f"Equation: {result['equation']}")
                print(f"Confidence: {result['confidence']:.4f}")
                
        elif choice == '3':
            confirm = input("This will retrain the model from scratch. Continue? (y/n): ").strip().lower()
            if confirm == 'y':

                # Remove existing model
                if os.path.exists('output/best_model.pth'):
                    os.remove('output/best_model.pth')
                model = load_or_train_model()
            
        elif choice == '4':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
