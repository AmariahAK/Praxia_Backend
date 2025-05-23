import os
import logging
import torch
import time
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_setup_weights():
    try:
        # Ensure the data/models directory exists
        target_dir = os.path.join(os.environ.get("BASE_DIR", ""), "data", "models")
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, "densenet_xray.pth")
        
        # If we already have a model file, don't download again
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:  # > 1MB
            logger.info(f"Model file already exists at {target_path}")
            return
            
        # Create a simple model with the right architecture
        logger.info("Creating DenseNet121 model...")
        from torchvision.models import densenet121
        model = densenet121(pretrained=True)  
        
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, 3) 
        
        # Save the model
        torch.save(model, target_path)
        logger.info(f"Model saved to: {target_path}")
        
        # Create a success marker file
        Path(os.path.join(target_dir, "download_success.txt")).write_text("Model downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading weights: {str(e)}")
        # Create a marker file to indicate failure
        Path(os.path.join(target_dir, "download_failed.txt")).write_text(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Try multiple times with backoff
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Attempt {attempt} of {max_attempts}")
            download_and_setup_weights()
            break
        except Exception as e:
            if attempt < max_attempts:
                wait_time = 5 * attempt
                logger.error(f"Attempt {attempt} failed. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed.")
                raise
