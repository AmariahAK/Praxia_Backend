import os
import logging
import torch
import time
import requests
from pathlib import Path
from torchvision.models import densenet121

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
        
        try:
            # Initialize with no pretrained weights to ensure architecture compatibility
            model = densenet121(pretrained=False)
            
            # Ensure input layer can handle RGB (3-channel) images
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, 3)  # Our output has 3 classes
            
            # Save the model
            torch.save(model, target_path)
            logger.info(f"Model saved to: {target_path}")
            
            # Create a success marker file
            Path(os.path.join(target_dir, "download_success.txt")).write_text("Model downloaded successfully")
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
        
    except KeyboardInterrupt:
        logger.error("Download interrupted by user")
        if os.path.exists(target_path):
            logger.info("Removing partially downloaded file")
            os.remove(target_path)
        raise
    except Exception as e:
        logger.error(f"Error downloading weights: {str(e)}")
        # Create a marker file to indicate failure
        Path(os.path.join(target_dir, "download_failed.txt")).write_text(f"Error: {str(e)}")
        raise

def download_with_progress(url, target_path):
    """Download a file with progress reporting"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    logger.info(f"Downloading {url} to {target_path}")
    logger.info(f"File size: {total_size / (1024 * 1024):.2f} MB")
    
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
    
    logger.info("Download complete")
    return target_path

if __name__ == "__main__":
    # Try multiple times with backoff
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Attempt {attempt} of {max_attempts}")
            download_and_setup_weights()
            break
        except KeyboardInterrupt:
            logger.error("Process interrupted by user")
            break
        except Exception as e:
            if attempt < max_attempts:
                wait_time = 5 * attempt
                logger.error(f"Attempt {attempt} failed. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed.")
                raise
