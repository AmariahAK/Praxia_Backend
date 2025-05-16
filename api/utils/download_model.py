import torchxrayvision as xrv
import os
import logging
import torch
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_setup_weights():
    try:
        # Load the pre-trained DenseNet model (triggers download)
        logger.info("Downloading DenseNet121 weights...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Ensure the data/models directory exists
        target_dir = os.path.join(os.environ.get("BASE_DIR", ""), "data", "models")
        os.makedirs(target_dir, exist_ok=True)
        
        # Get the model's state dict and save it directly
        target_path = os.path.join(target_dir, "densenet_xray.pth")
        torch.save(model.state_dict(), target_path)
        logger.info(f"Weights saved to: {target_path}")
        
    except Exception as e:
        logger.error(f"Error downloading weights: {str(e)}")
        raise

if __name__ == "__main__":
    download_and_setup_weights()
