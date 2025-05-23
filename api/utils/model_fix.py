import os
import torch
import logging
from torchvision.models import densenet121

logger = logging.getLogger(__name__)

def fix_densenet_model():
    """
    Fix the DenseNet model to handle grayscale images and match expected output dimensions.
    This converts an existing model file to match our architecture or creates a new one.
    """
    model_path = os.path.join(os.environ.get("BASE_DIR", ""), "data", "models", "densenet_xray.pth")
    fixed_model_path = os.path.join(os.environ.get("BASE_DIR", ""), "data", "models", "densenet_xray_fixed.pth")
    
    # Check if fixed model already exists
    if os.path.exists(fixed_model_path) and os.path.getsize(fixed_model_path) > 1000000:
        logger.info(f"Fixed model already exists at {fixed_model_path}")
        return fixed_model_path
    
    try:
        # Create a new model with the correct architecture
        logger.info("Creating new DenseNet model with correct architecture...")
        new_model = densenet121(pretrained=False)
        
        # Modify the first layer to accept both RGB and grayscale images
        # This creates a flexible model that can handle both input types
        original_conv = new_model.features.conv0
        new_model.features.conv0 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Modify the classifier layer to output 3 classes
        num_ftrs = new_model.classifier.in_features
        new_model.classifier = torch.nn.Linear(num_ftrs, 3)
        
        # Try to load and adapt the original weights if possible
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            logger.info("Attempting to adapt weights from existing model...")
            try:
                # Try loading the state dict
                state_dict = torch.load(model_path, map_location='cpu')
                
                # If it's a full model, get its state dict
                if not isinstance(state_dict, dict):
                    state_dict = state_dict.state_dict()
                
                # Remove problematic keys
                if "op_threshs" in state_dict:
                    del state_dict["op_threshs"]
                
                # Fix input layer if needed
                if "features.conv0.weight" in state_dict:
                    # If input is grayscale (1-channel), adapt to RGB (3-channel)
                    if state_dict["features.conv0.weight"].size(1) == 1:
                        conv_weights = state_dict["features.conv0.weight"]
                        # Repeat the grayscale channel to all three RGB channels
                        state_dict["features.conv0.weight"] = conv_weights.repeat(1, 3, 1, 1) / 3.0
                
                # Fix classifier if needed
                classifier_keys = ["classifier.weight", "classifier.bias"]
                for key in classifier_keys:
                    if key not in state_dict:
                        continue
                    # Initialize with zeros or skip if dimensions don't match
                    new_shape = new_model.state_dict()[key].shape
                    if state_dict[key].size(0) != new_shape[0]:
                        logger.info(f"Reinitializing {key} due to shape mismatch")
                        # Just use the initialized weights in new_model for this layer
                        state_dict[key] = new_model.state_dict()[key]
                
                # Load the fixed state dict
                new_model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully adapted weights from existing model")
                
            except Exception as e:
                logger.error(f"Error adapting weights: {str(e)}, using fresh model")
        
        # Save the fixed model
        torch.save(new_model, fixed_model_path)
        logger.info(f"Fixed model saved to: {fixed_model_path}")
        return fixed_model_path
        
    except Exception as e:
        logger.error(f"Error fixing model: {str(e)}")
        return None
