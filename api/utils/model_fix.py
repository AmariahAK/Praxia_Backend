import os
import torch
import logging
from torchvision.models import densenet121
import torchvision.models.densenet
from torch.serialization import add_safe_globals
import torch.nn.modules.container

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
        # Add required modules to safe globals
        add_safe_globals([
            torchvision.models.densenet.DenseNet,
            torch.nn.modules.container.Sequential
        ])
        
        # Create a new model with the correct architecture
        logger.info("Creating new DenseNet model with correct architecture...")
        new_model = densenet121(weights=None)  
        
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
                # Fall back to weights_only=False since we need the Sequential modules
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                logger.info("Successfully loaded model with weights_only=False")
                
                if not isinstance(state_dict, dict):
                    state_dict = state_dict.state_dict()
                
                if "op_threshs" in state_dict:
                    del state_dict["op_threshs"]
                
                if "features.conv0.weight" in state_dict:
                    if state_dict["features.conv0.weight"].size(1) == 1:
                        conv_weights = state_dict["features.conv0.weight"]
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
