from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import XRayAnalysis
from .AI.praxia_model import PraxiaAI
import json
import logging

# Create a logger instance
logger = logging.getLogger(__name__)

# Removed the duplicate UserProfile signals since they're already in models/user.py

@receiver(post_save, sender=XRayAnalysis)
def process_xray_analysis(sender, instance, created, **kwargs):
    """Process X-ray analysis when a new XRayAnalysis is created"""
    if created and instance.image and instance.analysis_result == "Processing...":
        # Process the X-ray image asynchronously
        praxia = PraxiaAI()
        analysis_task = praxia.analyze_xray.delay(instance.image.path)
        
        # Set up a callback to update the analysis when complete
        @analysis_task.on_success
        def update_analysis(result, *args, **kwargs):
            if isinstance(result, dict):
                instance.analysis_result = json.dumps(result)
                if 'detected_conditions' in result:
                    instance.detected_conditions = result['detected_conditions']
                if 'confidence_scores' in result:
                    instance.confidence_scores = result['confidence_scores']
                instance.save()
                logger.info("X-ray analysis updated", xray_id=instance.id)
