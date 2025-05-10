from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserProfile, XRayAnalysis
from .AI.praxia_model import PraxiaAI

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create a UserProfile when a new User is created"""
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save the UserProfile when the User is saved"""
    instance.profile.save()

@receiver(post_save, sender=XRayAnalysis)
def process_xray_analysis(sender, instance, created, **kwargs):
    """Process X-ray analysis when a new XRayAnalysis is created"""
    if created and instance.image and not instance.analysis_result:
        # Process the X-ray image asynchronously
        praxia = PraxiaAI()
        praxia.analyze_xray.delay(instance.image.path)
