from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    """Extended user profile with health information"""
    GENDER_CHOICES = (
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
        ('prefer_not_to_say', 'Prefer not to say'),
    )
    
    LANGUAGE_CHOICES = (
        ('en', 'English'),
        ('es', 'Spanish'),
        ('fr', 'French'),
    )
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES, null=True, blank=True)
    gender_locked = models.BooleanField(default=False, help_text="Once set, gender cannot be changed")
    weight = models.FloatField(null=True, blank=True, help_text="Weight in kilograms")
    height = models.FloatField(null=True, blank=True, help_text="Height in centimeters")
    country = models.CharField(max_length=100, blank=True)
    allergies = models.TextField(blank=True, help_text="List any allergies separated by commas")
    preferred_language = models.CharField(max_length=2, choices=LANGUAGE_CHOICES, default='en')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    """Create or update a UserProfile when a User is created or updated"""
    if created:
        # Check if profile already exists to prevent duplicate creation
        if not hasattr(instance, 'profile'):
            UserProfile.objects.create(user=instance)
    else:
        # Only save the profile if it exists
        try:
            instance.profile.save()
        except UserProfile.DoesNotExist:
            # Create profile if it doesn't exist
            UserProfile.objects.create(user=instance)
