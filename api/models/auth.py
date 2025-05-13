from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
import uuid
from datetime import timedelta
from django.utils import timezone
import pyotp
import qrcode
import base64
import io
from django.conf import settings

class UserToken(models.Model):
    """Custom token model for user authentication"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='custom_auth_token')
    token = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Token for {self.user.username}"

class EmailVerificationToken(models.Model):
    """Token for email verification"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='email_verification_token')
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    
    def save(self, *args, **kwargs):
        if not self.expires_at:
            # Set expiration to 24 hours from creation
            self.expires_at = timezone.now() + timedelta(hours=24)
        super().save(*args, **kwargs)
    
    def is_valid(self):
        """Check if token is valid (not expired and not used)"""
        return not self.is_used and timezone.now() < self.expires_at
    
    def __str__(self):
        return f"Email verification token for {self.user.username}"

class PasswordResetToken(models.Model):
    """Token for password reset"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='password_reset_tokens')
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    
    def save(self, *args, **kwargs):
        if not self.expires_at:
            # Set expiration to 1 hour from creation
            self.expires_at = timezone.now() + timedelta(hours=1)
        super().save(*args, **kwargs)
    
    def is_valid(self):
        """Check if token is valid (not expired and not used)"""
        return not self.is_used and timezone.now() < self.expires_at
    
    def __str__(self):
        return f"Password reset token for {self.user.username}"

class UserEmailStatus(models.Model):
    """Track user email verification status"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='email_status')
    is_verified = models.BooleanField(default=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        status = "verified" if self.is_verified else "unverified"
        return f"Email {status} for {self.user.username}"

@receiver(post_save, sender=User)
def create_user_email_status(sender, instance, created, **kwargs):
    """Create UserEmailStatus when a new User is created"""
    if created:
        UserEmailStatus.objects.create(user=instance)

class UserTOTP(models.Model):
    """TOTP-based two-factor authentication for users"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='totp')
    secret_key = models.CharField(max_length=32, unique=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"TOTP for {self.user.username}"
    
    def get_totp_uri(self):
        """Generate the TOTP URI for QR code generation"""
        issuer = settings.OTP_TOTP_ISSUER
        return pyotp.totp.TOTP(self.secret_key).provisioning_uri(
            name=self.user.email, 
            issuer_name=issuer
        )
    
    def get_qr_code(self):
        """Generate QR code as base64 image"""
        totp_uri = self.get_totp_uri()
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_token(self, token):
        """Verify a TOTP token"""
        totp = pyotp.TOTP(self.secret_key)
        return totp.verify(token)
