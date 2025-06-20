from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from ..models import UserToken, EmailVerificationToken, PasswordResetToken, UserEmailStatus

class RegisterSerializer(serializers.ModelSerializer):
    """Serializer for user registration"""
    full_name = serializers.CharField(required=True)
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    
    class Meta:
        model = User
        fields = ('id', 'full_name', 'email', 'password', 'password2')
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        
        if User.objects.filter(email=attrs['email']).exists():
            raise serializers.ValidationError({"email": "A user with this email already exists."})
        
        return attrs
    
    def create(self, validated_data):
        # Split full name into first_name and last_name
        full_name = validated_data.pop('full_name')
        name_parts = full_name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        # Remove password2 from validated data
        validated_data.pop('password2')
        
        # Create username from email
        email = validated_data.get('email')
        username = email.split('@')[0]
        
        # Create user
        user = User.objects.create(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            is_active=False  # User starts as inactive until email is verified
        )
        
        user.set_password(validated_data['password'])
        user.save()
        
        return user

class LoginSerializer(serializers.Serializer):
    """Serializer for user login"""
    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True)

class EmailVerificationSerializer(serializers.Serializer):
    """Serializer for email verification"""
    token = serializers.UUIDField(required=True)

class PasswordResetRequestSerializer(serializers.Serializer):
    """Serializer for password reset request"""
    email = serializers.EmailField(required=True)
    
    def validate_email(self, value):
        if not User.objects.filter(email=value).exists():
            raise serializers.ValidationError("No user is registered with this email address.")
        return value

class PasswordResetConfirmSerializer(serializers.Serializer):
    """Serializer for password reset confirmation"""
    token = serializers.UUIDField(required=True)
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

class RefreshTokenSerializer(serializers.Serializer):
    """Serializer for token refresh"""
    refresh_token = serializers.CharField(required=True)
    session_key = serializers.CharField(required=True)

# Add to existing auth_serializer.py file
import pyotp
from ..models import UserTOTP

class TOTPSetupSerializer(serializers.ModelSerializer):
    """Serializer for setting up TOTP-based 2FA"""
    qr_code = serializers.SerializerMethodField()
    
    class Meta:
        model = UserTOTP
        fields = ('qr_code', 'created_at')
        read_only_fields = ('qr_code', 'created_at')
    
    def get_qr_code(self, obj):
        return obj.get_qr_code()

class TOTPVerifySerializer(serializers.Serializer):
    """Serializer for verifying TOTP token"""
    token = serializers.CharField(max_length=6, min_length=6)

class TOTPLoginSerializer(serializers.Serializer):
    """Serializer for 2FA login"""
    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True)
    token = serializers.CharField(max_length=6, min_length=6, required=False)

class UserSessionSerializer(serializers.Serializer):
    """Serializer for user session data"""
    session_key = serializers.CharField()
    device_info = serializers.CharField()
    ip_address = serializers.IPAddressField()
    user_agent = serializers.CharField()
    created_at = serializers.DateTimeField()
    last_activity = serializers.DateTimeField()
    is_current = serializers.BooleanField()
