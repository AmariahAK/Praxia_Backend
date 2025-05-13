from rest_framework import serializers
from django.contrib.auth.models import User
from ..models import UserProfile

class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for user profile"""
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ('username', 'email', 'profile_picture', 'age', 'gender', 'gender_locked', 
                  'weight', 'height', 'country', 'allergies', 'preferred_language', 'created_at', 'updated_at')
        read_only_fields = ('created_at', 'updated_at', 'gender_locked')

class UserProfileUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile"""
    
    class Meta:
        model = UserProfile
        fields = ('profile_picture', 'age', 'gender', 'weight', 'height', 'country', 'allergies', 'preferred_language')
    
    def validate_gender(self, value):
        """Validate that gender cannot be changed once set and locked"""
        instance = getattr(self, 'instance', None)
        if instance and instance.gender and instance.gender_locked and instance.gender != value:
            raise serializers.ValidationError("Gender cannot be changed once it has been set and confirmed.")
        return value
    
    def update(self, instance, validated_data):
        """Custom update to handle gender locking"""
        # If gender is being set for the first time, don't lock it yet
        if 'gender' in validated_data and instance.gender != validated_data['gender'] and not instance.gender_locked:
            # Gender will be updated but not locked until confirmation
            pass
        
        return super().update(instance, validated_data)
