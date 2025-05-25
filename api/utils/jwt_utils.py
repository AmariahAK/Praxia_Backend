import jwt
from datetime import datetime, timedelta
from django.conf import settings
from django.contrib.auth.models import User
import secrets

class JWTManager:
    """Utility class for JWT token management"""
    
    @staticmethod
    def generate_tokens(user):
        """Generate access and refresh tokens for a user"""
        now = datetime.utcnow()
        
        # Access token payload (short-lived: 15 minutes)
        access_payload = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'exp': now + timedelta(minutes=15),
            'iat': now,
            'type': 'access'
        }
        
        # Refresh token payload (long-lived: 7 days)
        refresh_payload = {
            'user_id': user.id,
            'exp': now + timedelta(days=7),
            'iat': now,
            'type': 'refresh',
            'jti': secrets.token_urlsafe(16)  # JWT ID for tracking
        }
        
        access_token = jwt.encode(access_payload, settings.SECRET_KEY, algorithm='HS256')
        refresh_token = jwt.encode(refresh_payload, settings.SECRET_KEY, algorithm='HS256')
        
        return access_token, refresh_token
    
    @staticmethod
    def verify_token(token, token_type='access'):
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
            
            if payload.get('type') != token_type:
                return None
                
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def refresh_access_token(refresh_token):
        """Generate new access token from refresh token"""
        payload = JWTManager.verify_token(refresh_token, 'refresh')
        if not payload:
            return None
            
        try:
            user = User.objects.get(id=payload['user_id'])
            access_token, _ = JWTManager.generate_tokens(user)
            return access_token
        except User.DoesNotExist:
            return None
