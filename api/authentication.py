from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth.models import User
from .models.auth import UserSession
from .utils.jwt_utils import JWTManager
from django.utils import timezone

class SessionJWTAuthentication(BaseAuthentication):
    """
    Custom authentication that validates both session and JWT token
    """
    
    def authenticate(self, request):
        session_key = request.headers.get('X-Session-Key')
        auth_header = request.headers.get('Authorization')
        
        if not session_key or not auth_header:
            return None
            
        if not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header.split(' ')[1]
        
        # Verify JWT token
        payload = JWTManager.verify_token(token, 'access')
        if not payload:
            raise AuthenticationFailed('Invalid or expired token')
        
        # Verify session
        try:
            session = UserSession.objects.get(
                session_key=session_key,
                user_id=payload['user_id']
            )
            
            if not session.is_valid():
                raise AuthenticationFailed('Session expired')
                
            # Refresh session activity
            session.refresh()
            
            user = session.user
            return (user, {'session': session, 'token_payload': payload})
            
        except UserSession.DoesNotExist:
            raise AuthenticationFailed('Invalid session')
    
    def authenticate_header(self, request):
        return 'Bearer'
