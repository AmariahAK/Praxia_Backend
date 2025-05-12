from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.utils import timezone
from django.shortcuts import get_object_or_404
from ..serializers import (
    RegisterSerializer, 
    LoginSerializer, 
    EmailVerificationSerializer,
    PasswordResetRequestSerializer,
    PasswordResetConfirmSerializer
)
from ..models import EmailVerificationToken, PasswordResetToken, UserEmailStatus
from ..utils.email import (
    send_verification_email,
    send_password_reset_email,
    send_password_changed_email
)
import structlog

logger = structlog.get_logger(__name__)

class RegisterView(APIView):
    """View for user registration"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            
            # Create email verification token
            verification_token = EmailVerificationToken.objects.create(user=user)
            
            # Send verification email
            send_verification_email(user, verification_token.token)
            
            return Response({
                'message': 'User registered successfully. Please check your email to verify your account.',
                'user_id': user.pk,
                'email': user.email
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class EmailVerificationView(APIView):
    """View for email verification"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = EmailVerificationSerializer(data=request.data)
        if serializer.is_valid():
            token_uuid = serializer.validated_data['token']
            
            try:
                verification_token = EmailVerificationToken.objects.get(token=token_uuid)
                
                # Check if token is valid
                if not verification_token.is_valid():
                    return Response({
                        'error': 'Verification link has expired or already been used.'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Mark token as used
                verification_token.is_used = True
                verification_token.save()
                
                # Activate user
                user = verification_token.user
                user.is_active = True
                user.save()
                
                # Update email status
                email_status = UserEmailStatus.objects.get(user=user)
                email_status.is_verified = True
                email_status.verified_at = timezone.now()
                email_status.save()
                
                # Create auth token for automatic login
                token, created = Token.objects.get_or_create(user=user)
                
                logger.info("Email verified successfully", user_id=user.id, email=user.email)
                
                return Response({
                    'message': 'Email verified successfully.',
                    'token': token.key,
                    'user_id': user.pk,
                    'email': user.email
                })
                
            except EmailVerificationToken.DoesNotExist:
                return Response({
                    'error': 'Invalid verification token.'
                }, status=status.HTTP_400_BAD_REQUEST)
                
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ResendVerificationEmailView(APIView):
    """View for resending verification email"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({
                'error': 'Email is required.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
            
            # Check if email is already verified
            email_status = UserEmailStatus.objects.get(user=user)
            if email_status.is_verified:
                return Response({
                    'message': 'Email is already verified. Please login.'
                })
            
            # Delete existing token if any
            EmailVerificationToken.objects.filter(user=user).delete()
            
            # Create new verification token
            verification_token = EmailVerificationToken.objects.create(user=user)
            
            # Send verification email
            send_verification_email(user, verification_token.token)
            
            return Response({
                'message': 'Verification email has been sent. Please check your inbox.'
            })
            
        except User.DoesNotExist:
            # Don't reveal that the user doesn't exist for security reasons
            return Response({
                'message': 'If a user with this email exists, a verification email has been sent.'
            })

class LoginView(APIView):
    """View for user login"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
            
            # Check if user is active (email verified)
            if not user.is_active:
                return Response({
                    'error': 'Email not verified. Please check your inbox for verification email.',
                    'email_verified': False
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user = authenticate(username=user.username, password=password)
            if user:
                token, created = Token.objects.get_or_create(user=user)
                return Response({
                    'token': token.key,
                    'user_id': user.pk,
                    'email': user.email,
                    'email_verified': True
                })
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LogoutView(APIView):
    """View for user logout"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            # Delete the user's token to logout
            request.user.auth_token.delete()
            return Response({'message': 'Successfully logged out'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PasswordResetRequestView(APIView):
    """View for requesting password reset"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            
            try:
                user = User.objects.get(email=email)
                
                # Create password reset token
                reset_token = PasswordResetToken.objects.create(user=user)
                
                # Send password reset email
                send_password_reset_email(user, reset_token.token)
                
                return Response({
                    'message': 'Password reset email has been sent. Please check your inbox.'
                })
                
            except User.DoesNotExist:
                # Don't reveal that the user doesn't exist for security reasons
                return Response({
                    'message': 'If a user with this email exists, a password reset email has been sent.'
                })
                
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PasswordResetConfirmView(APIView):
    """View for confirming password reset"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        if serializer.is_valid():
            token_uuid = serializer.validated_data['token']
            password = serializer.validated_data['password']
            
            try:
                reset_token = PasswordResetToken.objects.get(token=token_uuid)
                
                # Check if token is valid
                if not reset_token.is_valid():
                    return Response({
                        'error': 'Password reset link has expired or already been used.'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Mark token as used
                reset_token.is_used = True
                reset_token.save()
                
                # Update user password
                user = reset_token.user
                user.set_password(password)
                user.save()
                
                # Send password changed confirmation email
                send_password_changed_email(user)
                
                # Create auth token for automatic login
                token, created = Token.objects.get_or_create(user=user)
                
                logger.info("Password reset successfully", user_id=user.id, email=user.email)
                
                return Response({
                    'message': 'Password has been reset successfully.',
                    'token': token.key,
                    'user_id': user.pk,
                    'email': user.email
                })
                
            except PasswordResetToken.DoesNotExist:
                return Response({
                    'error': 'Invalid password reset token.'
                }, status=status.HTTP_400_BAD_REQUEST)
                
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CheckEmailVerificationStatusView(APIView):
    """View for checking email verification status"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({
                'error': 'Email is required.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
            email_status = UserEmailStatus.objects.get(user=user)
            
            return Response({
                'is_verified': email_status.is_verified,
                'user_id': user.pk,
                'email': user.email
            })
            
        except User.DoesNotExist:
            return Response({
                'error': 'User with this email does not exist.'
            }, status=status.HTTP_404_NOT_FOUND)
