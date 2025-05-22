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
import traceback

logger = structlog.get_logger(__name__)

class RegisterView(APIView):
    """View for user registration"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            try:
                user = serializer.save()
            
                # Create email verification token
                verification_token = EmailVerificationToken.objects.create(user=user)
            
                # Send verification email
                email_sent = send_verification_email(user, verification_token.token)
            
                response_data = {
                    'message': 'User registered successfully. Please check your email to verify your account.',
                    'user_id': user.pk,
                    'email': user.email
                }
            
                # Add email status to response for debugging
                if not email_sent:
                    logger.warning("Verification email failed to send", user_id=user.pk, email=user.email)
                    response_data['email_status'] = 'failed_to_send'
            
                return Response(response_data, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.error("Registration error", error=str(e), traceback=traceback.format_exc())
                return Response({
                    'error': 'Registration failed due to an internal error. Please try again later.'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
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
            token = serializer.validated_data.get('token', '')
            
            try:
                user = User.objects.get(email=email)
                # Check if user is active (email verified)
                if not user.is_active:
                    return Response({
                        'error': 'Email not verified. Please check your inbox for verification email.',
                        'email_verified': False
                    }, status=status.HTTP_401_UNAUTHORIZED)
            
            except User.DoesNotExist:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
            
            # Authenticate with username and password
            user = authenticate(username=user.username, password=password)
            if not user:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
            
            # Check if 2FA is enabled
            try:
                totp = UserTOTP.objects.get(user=user, is_verified=True)
                
                # If 2FA is enabled, verify token
                if not token:
                    return Response({
                        'error': '2FA is enabled for this account. Please provide a verification code.',
                        'requires_2fa': True
                    }, status=status.HTTP_401_UNAUTHORIZED)
                
                # Verify TOTP token
                if not totp.verify_token(token):
                    return Response({
                        'error': 'Invalid verification code.',
                        'requires_2fa': True
                    }, status=status.HTTP_401_UNAUTHORIZED)
                
            except UserTOTP.DoesNotExist:
                # 2FA not enabled, continue with normal login
                pass
            
            # Create or get token
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user_id': user.pk,
                'email': user.email,
                'email_verified': True,
                'has_2fa': UserTOTP.objects.filter(user=user, is_verified=True).exists()
            })
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LogoutView(APIView):
    """View for user logout"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            # Delete the user's token to logout
            request.user.custom_auth_token.delete()
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

import pyotp
from ..models import UserTOTP
from ..serializers import TOTPSetupSerializer, TOTPVerifySerializer

class TOTPSetupView(APIView):
    """View for setting up 2FA"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get or create TOTP setup for user"""
        # Check if user already has TOTP setup
        totp, created = UserTOTP.objects.get_or_create(
            user=request.user,
            defaults={'secret_key': pyotp.random_base32()}
        )
        
        # If not created and already verified, return error
        if not created and totp.is_verified:
            return Response({
                'error': '2FA is already set up for this account.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # If not created but not verified, regenerate secret key
        if not created and not totp.is_verified:
            totp.secret_key = pyotp.random_base32()
            totp.save()
        
        serializer = TOTPSetupSerializer(totp)
        return Response(serializer.data)

class TOTPVerifyView(APIView):
    """View for verifying 2FA setup"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        """Verify TOTP token and activate 2FA"""
        serializer = TOTPVerifySerializer(data=request.data)
        if serializer.is_valid():
            token = serializer.validated_data['token']
            
            try:
                totp = UserTOTP.objects.get(user=request.user)
                
                # Verify token
                if totp.verify_token(token):
                    totp.is_verified = True
                    totp.save()
                    return Response({
                        'message': '2FA has been successfully set up.',
                        'is_verified': True
                    })
                else:
                    return Response({
                        'error': 'Invalid verification code.'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
            except UserTOTP.DoesNotExist:
                return Response({
                    'error': 'TOTP setup not found. Please set up 2FA first.'
                }, status=status.HTTP_404_NOT_FOUND)
                
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TOTPDisableView(APIView):
    """View for disabling 2FA"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        """Disable 2FA for user"""
        try:
            totp = UserTOTP.objects.get(user=request.user)
            totp.delete()
            return Response({
                'message': '2FA has been successfully disabled.'
            })
        except UserTOTP.DoesNotExist:
            return Response({
                'error': '2FA is not enabled for this account.'
            }, status=status.HTTP_404_NOT_FOUND)
