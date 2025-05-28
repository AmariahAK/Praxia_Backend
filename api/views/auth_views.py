from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.utils import timezone
from ..serializers import (
    RegisterSerializer, 
    LoginSerializer, 
    EmailVerificationSerializer,
    PasswordResetRequestSerializer,
    PasswordResetConfirmSerializer
)
from ..models import EmailVerificationToken, PasswordResetToken, UserEmailStatus, UserSession
from ..utils.mail_service import (
    send_verification_email,
    send_password_reset_email,
    send_password_changed_email
)
from ..utils.jwt_utils import JWTManager
from ..authentication import SessionJWTAuthentication
import structlog
import traceback

logger = structlog.get_logger(__name__)

def get_client_info(request):
    """Extract client information from request"""
    return {
        'ip_address': request.META.get('REMOTE_ADDR'),
        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        'device_info': request.META.get('HTTP_X_DEVICE_INFO', '')
    }

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
    authentication_classes = []  
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
                
                # Generate JWT tokens
                access_token, refresh_token = JWTManager.generate_tokens(user)
                
                # Create session
                client_info = get_client_info(request)
                session = UserSession.objects.create(
                    user=user,
                    jwt_token=access_token,
                    **client_info
                )
                
                logger.info("Email verified successfully", user_id=user.id, email=user.email)
                
                return Response({
                    'message': 'Email verified successfully.',
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'session_key': session.session_key,
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
    authentication_classes = []  
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        email = request.data.get('email')
        if not email:
            logger.warning("Resend verification attempt without email")
            return Response({
                'error': 'Email is required.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        logger.info(f"Attempting to resend verification email to: {email}")
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
            
            # Generate JWT tokens
            access_token, refresh_token = JWTManager.generate_tokens(user)
            
            # Create session
            client_info = get_client_info(request)
            session = UserSession.objects.create(
                user=user,
                jwt_token=access_token,
                **client_info
            )
            
            # Check if user has 2FA enabled (for info only)
            has_2fa = UserTOTP.objects.filter(user=user, is_verified=True).exists()
            
            return Response({
                'access_token': access_token,
                'refresh_token': refresh_token,
                'session_key': session.session_key,
                'user_id': user.pk,
                'email': user.email,
                'email_verified': True,
                'has_2fa': has_2fa
            })
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LogoutView(APIView):
    """View for user logout"""
    authentication_classes = [SessionJWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            session_key = request.headers.get('X-Session-Key')
            if session_key:
                # Deactivate the specific session
                UserSession.objects.filter(
                    user=request.user,
                    session_key=session_key
                ).update(is_active=False)
            else:
                # Deactivate all user sessions
                UserSession.objects.filter(user=request.user).update(is_active=False)
            
            return Response({'message': 'Successfully logged out'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LogoutAllView(APIView):
    """View for logging out from all devices"""
    authentication_classes = [SessionJWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            # Deactivate all user sessions
            UserSession.objects.filter(user=request.user).update(is_active=False)
            return Response({'message': 'Successfully logged out from all devices'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RefreshTokenView(APIView):
    """View for refreshing access token"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        refresh_token = request.data.get('refresh_token')
        session_key = request.data.get('session_key')
        
        if not refresh_token or not session_key:
            return Response({
                'error': 'Refresh token and session key are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Verify refresh token
        payload = JWTManager.verify_token(refresh_token, 'refresh')
        if not payload:
            return Response({
                'error': 'Invalid or expired refresh token'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            # Verify session exists and is valid
            session = UserSession.objects.get(
                session_key=session_key,
                user_id=payload['user_id'],
                is_active=True
            )
            
            if not session.is_valid():
                return Response({
                    'error': 'Session expired'
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            # Generate new access token
            user = session.user
            new_access_token, _ = JWTManager.generate_tokens(user)
            
            # Update session with new token
            session.jwt_token = new_access_token
            session.refresh()
            
            return Response({
                'access_token': new_access_token,
                'session_key': session.session_key
            })
            
        except UserSession.DoesNotExist:
            return Response({
                'error': 'Invalid session'
            }, status=status.HTTP_401_UNAUTHORIZED)

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
                
                # Invalidate all existing sessions for security
                UserSession.objects.filter(user=user).update(is_active=False)
                
                # Send password changed confirmation email
                send_password_changed_email(user)
                
                # Generate new JWT tokens and session
                access_token, refresh_token = JWTManager.generate_tokens(user)
                
                # Create new session
                client_info = get_client_info(request)
                session = UserSession.objects.create(
                    user=user,
                    jwt_token=access_token,
                    **client_info
                )
                
                logger.info("Password reset successfully", user_id=user.id, email=user.email)
                
                return Response({
                    'message': 'Password has been reset successfully.',
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'session_key': session.session_key,
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

# 2FA Views
import pyotp
from ..models import UserTOTP
from ..serializers import TOTPSetupSerializer, TOTPVerifySerializer

class TOTPSetupView(APIView):
    """View for setting up 2FA"""
    authentication_classes = [SessionJWTAuthentication]
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
    authentication_classes = [SessionJWTAuthentication]
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
    authentication_classes = [SessionJWTAuthentication]
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

class TOTPStatusView(APIView):
    """View for checking 2FA status"""
    authentication_classes = [SessionJWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Check if user has 2FA enabled"""
        try:
            totp = UserTOTP.objects.get(user=request.user, is_verified=True)
            return Response({
                'has_2fa': True,
                'created_at': totp.created_at
            })
        except UserTOTP.DoesNotExist:
            return Response({
                'has_2fa': False
            })

class UserSessionsView(APIView):
    """View for managing user sessions"""
    authentication_classes = [SessionJWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get all active sessions for the user"""
        sessions = UserSession.objects.filter(
            user=request.user,
            is_active=True
        ).order_by('-last_activity')
        
        session_data = []
        current_session_key = request.headers.get('X-Session-Key')
        
        for session in sessions:
            session_data.append({
                'session_key': session.session_key,
                'device_info': session.device_info,
                'ip_address': session.ip_address,
                'user_agent': session.user_agent,
                'created_at': session.created_at,
                'last_activity': session.last_activity,
                'is_current': session.session_key == current_session_key
            })
        
        return Response({'sessions': session_data})
    
    def delete(self, request):
        """Terminate a specific session"""
        session_key = request.data.get('session_key')
        if not session_key:
            return Response({
                'error': 'Session key is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            session = UserSession.objects.get(
                user=request.user,
                session_key=session_key,
                is_active=True
            )
            session.is_active = False
            session.save()
            
            return Response({
                'message': 'Session terminated successfully'
            })
        except UserSession.DoesNotExist:
            return Response({
                'error': 'Session not found'
            }, status=status.HTTP_404_NOT_FOUND)
