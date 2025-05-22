from django.core.management.base import BaseCommand
from api.utils.email import send_verification_email, send_email
from django.contrib.auth.models import User
from django.conf import settings
import uuid
import structlog

logger = structlog.get_logger(__name__)

class Command(BaseCommand):
    help = 'Test email sending functionality'

    def add_arguments(self, parser):
        parser.add_argument('--email', type=str, help='Email address to send test to')
        parser.add_argument('--method', type=str, default='all', 
                           help='Test method: all, direct, template, verification')

    def handle(self, *args, **options):
        test_email = options.get('email')
        method = options.get('method', 'all')
        
        self.stdout.write(self.style.SUCCESS(f'Testing email functionality with method: {method}'))
        
        # Log email settings
        self.stdout.write(f"Email settings:")
        self.stdout.write(f"  EMAIL_BACKEND: {settings.EMAIL_BACKEND}")
        self.stdout.write(f"  EMAIL_HOST: {settings.EMAIL_HOST}")
        self.stdout.write(f"  EMAIL_PORT: {settings.EMAIL_PORT}")
        self.stdout.write(f"  EMAIL_USE_TLS: {settings.EMAIL_USE_TLS}")
        self.stdout.write(f"  EMAIL_HOST_USER: {settings.EMAIL_HOST_USER}")
        self.stdout.write(f"  DEFAULT_FROM_EMAIL: {settings.DEFAULT_FROM_EMAIL}")
        
        if method in ['all', 'direct']:
            self._test_direct_email(test_email)
            
        if method in ['all', 'template']:
            self._test_template_email(test_email)
            
        if method in ['all', 'verification']:
            self._test_verification_email()

    def _test_direct_email(self, test_email):
        """Test sending a direct email without templates"""
        self.stdout.write(self.style.SUCCESS('Testing direct email sending...'))
        
        try:
            from django.core.mail import send_mail
            
            recipient = test_email or settings.EMAIL_HOST_USER
            
            result = send_mail(
                'Praxia Test Email (Direct)',
                'This is a test email from Praxia sent directly without templates.',
                settings.DEFAULT_FROM_EMAIL,
                [recipient],
                fail_silently=False,
            )
            
            if result:
                self.stdout.write(self.style.SUCCESS(f'Direct email sent successfully to {recipient}'))
            else:
                self.stdout.write(self.style.ERROR(f'Failed to send direct email to {recipient}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error sending direct email: {str(e)}'))

    def _test_template_email(self, test_email):
        """Test sending an email with template"""
        self.stdout.write(self.style.SUCCESS('Testing template email sending...'))
        
        try:
            recipient = test_email or settings.EMAIL_HOST_USER
            test_token = uuid.uuid4()
            
            # Build the verification URL
            verification_url = f"{settings.FRONTEND_URL}/auth/verify_email?token={test_token}"
            
            context = {
                'user': {'first_name': 'Test User', 'email': recipient},
                'verification_token': test_token,
                'site_url': settings.FRONTEND_URL,
                'verification_url': verification_url,  
            }
            
            result = send_email(
                'Praxia Test Email (Template)',
                'verification_email',
                context,
                [recipient]
            )
            
            if result:
                self.stdout.write(self.style.SUCCESS(f'Template email sent successfully to {recipient}'))
                self.stdout.write(self.style.SUCCESS(f'Using frontend URL: {settings.FRONTEND_URL}'))
                self.stdout.write(self.style.SUCCESS(f'Verification URL: {verification_url}'))
            else:
                self.stdout.write(self.style.ERROR(f'Failed to send template email to {recipient}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error sending template email: {str(e)}'))

    def _test_verification_email(self):
        """Test sending a verification email to an existing user"""
        self.stdout.write(self.style.SUCCESS('Testing verification email to existing user...'))
        
        try:
            # Get a user
            user = User.objects.first()
            if not user:
                self.stdout.write(self.style.ERROR('No users found in the database'))
                return
                
            # Generate a test token
            test_token = uuid.uuid4()
            
            # Try to send email
            result = send_verification_email(user, test_token)
            
            if result:
                self.stdout.write(self.style.SUCCESS(f'Verification email sent successfully to {user.email}'))
            else:
                self.stdout.write(self.style.ERROR(f'Failed to send verification email to {user.email}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error sending verification email: {str(e)}'))

    def _test_password_reset_email(self, test_email):
        """Test sending a password reset email"""
        self.stdout.write(self.style.SUCCESS('Testing password reset email sending...'))
        
        try:
            recipient = test_email or settings.EMAIL_HOST_USER
            
            # Create a test reset token
            test_token = uuid.uuid4()
            
            # Build the reset URL
            reset_url = f"{settings.FRONTEND_URL}/auth/reset_password?token={test_token}"
            
            context = {
                'user': {'first_name': 'Test User', 'email': recipient},
                'reset_token': test_token,
                'site_url': settings.FRONTEND_URL,
                'reset_url': reset_url,
            }
            
            result = send_email(
                'Praxia Test Password Reset Email',
                'password_reset',
                context,
                [recipient]
            )
            
            if result:
                self.stdout.write(self.style.SUCCESS(f'Password reset email sent successfully to {recipient}'))
                self.stdout.write(self.style.SUCCESS(f'Using frontend URL: {settings.FRONTEND_URL}'))
                self.stdout.write(self.style.SUCCESS(f'Reset URL: {reset_url}'))
            else:
                self.stdout.write(self.style.ERROR(f'Failed to send password reset email to {recipient}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error sending password reset email: {str(e)}'))
