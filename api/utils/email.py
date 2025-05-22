from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
import structlog
import traceback

logger = structlog.get_logger(__name__)

def send_email(subject, template_name, context, recipient_list, from_email=None):
    """
    Send an HTML email using a Django template.
    
    Args:
        subject (str): Email subject
        template_name (str): Name of the template to use (without .html extension)
        context (dict): Context data for the template
        recipient_list (list): List of recipient email addresses
        from_email (str, optional): Sender email address. Defaults to DEFAULT_FROM_EMAIL.
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    if not from_email:
        from_email = settings.DEFAULT_FROM_EMAIL
    
    # Add common context variables
    context.update({
        'site_name': 'Praxia',
        'site_url': settings.FRONTEND_URL,
        'support_email': settings.EMAIL_HOST_USER,
    })
    
    try:
        # Log email configuration for debugging
        logger.info("Email configuration", 
                   host=settings.EMAIL_HOST,
                   port=settings.EMAIL_PORT,
                   use_tls=settings.EMAIL_USE_TLS,
                   from_email=from_email)
        
        # Log template path for debugging
        template_path = f'api/email_templates/{template_name}.html'
        logger.info("Rendering email template", template_path=template_path)
        
        # Render HTML content
        try:
            html_content = render_to_string(template_path, context)
            logger.info("Template rendered successfully", length=len(html_content))
        except Exception as template_error:
            logger.error("Template rendering failed", 
                        error=str(template_error),
                        traceback=traceback.format_exc())
            return False
        
        # Create plain text content by stripping HTML
        text_content = strip_tags(html_content)
        
        # Create email message
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=from_email,
            to=recipient_list
        )
        
        # Attach HTML content
        email.attach_alternative(html_content, "text/html")
        
        # Send email
        email.send()
        logger.info("Email sent successfully", 
                   subject=subject, 
                   template=template_name, 
                   recipients=recipient_list)
        return True
    
    except Exception as e:
        logger.error("Failed to send email", 
                    error=str(e),
                    traceback=traceback.format_exc(),
                    subject=subject, 
                    template=template_name, 
                    recipients=recipient_list)
        return False

def build_url(path):
    """Build a full URL using the FRONTEND_URL from settings"""
    base_url = settings.FRONTEND_URL.rstrip('/')
    path = path.lstrip('/')
    return f"{base_url}/{path}"

def send_verification_email(user, verification_token):
    """Send email verification link to user"""
    try:
        subject = "Verify your Praxia account"
        template_name = "verification_email"
        
        # Build the verification URL
        verification_url = build_url(f"auth/verify-email?token={verification_token}")
        
        context = {
            'user': user,
            'verification_token': verification_token,
            'verification_url': verification_url,  
        }
        
        logger.info("Sending verification email", 
                   user_id=user.id, 
                   email=user.email,
                   token=verification_token,
                   url=verification_url) 
                   
        result = send_email(subject, template_name, context, [user.email])
        logger.info("Verification email result", success=result)
        return result
    except Exception as e:
        logger.error("Exception in send_verification_email", 
                    error=str(e),
                    traceback=traceback.format_exc(),
                    user_email=user.email if user else "None")
        return False

def send_password_reset_email(user, reset_token):
    """Send password reset link to user"""
    subject = "Reset your Praxia password"
    template_name = "password_reset"
    context = {
        'user': user,
        'reset_token': reset_token,
    }
    return send_email(subject, template_name, context, [user.email])

def send_password_changed_email(user):
    """Send password changed confirmation to user"""
    subject = "Your Praxia password has been changed"
    template_name = "password_changed"
    context = {
        'user': user,
    }
    return send_email(subject, template_name, context, [user.email])
