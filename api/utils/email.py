from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
import structlog

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
        'site_url': 'https://praxia.example.com',
        'support_email': settings.EMAIL_HOST_USER,
    })
    
    try:
        # Render HTML content
        html_content = render_to_string(f'api/email_templates/{template_name}.html', context)
        
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
                    subject=subject, 
                    template=template_name, 
                    recipients=recipient_list)
        return False

def send_verification_email(user, verification_token):
    """Send email verification link to user"""
    subject = "Verify your Praxia account"
    template_name = "verification_email"
    context = {
        'user': user,
        'verification_token': verification_token,
    }
    return send_email(subject, template_name, context, [user.email])

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
