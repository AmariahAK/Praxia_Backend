from django.urls import path, include
from rest_framework.routers import DefaultRouter
from ..views import (
    RegisterView, 
    LoginView, 
    LogoutView,
    EmailVerificationView,
    ResendVerificationEmailView,
    PasswordResetRequestView,
    PasswordResetConfirmView,
    CheckEmailVerificationStatusView,
    UserProfileView,
    ConfirmGenderView,
    ChatSessionViewSet,
    ChatMessageView,
    MedicalConsultationView,
    XRayAnalysisView,
    ResearchQueryView,
    HealthCheckView,
    HealthNewsView,
    TOTPDisableView,
    TOTPSetupView,
    TOTPVerifyView,
    AuthenticatedHealthCheckView,
    XRayAnalysisDetailView
)

# Create a router for ViewSets
router = DefaultRouter()
router.register(r'chat-sessions', ChatSessionViewSet, basename='chat-session')

# URL patterns
urlpatterns = [
    # Health check URL
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('health/authenticated/', AuthenticatedHealthCheckView.as_view(), name='authenticated-health-check'),
    
    # Authentication URLs
    path('auth/', include([
        path('register/', RegisterView.as_view(), name='register'),
        path('login/', LoginView.as_view(), name='login'),
        path('logout/', LogoutView.as_view(), name='logout'),
        path('verify_email/', EmailVerificationView.as_view(), name='verify-email'),
        path('resend-verification-email/', ResendVerificationEmailView.as_view(), name='resend-verification-email'),
        path('password-reset-request/', PasswordResetRequestView.as_view(), name='password-reset-request'),
        path('password-reset-confirm/', PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
        path('check-email-verification/', CheckEmailVerificationStatusView.as_view(), name='check-email-verification'),
        path('2fa/setup/', TOTPSetupView.as_view(), name='2fa-setup'),
        path('2fa/verify/', TOTPVerifyView.as_view(), name='2fa-verify'),
        path('2fa/disable/', TOTPDisableView.as_view(), name='2fa-disable'),
    ])),
    
    # User profile URLs
    path('profile/', UserProfileView.as_view(), name='user-profile'),
    path('profile/confirm-gender/', ConfirmGenderView.as_view(), name='confirm-gender'),
    
    # Chat URLs
    path('', include(router.urls)),
    path('chat-sessions/<int:session_id>/messages/', ChatMessageView.as_view(), name='chat-messages'),
    
    # Medical consultation URLs
    path('consultations/', MedicalConsultationView.as_view(), name='consultations'),
    
    # X-ray analysis URLs
    path('xray-analyses/', XRayAnalysisView.as_view(), name='xray-analyses'),
    path('xray-analyses/<int:pk>/', XRayAnalysisDetailView.as_view(), name='xray-analysis-detail'),
    
    # Research query URLs
    path('research/', ResearchQueryView.as_view(), name='research'),
    path('health-news/', HealthNewsView.as_view(), name='health-news'),
]
