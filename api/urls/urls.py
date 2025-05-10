from django.urls import path, include
from rest_framework.routers import DefaultRouter
from ..views import (
    RegisterView, 
    LoginView, 
    LogoutView,
    UserProfileView,
    ConfirmGenderView,
    ChatSessionViewSet,
    ChatMessageView,
    MedicalConsultationView,
    XRayAnalysisView,
    ResearchQueryView,
    HealthCheckView
)

# Create a router for ViewSets
router = DefaultRouter()
router.register(r'chat-sessions', ChatSessionViewSet, basename='chat-session')

# URL patterns
urlpatterns = [
    # Health check URL
    path('health/', HealthCheckView.as_view(), name='health-check'),
    
    # Authentication URLs
    path('auth/', include([
        path('register/', RegisterView.as_view(), name='register'),
        path('login/', LoginView.as_view(), name='login'),
        path('logout/', LogoutView.as_view(), name='logout'),
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
    
    # Research query URLs
    path('research/', ResearchQueryView.as_view(), name='research'),
]
