from django.contrib import admin
from .models import (
    UserProfile, 
    ChatSession, 
    ChatMessage, 
    MedicalConsultation, 
    XRayAnalysis, 
    ResearchQuery
)

# Register models
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'age', 'country', 'created_at', 'updated_at')
    search_fields = ('user__username', 'user__email', 'country')
    list_filter = ('created_at', 'updated_at')

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'created_at', 'updated_at')
    search_fields = ('title', 'user__username')
    list_filter = ('created_at', 'updated_at')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'role', 'created_at')
    search_fields = ('content', 'session__title')
    list_filter = ('role', 'created_at')

@admin.register(MedicalConsultation)
class MedicalConsultationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'created_at')
    search_fields = ('symptoms', 'diagnosis', 'user__username')
    list_filter = ('created_at',)
    readonly_fields = ('diagnosis',)

@admin.register(XRayAnalysis)
class XRayAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'created_at')
    search_fields = ('analysis_result', 'user__username')
    list_filter = ('created_at',)

@admin.register(ResearchQuery)
class ResearchQueryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'query', 'created_at')
    search_fields = ('query', 'user__username')
    list_filter = ('created_at',)
