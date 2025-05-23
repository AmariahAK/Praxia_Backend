from django.db import models
from django.contrib.auth.models import User

class ChatSession(models.Model):
    """Record of a chat session between a user and Praxia"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions', db_index=True)
    title = models.CharField(max_length=255, default="New Chat", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)
    
    def __str__(self):
        return f"Chat with {self.user.username} - {self.title}"

class ChatMessage(models.Model):
    """Individual message in a chat session"""
    ROLE_CHOICES = (
        ('user', 'User'),
        ('assistant', 'Assistant'),
    )
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages', db_index=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, db_index=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['role', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.role} message in {self.session}"

class MedicalConsultation(models.Model):
    """Record of a user's medical consultation with Praxia"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='consultations', db_index=True)
    symptoms = models.TextField()
    diagnosis = models.TextField()
    language = models.CharField(max_length=20, default='en', choices=(
        ('en', 'English'),
        ('fr', 'French'),
        ('es', 'Spanish'),
    ))
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"Consultation for {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"

def user_xray_path(instance, filename):
    return f'user_{instance.user.id}/xrays/{filename}'

class XRayAnalysis(models.Model):
    """Record of an X-ray analysis"""
    CONDITION_CHOICES = (
        ('pneumonia', 'Pneumonia'),
        ('fracture', 'Fracture'),
        ('tumor', 'Tumor'),
        ('normal', 'Normal'),
        ('other', 'Other'),
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='xray_analyses', db_index=True)
    image = models.ImageField(upload_to=user_xray_path)
    analysis_result = models.TextField()
    detected_conditions = models.JSONField(default=dict, blank=True)
    confidence_scores = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"X-Ray Analysis for {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"

class ResearchQuery(models.Model):
    """Record of a medical research query"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='research_queries', db_index=True)
    query = models.CharField(max_length=255, db_index=True)
    results = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'query']),
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"Research Query: {self.query}"

class HealthNews(models.Model):
    """Record of health news articles"""
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=100)
    url = models.URLField()
    summary = models.TextField()
    original_content = models.TextField()
    image_url = models.URLField(null=True, blank=True)
    published_date = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        verbose_name_plural = "Health News"
        ordering = ['-published_date', '-created_at']
        indexes = [
            models.Index(fields=['source', 'published_date']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.source})"

class HealthCheckResult(models.Model):
    """Stores results of periodic health checks"""
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    status = models.CharField(max_length=20, default="operational")
    services_status = models.JSONField(default=dict)
    external_data = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"Health Check at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
