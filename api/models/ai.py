from django.db import models
from django.contrib.auth.models import User

class ChatSession(models.Model):
    """Record of a chat session between a user and Praxia"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    title = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chat with {self.user.username} - {self.title}"

class ChatMessage(models.Model):
    """Individual message in a chat session"""
    ROLE_CHOICES = (
        ('user', 'User'),
        ('assistant', 'Assistant'),
    )
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.role} message in {self.session}"

class MedicalConsultation(models.Model):
    """Record of a user's medical consultation with Praxia"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='consultations')
    symptoms = models.TextField()
    diagnosis = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Consultation for {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"

class XRayAnalysis(models.Model):
    """Record of an X-ray analysis"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='xray_analyses')
    image = models.ImageField(upload_to='xrays/')
    analysis_result = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"X-Ray Analysis for {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"

class ResearchQuery(models.Model):
    """Record of a medical research query"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='research_queries')
    query = models.CharField(max_length=255)
    results = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Research Query: {self.query}"
