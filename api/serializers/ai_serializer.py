from rest_framework import serializers
from ..models import (
    ChatSession, 
    ChatMessage, 
    MedicalConsultation, 
    XRayAnalysis, 
    ResearchQuery
)

class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages"""
    class Meta:
        model = ChatMessage
        fields = ('id', 'role', 'content', 'created_at')

class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for chat sessions"""
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ('id', 'title', 'created_at', 'updated_at', 'messages')

class ChatSessionListSerializer(serializers.ModelSerializer):
    """Serializer for listing chat sessions"""
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = ('id', 'title', 'created_at', 'updated_at', 'last_message')
    
    def get_last_message(self, obj):
        last_message = obj.messages.order_by('-created_at').first()
        if last_message:
            return ChatMessageSerializer(last_message).data
        return None

class MedicalConsultationSerializer(serializers.ModelSerializer):
    """Serializer for medical consultations"""
    class Meta:
        model = MedicalConsultation
        fields = ('id', 'symptoms', 'diagnosis', 'created_at')
        read_only_fields = ('diagnosis', 'created_at')

class XRayAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for X-ray analyses"""
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = XRayAnalysis
        fields = ('id', 'image', 'image_url', 'analysis_result', 'created_at')
        read_only_fields = ('analysis_result', 'created_at')
        extra_kwargs = {
            'image': {'write_only': True}
        }
    
    def get_image_url(self, obj):
        if obj.image:
            return self.context['request'].build_absolute_uri(obj.image.url)
        return None

class ResearchQuerySerializer(serializers.ModelSerializer):
    """Serializer for research queries"""
    class Meta:
        model = ResearchQuery
        fields = ('id', 'query', 'results', 'created_at')
        read_only_fields = ('results', 'created_at')
