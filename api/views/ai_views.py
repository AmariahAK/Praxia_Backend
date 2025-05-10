from rest_framework import status, permissions, viewsets
import json
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from django_filters.rest_framework import DjangoFilterBackend
from ..models import ChatSession, ChatMessage, MedicalConsultation, XRayAnalysis, ResearchQuery
from ..serializers import (
    ChatSessionSerializer, 
    ChatSessionListSerializer,
    ChatMessageSerializer, 
    MedicalConsultationSerializer, 
    XRayAnalysisSerializer, 
    ResearchQuerySerializer
)
from ..AI.praxia_model import PraxiaAI
from ..middleware.throttling import (
    AIChatRateThrottle,
    AIConsultationRateThrottle,
    AIXRayRateThrottle,
    AIResearchRateThrottle
)
import structlog

logger = structlog.get_logger(__name__)

class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for chat sessions"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ChatSessionSerializer
    throttle_classes = [AIChatRateThrottle]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['title', 'created_at']
    
    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by('-updated_at')
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ChatSessionListSerializer
        return ChatSessionSerializer
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
        logger.info("Chat session created", user=self.request.user.username)

class ChatMessageView(APIView):
    """View for creating and retrieving chat messages"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIChatRateThrottle]
    
    def get(self, request, session_id):
        """Get all messages for a chat session"""
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        messages = session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        logger.info("Chat messages retrieved", session_id=session_id, user=request.user.username)
        return Response(serializer.data)
    
    def post(self, request, session_id):
        """Create a new message and get AI response"""
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        user_message_serializer = ChatMessageSerializer(data={
            'role': 'user',
            'content': request.data.get('content', '')
        })
        
        if not user_message_serializer.is_valid():
            logger.error("Invalid user message", errors=user_message_serializer.errors)
            return Response(user_message_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        user_message = user_message_serializer.save(session=session)
        
        user_profile = {
            'age': request.user.profile.age,
            'weight': request.user.profile.weight,
            'height': request.user.profile.height,
            'country': request.user.profile.country,
            'allergies': request.user.profile.allergies
        }
        
        praxia = PraxiaAI()
        ai_response = praxia.diagnose_symptoms(user_message.content, user_profile)
        
        ai_message = ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=json.dumps(ai_response)
        )
        
        session.save()
        logger.info("Chat message processed", session_id=session_id, user=request.user.username)
        return Response({
            'user_message': ChatMessageSerializer(user_message).data,
            'ai_message': ChatMessageSerializer(ai_message).data
        })

class MedicalConsultationView(APIView):
    """View for medical consultations"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIConsultationRateThrottle]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['symptoms', 'created_at']
    
    def get(self, request):
        """Get all consultations for the authenticated user"""
        consultations = MedicalConsultation.objects.filter(user=request.user).order_by('-created_at')
        serializer = MedicalConsultationSerializer(consultations, many=True)
        logger.info("Consultations retrieved", user=request.user.username)
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new consultation"""
        serializer = MedicalConsultationSerializer(data=request.data)
        if serializer.is_valid():
            user_profile = {
                'age': request.user.profile.age,
                'weight': request.user.profile.weight,
                'height': request.user.profile.height,
                'country': request.user.profile.country,
                'allergies': request.user.profile.allergies
            }
            
            praxia = PraxiaAI()
            diagnosis_result = praxia.diagnose_symptoms(serializer.validated_data['symptoms'], user_profile)
            
            consultation = serializer.save(
                user=request.user,
                diagnosis=json.dumps(diagnosis_result)
            )
            logger.info("Consultation created", user=request.user.username, symptoms=serializer.validated_data['symptoms'])
            return Response(MedicalConsultationSerializer(consultation).data, status=status.HTTP_201_CREATED)
        logger.error("Invalid consultation data", errors=serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class XRayAnalysisView(APIView):
    """View for X-ray analysis"""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [AIXRayRateThrottle]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['created_at']
    
    def get(self, request):
        """Get all X-ray analyses for the authenticated user"""
        analyses = XRayAnalysis.objects.filter(user=request.user).order_by('-created_at')
        serializer = XRayAnalysisSerializer(analyses, many=True, context={'request': request})
        logger.info("X-ray analyses retrieved", user=request.user.username)
        return Response(serializer.data)
    
    def post(self, request):
        """Upload an X-ray image for analysis"""
        serializer = XRayAnalysisSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            xray = serializer.save(user=request.user, analysis_result="Processing...")
            praxia = PraxiaAI()
            analysis_task = praxia.analyze_xray.delay(xray.image.path)
            logger.info("X-ray analysis queued", user=request.user.username, task_id=analysis_task.id)
            return Response(
                XRayAnalysisSerializer(xray, context={'request': request}).data,
                status=status.HTTP_201_CREATED
            )
        logger.error("Invalid X-ray data", errors=serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ResearchQueryView(APIView):
    """View for medical research queries"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIResearchRateThrottle]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['query', 'created_at']
    
    def get(self, request):
        """Get all research queries for the authenticated user"""
        queries = ResearchQuery.objects.filter(user=request.user).order_by('-created_at')
        serializer = ResearchQuerySerializer(queries, many=True)
        logger.info("Research queries retrieved", user=request.user.username)
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new research query"""
        serializer = ResearchQuerySerializer(data=request.data)
        if serializer.is_valid():
            praxia = PraxiaAI()
            research_results = praxia.get_medical_research(serializer.validated_data['query'])
            query = serializer.save(
                user=request.user,
                results=research_results
            )
            logger.info("Research query created", user=request.user.username, query=serializer.validated_data['query'])
            return Response(ResearchQuerySerializer(query).data, status=status.HTTP_201_CREATED)
        logger.error("Invalid research query", errors=serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)