from rest_framework import status, permissions, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
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

class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for chat sessions"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ChatSessionSerializer
    throttle_classes = [AIChatRateThrottle]
    
    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by('-updated_at')
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ChatSessionListSerializer
        return ChatSessionSerializer
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ChatMessageView(APIView):
    """View for creating and retrieving chat messages"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIChatRateThrottle]
    
    def get(self, request, session_id):
        """Get all messages for a chat session"""
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        messages = session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    def post(self, request, session_id):
        """Create a new message and get AI response"""
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        # Validate user message
        user_message_serializer = ChatMessageSerializer(data={
            'role': 'user',
            'content': request.data.get('content', '')
        })
        
        if not user_message_serializer.is_valid():
            return Response(user_message_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Save user message
        user_message = user_message_serializer.save(session=session)
        
        # Get user profile for context
        user_profile = {
            'age': request.user.profile.age,
            'weight': request.user.profile.weight,
            'height': request.user.profile.height,
            'country': request.user.profile.country,
            'allergies': request.user.profile.allergies
        }
        
        # Get AI response
        praxia = PraxiaAI()
        ai_response = praxia.diagnose_symptoms(user_message.content, user_profile)
        
        # Save AI response
        ai_message = ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=ai_response.get('diagnosis', 'I apologize, but I was unable to process your request.')
        )
        
        # Update session timestamp
        session.save()  # This will update the updated_at field
        
        # Return both messages
        return Response({
            'user_message': ChatMessageSerializer(user_message).data,
            'ai_message': ChatMessageSerializer(ai_message).data
        })

class MedicalConsultationView(APIView):
    """View for medical consultations"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIConsultationRateThrottle]
    
    def get(self, request):
        """Get all consultations for the authenticated user"""
        consultations = MedicalConsultation.objects.filter(user=request.user).order_by('-created_at')
        serializer = MedicalConsultationSerializer(consultations, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new consultation"""
        serializer = MedicalConsultationSerializer(data=request.data)
        if serializer.is_valid():
            # Get user profile for context
            user_profile = {
                'age': request.user.profile.age,
                'weight': request.user.profile.weight,
                'height': request.user.profile.height,
                'country': request.user.profile.country,
                'allergies': request.user.profile.allergies
            }
            
            # Get diagnosis from Praxia
            praxia = PraxiaAI()
            diagnosis_result = praxia.diagnose_symptoms(serializer.validated_data['symptoms'], user_profile)
            
            # Save consultation with diagnosis
            consultation = serializer.save(
                user=request.user,
                diagnosis=diagnosis_result.get('diagnosis', 'Unable to generate diagnosis')
            )
            
            return Response(MedicalConsultationSerializer(consultation).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class XRayAnalysisView(APIView):
    """View for X-ray analysis"""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [AIXRayRateThrottle]
    
    def get(self, request):
        """Get all X-ray analyses for the authenticated user"""
        analyses = XRayAnalysis.objects.filter(user=request.user).order_by('-created_at')
        serializer = XRayAnalysisSerializer(analyses, many=True, context={'request': request})
        return Response(serializer.data)
    
    def post(self, request):
        """Upload an X-ray image for analysis"""
        serializer = XRayAnalysisSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            # Save the X-ray image first
            xray = serializer.save(user=request.user, analysis_result="Processing...")
            
            # Process the X-ray image asynchronously
            praxia = PraxiaAI()
            analysis_task = praxia.analyze_xray.delay(xray.image.path)
            
            # Return the X-ray object with a processing message
            return Response(
                XRayAnalysisSerializer(xray, context={'request': request}).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ResearchQueryView(APIView):
    """View for medical research queries"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIResearchRateThrottle]
    
    def get(self, request):
        """Get all research queries for the authenticated user"""
        queries = ResearchQuery.objects.filter(user=request.user).order_by('-created_at')
        serializer = ResearchQuerySerializer(queries, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new research query"""
        serializer = ResearchQuerySerializer(data=request.data)
        if serializer.is_valid():
            # Get research results from Praxia
            praxia = PraxiaAI()
            research_results = praxia.get_medical_research(serializer.validated_data['query'])
            
            # Save query with results
            query = serializer.save(
                user=request.user,
                results=research_results
            )
            
            return Response(ResearchQuerySerializer(query).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
