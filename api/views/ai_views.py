from rest_framework import status, permissions, viewsets
import json
from ..models import TranslationService
from rest_framework.response import Response
from rest_framework.generics import RetrieveAPIView
from rest_framework.permissions import AllowAny
from ..AI.praxia_model import (
    PraxiaAI,
    analyze_xray_task,
    scrape_health_news,
)
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.shortcuts import get_object_or_404
from django_filters.rest_framework import DjangoFilterBackend
from ..models import ChatSession, ChatMessage, MedicalConsultation, XRayAnalysis, ResearchQuery, HealthNews
from ..serializers import (
    ChatSessionSerializer, 
    ChatSessionListSerializer,
    ChatMessageSerializer, 
    MedicalConsultationSerializer, 
    XRayAnalysisSerializer, 
    ResearchQuerySerializer,
    HealthNewsSerializer,
    HealthCheckResultSerializer
)
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
    parser_classes = [MultiPartParser, FormParser, JSONParser] 
    
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
        
        try:
            # Handle both multipart form data and JSON content
            if request.content_type and 'multipart/form-data' in request.content_type:
                xray_image = request.FILES.get('xray_image')
                message_content = request.data.get('content', '').strip()
            else:
                xray_image = None
                message_content = request.data.get('content', '').strip()
        
            # Ensure we have valid content
            if not message_content:
                if xray_image:
                    message_content = "Please analyze this X-ray image."
                else:
                    message_content = "Hello, I need help with my health."
            
            # Clean up the message content to prevent parsing errors
            message_content = message_content.strip()
        
            # Validate message content length and format
            if len(message_content) < 2:
                message_content = "Hello, I need help with my health."
            
            user_message_serializer = ChatMessageSerializer(data={
                'role': 'user',
                'content': message_content
            })
        
            if not user_message_serializer.is_valid():
                logger.error("Invalid user message", errors=user_message_serializer.errors)
                return Response(user_message_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
            user_message = user_message_serializer.save(session=session)
        
            user_profile = {
                'gender': request.user.profile.gender,
                'age': request.user.profile.age,
                'weight': request.user.profile.weight,
                'height': request.user.profile.height,
                'country': request.user.profile.country,
                'allergies': request.user.profile.allergies
            }
        
            # Filter out None values from user profile
            user_profile = {k: v for k, v in user_profile.items() if v is not None}
        
            ai_response = None
        
            # If there's an X-ray image, process it
            if xray_image:
                # Create an XRayAnalysis object to track the analysis
                xray = XRayAnalysis.objects.create(
                    user=request.user,
                    image=xray_image,
                    analysis_result="Processing...",
                    detected_conditions={},
                    confidence_scores={}
                )
            
                # Start asynchronous analysis
                analyze_xray_task.delay(xray.id, xray.image.path)
            
                # Create an initial AI response indicating the analysis is in progress
                ai_response = {
                    "message": "I'm analyzing your X-ray image. This may take a minute or two.",
                    "xray_analysis_id": xray.id,
                    "status": "processing"
                }
            
                logger.info("X-ray analysis queued from chat", user=request.user.username, xray_id=xray.id)
            else:
                # Regular text message processing
                praxia = PraxiaAI()
                try:
                    # Get the current session title to use as chat topic
                    chat_topic = session.title if session.title != "New Chat" else None
                    
                    if "analyze my diet" in message_content.lower() or "diet analysis" in message_content.lower():
                        ai_response = praxia.analyze_diet(message_content, user_profile)
                    elif "medication" in message_content.lower() or "drug interaction" in message_content.lower():
                        ai_response = praxia.analyze_medication(message_content, user_profile)
                    else:
                        # Pass the chat topic to the diagnosis function
                        ai_response = praxia.diagnose_symptoms(message_content, user_profile, chat_topic)
                    
                    # Ensure we have a valid response
                    if not ai_response or not isinstance(ai_response, dict):
                        raise ValueError("Invalid response from AI")
                    
                except Exception as e:
                    logger.error("Error processing message", error=str(e), symptoms=message_content[:50])
                    # Create a more helpful structured response
                    ai_response = {
                        "diagnosis": {
                            "conditions": ["I need more information to help you better"],
                            "next_steps": [
                                "Could you describe your symptoms in more detail?",
                                "How long have you been experiencing these symptoms?",
                                "Are there any specific triggers you've noticed?"
                            ],
                            "urgent": [],
                            "advice": "I'd be happy to help you with your health concerns. Could you provide more specific details about what you're experiencing?",
                            "clarification": [
                                "What specific symptoms are you experiencing?",
                                "When did these symptoms start?",
                                "Have you tried any treatments so far?"
                            ]
                        },
                        "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
                    }
        
            if ai_response and isinstance(ai_response, dict):
                ai_message = ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=json.dumps(ai_response)  
                )
            
                # Update session title if it's a new session with generic title
                if session.title == "New Chat" and len(session.messages.all()) <= 2:
                    praxia = PraxiaAI()
                    try:
                        # Create a safer topic generation prompt
                        topic_prompt = f"Generate a short 3-5 word title for a medical conversation about: {message_content[:100]}. Respond with only the title, no quotes or extra text."
                        topic = praxia._call_together_ai(topic_prompt).strip()
                        topic = topic.replace('"', '').replace("'", "").strip()
                        if topic and len(topic) > 0 and len(topic) < 100:
                            session.title = topic[:50]  # Limit to 50 chars
                            session.save()
                            logger.info("Generated chat topic", topic=topic)
                    except Exception as e:
                        logger.error("Failed to generate topic", error=str(e))
                        # Set a default meaningful title
                        session.title = "Health Consultation"
                        session.save()
        
            session.save()
            logger.info("Chat message processed", session_id=session_id, user=request.user.username)
            return Response({
                'user_message': ChatMessageSerializer(user_message).data,
                'ai_message': ChatMessageSerializer(ai_message).data
            })
        except Exception as e:
            logger.error("Unexpected error processing chat message", error=str(e))
            # Return a more helpful error response
            try:
                if 'user_message' in locals():
                    error_message = ChatMessage.objects.create(
                        session=session,
                        role='assistant',
                        content=json.dumps({
                            "diagnosis": {
                                "conditions": ["I'm experiencing a temporary issue"],
                                "next_steps": ["Please try rephrasing your question", "If the problem continues, please contact support"],
                                "urgent": [],
                                "advice": "I apologize for the technical difficulty. Could you try asking your question in a different way?",
                                "clarification": ["Could you rephrase your health question?"]
                            },
                            "disclaimer": "This is a system message due to a temporary technical issue."
                        })
                    )
                    return Response({
                        'user_message': ChatMessageSerializer(user_message).data,
                        'ai_message': ChatMessageSerializer(error_message).data
                    })
                else:
                    return Response({
                        'error': 'Unable to process your message at this time. Please try again.'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as inner_e:
                logger.error("Failed to create error response", error=str(inner_e))
                return Response({
                    'error': 'System temporarily unavailable. Please try again.'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class MedicalConsultationView(APIView):
    """View for medical consultations"""
    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [AIConsultationRateThrottle]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['symptoms', 'created_at', 'language']
    
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
                'gender': request.user.profile.gender,
                'age': request.user.profile.age,
                'weight': request.user.profile.weight,
                'height': request.user.profile.height,
                'country': request.user.profile.country,
                'allergies': request.user.profile.allergies
            }
            
            symptoms = serializer.validated_data['symptoms']
            language = serializer.validated_data.get('language', 'en')
            
            # Translate symptoms to English if needed
            translator = TranslationService()
            if language != 'en':
                english_symptoms = translator.translate(symptoms, language, 'en')
            else:
                english_symptoms = symptoms
            
            # Synchronous call (for immediate response)
            praxia = PraxiaAI()
            diagnosis_result = praxia.diagnose_symptoms(english_symptoms, user_profile)
            
            # Translate diagnosis back to original language if needed
            if language != 'en' and isinstance(diagnosis_result, dict):
                diagnosis_json = json.dumps(diagnosis_result)
                translated_diagnosis = translator.translate(diagnosis_json, 'en', language)
                try:
                    diagnosis_result = json.loads(translated_diagnosis)
                except json.JSONDecodeError:
                    pass
            
            consultation = serializer.save(
                user=request.user,
                diagnosis=json.dumps(diagnosis_result)
            )
            logger.info("Consultation created", user=request.user.username, symptoms=symptoms, language=language)
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
            xray = serializer.save(
                user=request.user, 
                analysis_result="Processing...",
                detected_conditions={},
                confidence_scores={}
            )
            
            # Pass the xray ID and image path to the Celery task
            analyze_xray_task.delay(xray.id, xray.image.path)
            
            logger.info("X-ray analysis queued", user=request.user.username, xray_id=xray.id)
            return Response(
                XRayAnalysisSerializer(xray, context={'request': request}).data,
                status=status.HTTP_201_CREATED
            )
        logger.error("Invalid X-ray data", errors=serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class XRayAnalysisDetailView(RetrieveAPIView):
    """View for retrieving a specific X-ray analysis"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = XRayAnalysisSerializer
    queryset = XRayAnalysis.objects.all()

    def get_queryset(self):
        # Only allow users to access their own analyses
        return XRayAnalysis.objects.filter(user=self.request.user)

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

class HealthNewsView(APIView):
    """View for health news"""
    permission_classes = [AllowAny]
    throttle_classes = [AIResearchRateThrottle]
    
    def get(self, request):
        """Get health news articles"""
        source = request.query_params.get('source', 'all')
        limit = min(int(request.query_params.get('limit', 3)), 10)  # Cap at 10 articles
        
        # Use Celery task, but block for result (or you can return a task id and poll)
        news_task = scrape_health_news.delay(source=source, limit=limit)
        try:
            news_articles = news_task.get(timeout=15)
            saved_articles = []
            for article in news_articles:
                obj, created = HealthNews.objects.get_or_create(
                    url=article['url'],
                    defaults={
                        'title': article['title'],
                        'source': article['source'],
                        'summary': article.get('summary', ''),
                        'original_content': article.get('content', ''),
                        'image_url': article.get('image_url'),
                        'published_date': article.get('published_date')
                    }
                )
                saved_articles.append(HealthNewsSerializer(obj).data)
            logger.info("Health news retrieved", source=source, count=len(saved_articles))
            return Response(saved_articles)
        except Exception as e:
            logger.error("Error retrieving health news", error=str(e))
            articles = HealthNews.objects.filter(source__icontains=source if source != 'all' else '')[:limit]
            if articles.exists():
                serializer = HealthNewsSerializer(articles, many=True)
                return Response(serializer.data)
            return Response(
                {"error": str(e), "message": "Unable to retrieve health news at this time."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AuthenticatedHealthCheckView(APIView):
    """View for authenticated health check"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get the latest health check results"""
        from ..models import HealthCheckResult
        
        latest_check = HealthCheckResult.objects.order_by('-timestamp').first()
        
        if not latest_check:
            from ..AI.ai_healthcheck import scheduled_health_check
            health_data = scheduled_health_check()
            return Response({
                "timestamp": health_data["timestamp"],
                "status": health_data["status"],
                "services": health_data["services"],
                "message": "Health check performed on demand"
            })
        
        serializer = HealthCheckResultSerializer(latest_check)
        return Response(serializer.data)
