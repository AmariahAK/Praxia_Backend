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
            # IMPROVED: Better input validation and error handling
            if request.content_type and 'multipart/form-data' in request.content_type:
                xray_image = request.FILES.get('xray_image')
                message_content = request.data.get('content', '').strip()
            else:
                xray_image = None
                message_content = request.data.get('content', '').strip()
        
            # ENHANCED: Validate message content more thoroughly
            if not message_content and not xray_image:
                return Response({
                    'error': 'Either message content or X-ray image is required'
                }, status=status.HTTP_400_BAD_REQUEST)
        
            if not message_content:
                message_content = "Please analyze this X-ray image." if xray_image else "Hello, I need help with my health."
        
            # IMPROVED: Length validation
            if len(message_content) > 2000:
                message_content = message_content[:2000] + "..."
        
            # Create user message with better error handling
            try:
                user_message_serializer = ChatMessageSerializer(data={
                    'role': 'user',
                    'content': message_content
                })
        
                if not user_message_serializer.is_valid():
                    logger.error("Invalid user message", errors=user_message_serializer.errors)
                    return Response(user_message_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
                user_message = user_message_serializer.save(session=session)
            except Exception as e:
                logger.error("Failed to create user message", error=str(e))
                return Response({
                    'error': 'Failed to process message'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
            # ENHANCED: Get user profile with better error handling
            user_profile = {}
            try:
                if hasattr(request.user, 'profile'):
                    profile = request.user.profile
                    user_profile = {k: v for k, v in {
                        'gender': getattr(profile, 'gender', None),
                        'age': getattr(profile, 'age', None),
                        'weight': getattr(profile, 'weight', None),
                        'height': getattr(profile, 'height', None),
                        'country': getattr(profile, 'country', None),
                        'allergies': getattr(profile, 'allergies', None)
                    }.items() if v is not None}
            except Exception as e:
                logger.warning("Error getting user profile", error=str(e))
                user_profile = {}
        
            ai_response = None
        
            try:
                if xray_image:
                    # X-ray processing with size limits
                    if xray_image.size > 10 * 1024 * 1024:  # 10MB limit
                        return Response({
                            'error': 'X-ray image too large. Maximum size is 10MB.'
                        }, status=status.HTTP_400_BAD_REQUEST)
                
                    xray = XRayAnalysis.objects.create(
                        user=request.user,
                        image=xray_image,
                        analysis_result="Processing...",
                        detected_conditions={},
                        confidence_scores={}
                    )
            
                    # Use Celery with timeout
                    analyze_xray_task.apply_async(
                        args=[xray.id, xray.image.path],
                        countdown=2,  # Start after 2 seconds
                        expires=300   # Expire after 5 minutes
                    )
            
                    ai_response = {
                        "message": "I'm analyzing your X-ray image. This may take a minute or two.",
                        "xray_analysis_id": xray.id,
                        "status": "processing"
                    }
                    logger.info("X-ray analysis queued", user=request.user.username, xray_id=xray.id)
                else:
                    # ENHANCED: Regular text processing with timeout and fallback
                    from django.core.exceptions import ValidationError
                    
                    praxia = PraxiaAI()
                    chat_topic = session.title if session.title != "New Chat" else None
                    
                    # Add timeout wrapper
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("AI processing timeout")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(45)  # 45 second timeout
                    
                    try:
                        if "analyze my diet" in message_content.lower():
                            ai_response = praxia.analyze_diet(message_content, user_profile)
                        elif "medication" in message_content.lower():
                            ai_response = praxia.analyze_medication(message_content, user_profile)
                        else:
                            ai_response = praxia.diagnose_symptoms(message_content, user_profile, chat_topic)
                    
                        signal.alarm(0)  # Cancel timeout
                    
                    except TimeoutError:
                        logger.error("AI processing timeout", user=request.user.username)
                        ai_response = self._get_timeout_response()
                    except Exception as e:
                        signal.alarm(0)  # Cancel timeout
                        logger.error("AI processing error", error=str(e))
                        ai_response = self._get_error_response(str(e))
        
            except Exception as e:
                logger.error("Unexpected error in message processing", error=str(e))
                ai_response = self._get_error_response("System temporarily unavailable")
        
            # IMPROVED: Save AI response with validation
            if ai_response and isinstance(ai_response, dict):
                try:
                    # Ensure response is properly formatted
                    formatted_response = self._format_ai_response(ai_response)
                    
                    ai_message = ChatMessage.objects.create(
                        session=session,
                        role='assistant',
                        content=json.dumps(formatted_response, ensure_ascii=False)
                    )
                    
                    # Update session title if needed
                    self._update_session_title(session, ai_response, message_content)
                    
                    session.save()
                    logger.info("Message processed successfully", session_id=session_id, user=request.user.username)
                    
                    return Response({
                        'user_message': ChatMessageSerializer(user_message).data,
                        'ai_message': ChatMessageSerializer(ai_message).data
                    })
                    
                except Exception as e:
                    logger.error("Failed to save AI response", error=str(e))
                    return Response({
                        'error': 'Failed to process AI response'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                logger.error("Invalid AI response format")
                return Response({
                    'error': 'Invalid response from AI system'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        except Exception as e:
            logger.error("Critical error in chat message processing", error=str(e))
            return Response({
                'error': 'System error. Please try again.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _get_timeout_response(self):
        """Get response for timeout scenarios"""
        return {
            "diagnosis": {
                "conditions": ["Processing timeout occurred"],
                "next_steps": ["Please try again with a shorter message", "Contact support if the issue persists"],
                "urgent": [],
                "advice": "I'm taking longer than usual to process your request. Please try again.",
                "clarification": ["Could you try rephrasing your question more concisely?"]
            },
            "disclaimer": "This is a timeout response. Please try again."
        }

    def _get_error_response(self, error_msg):
        """Get response for error scenarios"""
        return {
            "diagnosis": {
                "conditions": ["System temporarily unavailable"],
                "next_steps": ["Please try again in a few moments", "Contact support if the issue persists"],
                "urgent": [],
                "advice": f"I'm experiencing technical difficulties: {error_msg}",
                "clarification": ["Would you like to try again?"]
            },
            "disclaimer": "This is an error response due to technical issues."
        }

    def _format_ai_response(self, response):
        """Ensure AI response is properly formatted"""
        if not isinstance(response, dict):
            return {"error": "Invalid response format"}
        
        # Ensure required fields exist
        if 'diagnosis' not in response:
            response['diagnosis'] = {
                "conditions": ["Response formatting error"],
                "next_steps": ["Please try again"],
                "urgent": [],
                "advice": "There was an issue formatting the response.",
                "clarification": []
            }
        
        return response

    def _update_session_title(self, session, ai_response, message_content):
        """Update session title if it's still 'New Chat'"""
        try:
            if session.title == "New Chat" and message_content:
                # Extract a meaningful title from the message
                title_words = message_content.split()[:5]  # First 5 words
                new_title = ' '.join(title_words)
                if len(new_title) > 50:
                    new_title = new_title[:50] + "..."
                session.title = new_title
                session.save()
        except Exception as e:
            logger.warning("Failed to update session title", error=str(e))

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
            # Get user profile safely
            user_profile = {}
            try:
                if hasattr(request.user, 'profile'):
                    user_profile = {
                        'gender': getattr(request.user.profile, 'gender', None),
                        'age': getattr(request.user.profile, 'age', None),
                        'weight': getattr(request.user.profile, 'weight', None),
                        'height': getattr(request.user.profile, 'height', None),
                        'country': getattr(request.user.profile, 'country', None),
                        'allergies': getattr(request.user.profile, 'allergies', None)
                    }
                    # Filter out None values from user profile
                    user_profile = {k: v for k, v in user_profile.items() if v is not None}
            except Exception as e:
                logger.warning("Error getting user profile", error=str(e))
                user_profile = {}
            
            symptoms = serializer.validated_data['symptoms']
            language = serializer.validated_data.get('language', 'en')
            
            # Translate symptoms to English if needed
            try:
                translator = TranslationService()
                if language != 'en':
                    english_symptoms = translator.translate(symptoms, language, 'en')
                else:
                    english_symptoms = symptoms
            except Exception as e:
                logger.warning("Translation service error", error=str(e))
                english_symptoms = symptoms
            
            # Synchronous call (for immediate response)
            praxia = PraxiaAI()
            diagnosis_result = praxia.diagnose_symptoms(english_symptoms, user_profile)
            
            # Translate diagnosis back to original language if needed
            try:
                if language != 'en' and isinstance(diagnosis_result, dict):
                    diagnosis_json = json.dumps(diagnosis_result)
                    translated_diagnosis = translator.translate(diagnosis_json, 'en', language)
                    try:
                        diagnosis_result = json.loads(translated_diagnosis)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.warning("Translation of diagnosis failed", error=str(e))
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
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

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
