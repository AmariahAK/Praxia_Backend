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

# Update the ChatMessageView post method to handle the new methods
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
                    try:
                        # Use the extracted medical topic for the session title
                        if ai_response and isinstance(ai_response, dict) and ai_response.get('medical_topic'):
                            topic = ai_response['medical_topic']
                            # Clean and format the topic for title
                            clean_title = ' '.join(topic.split()[:4])  # Limit to 4 words
                            clean_title = clean_title.title()  # Capitalize
                            if clean_title and len(clean_title) > 3:
                                session.title = clean_title[:50]  # Limit to 50 chars
                                session.save()
                                logger.info("Generated chat topic from medical topic", topic=clean_title)
                        else:
                            # Fallback to original method
                            praxia = PraxiaAI()
                            topic_prompt = f"Generate a short 3-4 word medical title for: {message_content[:50]}. Respond with only the title, no quotes."
                            topic = praxia._call_together_ai(topic_prompt).strip()
                            topic = topic.replace('"', '').replace("'", "").strip()
                            if topic and len(topic) > 0 and len(topic) < 100:
                                session.title = topic[:50]
                                session.save()
                                logger.info("Generated chat topic from AI", topic=topic)
                    except Exception as e:
                        logger.error("Failed to generate topic", error=str(e))
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
