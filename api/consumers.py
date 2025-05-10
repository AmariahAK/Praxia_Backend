import json
from datetime import datetime
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from .models import ChatSession, ChatMessage
from .AI.praxia_model import PraxiaAI
import structlog

logger = structlog.get_logger(__name__)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f'chat_{self.session_id}'
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        # Check if user is authenticated and has access to this chat session
        user = self.scope['user']
        if user.is_anonymous:
            await self.close()
            return
        
        has_access = await self.check_session_access(user, self.session_id)
        if not has_access:
            await self.close()
            return
        
        await self.accept()
        
        # Send chat history to the user
        messages = await self.get_chat_history(self.session_id)
        await self.send(text_data=json.dumps({
            'type': 'chat_history',
            'messages': messages
        }))
    
    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        # Save user message to database
        user = self.scope['user']
        user_message = await self.save_message(self.session_id, 'user', message)
        
        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': user_message
            }
        )
        
        # Get user profile
        user_profile = await self.get_user_profile(user)
        
        # Process with AI and get response
        praxia = PraxiaAI()
        ai_response = await self.process_ai_response(praxia, message, user_profile)
        
        # Save AI response to database
        ai_message = await self.save_message(self.session_id, 'assistant', json.dumps(ai_response))
        
        # Send AI response to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': ai_message
            }
        )
    
    # Receive message from room group
    async def chat_message(self, event):
        message = event['message']
        
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'chat_message',
            'message': message
        }))
    
    @database_sync_to_async
    def check_session_access(self, user, session_id):
        try:
            return ChatSession.objects.filter(id=session_id, user=user).exists()
        except Exception as e:
            logger.error("Error checking session access", error=str(e))
            return False
    
    @database_sync_to_async
    def get_chat_history(self, session_id):
        try:
            messages = ChatMessage.objects.filter(session_id=session_id).order_by('created_at')
            return [
                {
                    'id': message.id,
                    'role': message.role,
                    'content': message.content,
                    'created_at': message.created_at.isoformat()
                }
                for message in messages
            ]
        except Exception as e:
            logger.error("Error getting chat history", error=str(e))
            return []
    
    @database_sync_to_async
    def save_message(self, session_id, role, content):
        try:
            message = ChatMessage.objects.create(
                session_id=session_id,
                role=role,
                content=content
            )
            return {
                'id': message.id,
                'role': message.role,
                'content': message.content,
                'created_at': message.created_at.isoformat()
            }
        except Exception as e:
            logger.error("Error saving message", error=str(e))
            return None
    
    @database_sync_to_async
    def get_user_profile(self, user):
        try:
            profile = user.profile
            return {
                'age': profile.age,
                'weight': profile.weight,
                'height': profile.height,
                'country': profile.country,
                'allergies': profile.allergies
            }
        except Exception as e:
            logger.error("Error getting user profile", error=str(e))
            return {}
    
    @database_sync_to_async
    def process_ai_response(self, praxia, message, user_profile):
        try:
            return praxia.diagnose_symptoms(message, user_profile)
        except Exception as e:
            logger.error("Error processing AI response", error=str(e))
            return {"error": "Unable to process your request at this time."}

class HealthCheckConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
    
    async def disconnect(self, close_code):
        pass
    
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message_type = text_data_json.get('type', '')
        
        if message_type == 'ping':
            await self.send(text_data=json.dumps({
                'type': 'pong',
                'timestamp': str(datetime.now())
            }))
        elif message_type == 'health_check':
            # Get latest health check results from cache
            results = await self.get_health_check_results()
            await self.send(text_data=json.dumps({
                'type': 'health_check_results',
                'results': results
            }))
    
    @database_sync_to_async
    def get_health_check_results(self):
        from django.core.cache import cache
        results = cache.get('health_check_results')
        if not results:
            from .AI.ai_healthcheck import scheduled_health_check
            results = scheduled_health_check()
        return results
