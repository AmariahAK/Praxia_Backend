import pytest
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from factory import Faker, SubFactory
from factory.django import DjangoModelFactory
from api.models import UserProfile, ChatSession, ChatMessage, MedicalConsultation, XRayAnalysis, ResearchQuery
from api.AI.praxia_model import PraxiaAI
import json

# Factories
class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
    username = Faker('user_name')
    email = Faker('email')
    password = Faker('password')

class UserProfileFactory(DjangoModelFactory):
    class Meta:
        model = UserProfile
    user = SubFactory(UserFactory)
    age = Faker('pyint', min_value=18, max_value=80)
    weight = Faker('pyfloat', min_value=50, max_value=150)
    height = Faker('pyfloat', min_value=150, max_value=200)
    country = Faker('country')
    allergies = Faker('word')

class ChatSessionFactory(DjangoModelFactory):
    class Meta:
        model = ChatSession
    user = SubFactory(UserFactory)
    title = Faker('sentence')

class ChatMessageFactory(DjangoModelFactory):
    class Meta:
        model = ChatMessage
    session = SubFactory(ChatSessionFactory)
    role = 'user'
    content = Faker('text')

class MedicalConsultationFactory(DjangoModelFactory):
    class Meta:
        model = MedicalConsultation
    user = SubFactory(UserFactory)
    symptoms = Faker('text')
    diagnosis = json.dumps({"conditions": ["Test"], "next_steps": ["Test"]})

class XRayAnalysisFactory(DjangoModelFactory):
    class Meta:
        model = XRayAnalysis
    user = SubFactory(UserFactory)
    image = Faker('file_path', extension='png')
    analysis_result = json.dumps({"findings": ["Test"], "confidence": 0.5})

class ResearchQueryFactory(DjangoModelFactory):
    class Meta:
        model = ResearchQuery
    user = SubFactory(UserFactory)
    query = Faker('sentence')
    results = [{"title": "Test", "authors": "Test"}]

# Model Tests
@pytest.mark.django_db
class ModelTests(TestCase):
    def test_user_profile_creation(self):
        user = UserFactory()
        profile = UserProfile.objects.get(user=user)
        assert profile.user == user
        assert profile.age is not None

    def test_chat_session_str(self):
        session = ChatSessionFactory()
        assert str(session) == f"Chat with {session.user.username} - {session.title}"

    def test_medical_consultation_creation(self):
        consultation = MedicalConsultationFactory()
        assert consultation.symptoms
        assert json.loads(consultation.diagnosis)["conditions"]

# Serializer Tests
@pytest.mark.django_db
class SerializerTests(TestCase):
    def test_chat_message_serializer(self):
        message = ChatMessageFactory()
        serializer = ChatMessageSerializer(message)
        assert serializer.data['role'] == 'user'
        assert serializer.data['content']

    def test_medical_consultation_serializer(self):
        consultation = MedicalConsultationFactory()
        serializer = MedicalConsultationSerializer(consultation)
        assert serializer.data['symptoms']
        assert serializer.data['diagnosis']

# View Tests
@pytest.mark.django_db
class ViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserFactory()
        self.client.force_authenticate(user=self.user)
    
    def test_chat_session_list(self):
        ChatSessionFactory.create_batch(3, user=self.user)
        response = self.client.get(reverse('chat-session-list'))
        assert response.status_code == 200
        assert len(response.data) == 3
    
    def test_medical_consultation_create(self):
        data = {"symptoms": "fever, cough"}
        response = self.client.post(reverse('consultations'), data)
        assert response.status_code == 201
        assert MedicalConsultation.objects.count() == 1
    
    def test_xray_analysis_create(self):
        data = {"image": open("test.png", "rb")}
        response = self.client.post(reverse('xray-analyses'), data, format='multipart')
        assert response.status_code == 201
        assert XRayAnalysis.objects.count() == 1

# PraxiaAI Tests
@pytest.mark.django_db
class PraxiaAITests(TestCase):
    def test_diagnose_symptoms(self):
        praxia = PraxiaAI()
        result = praxia.diagnose_symptoms("fever, cough", {"age": 30, "weight": 70})
        assert "conditions" in result or "error" in result
    
    def test_get_medical_research(self):
        praxia = PraxiaAI()
        result = praxia.get_medical_research("tuberculosis")
        assert isinstance(result, list)
        assert len(result) <= 5