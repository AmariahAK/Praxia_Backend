from locust import HttpUser, task, between
import json
import random

class PraxiaUser(HttpUser):
    wait_time = between(1, 5)
    token = None
    
    def on_start(self):
        # Login to get token
        response = self.client.post("/api/auth/login/", {
            "username": "testuser",
            "password": "testpassword"
        })
        self.token = response.json()["token"]
        self.client.headers.update({'Authorization': f'Token {self.token}'})
    
    @task(2)
    def get_chat_sessions(self):
        self.client.get("/api/chat-sessions/")
    
    @task(1)
    def create_chat_session(self):
        self.client.post("/api/chat-sessions/", {
            "title": f"Test Chat {random.randint(1, 1000)}"
        })
    
    @task(5)
    def send_chat_message(self):
        # Get available sessions
        response = self.client.get("/api/chat-sessions/")
        sessions = response.json()
        
        if sessions:
            session_id = sessions[0]["id"]
            symptoms = [
                "I have a headache and fever",
                "My throat hurts when I swallow",
                "I'm experiencing chest pain",
                "I have a rash on my arm",
                "My stomach hurts after eating"
            ]
            
            self.client.post(f"/api/chat-sessions/{session_id}/messages/", {
                "content": random.choice(symptoms)
            })
    
    @task(1)
    def medical_consultation(self):
        symptoms = [
            "Persistent cough for 3 days with mild fever",
            "Headache and dizziness when standing up quickly",
            "Joint pain in knees and elbows, worse in the morning",
            "Rash on chest and back with itching",
            "Difficulty sleeping and daytime fatigue"
        ]
        
        self.client.post("/api/consultations/", {
            "symptoms": random.choice(symptoms)
        })
