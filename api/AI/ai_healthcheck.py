import os
import requests
import inspect
import pybreaker
import json
import pymed
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from django.conf import settings
from django.core.cache import cache
from monai.networks.nets import UNETR
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureChannelFirst, Resize
from transformers import SamModel, SamProcessor
from celery import shared_task
from bs4 import BeautifulSoup
import structlog
from datetime import datetime
from ..circuit_breaker import (
    who_breaker, mayo_breaker, together_ai_breaker, pubmed_breaker,
    circuit_breaker_with_fallback, retry_with_backoff, cache_result,
    who_api_fallback, mayo_clinic_fallback, together_ai_fallback, pubmed_fallback
)

logger = structlog.get_logger(__name__)

class PraxiaAI:
    """
    Praxia AI model for healthcare assistance
    Developed by Amariah Kamau (https://www.linkedin.com/in/amariah-kamau-3156412a6/)
    GitHub: https://github.com/AmariahAK
    """
    
    def __init__(self):
        self.identity = self._load_identity()
        self.together_api_key = settings.TOGETHER_AI_API_KEY
        self.together_model = settings.TOGETHER_AI_MODEL
        self.cache_timeout = 60 * 60 * 24  # 24 hours cache
        self.pubmed_client = pymed.PubMed(tool="PraxiaAI", email="contact@praxia.ai")
        
        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xray_model = None
        self.sam_model = None
        if settings.INITIALIZE_XRAY_MODEL:
            self._initialize_xray_model()
            self._initialize_sam_model()
    
    def _load_identity(self):
        """Load the AI identity from the text file"""
        identity_path = os.path.join(settings.BASE_DIR, 'data', 'ai_identity.txt')
        try:
            with open(identity_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logger.error("AI identity file not found", path=identity_path)
            return "Praxia - A healthcare AI assistant by Amariah Kamau"
    
    def _initialize_xray_model(self):
        """Initialize the MONAI UNETR model for X-ray analysis"""
        try:
            self.xray_model = UNETR(
                in_channels=1,
                out_channels=2,  # Binary classification (normal/abnormal)
                img_size=(224, 224),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                num_layers=12,
                norm_name='instance',
                dropout_rate=0.0,
            ).to(self.device)
            
            model_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'xray_model.pth')
            if os.path.exists(model_path):
                self.xray_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.xray_model.eval()
            logger.info("MONAI UNETR model initialized", device=self.device)
        except Exception as e:
            logger.error("Error initializing X-ray model", error=str(e))
            self.xray_model = None
    
    def _initialize_sam_model(self):
        """Initialize SAM-Med2D for X-ray segmentation"""
        try:
            self.sam_model = SamModel.from_pretrained("facebook/segment-anything").to(self.device)
            self.sam_processor = SamProcessor.from_pretrained("facebook/segment-anything")
            model_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'sam_med2d.pth')
            if os.path.exists(model_path):
                self.sam_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.sam_model.eval()
            logger.info("SAM-Med2D model initialized", device=self.device)
        except Exception as e:
            logger.error("Error initializing SAM-Med2D model", error=str(e))
            self.sam_model = None
    
    @shared_task
    @cache_result(timeout=60*60*24, key_prefix='diagnosis')
    def diagnose_symptoms(self, symptoms, user_profile=None):
        """
        Analyze symptoms and provide potential diagnoses
        """
        context = self._build_user_context(user_profile)
        who_guidelines = self._fetch_who_guidelines(symptoms)
        mayo_info = self._scrape_mayo_clinic(symptoms)
        
        prompt = f"""You are Praxia, a medical AI assistant. {self.identity}
        
        {context}
        
        WHO Guidelines: {who_guidelines}
        Mayo Clinic Info: {mayo_info}
        Symptoms: {symptoms}
        
        Provide a detailed analysis including:
        1. Potential conditions
        2. Recommended next steps
        3. When to seek immediate medical attention
        4. General advice for managing symptoms
        
        Format in JSON:
        ```json
        {
            "conditions": [],
            "next_steps": [],
            "urgent": [],
            "advice": []
        }
        ```
        """
        
        try:
            response = self._call_together_ai(prompt)
            result = json.loads(response)
            result["related_research"] = self.get_medical_research(symptoms, limit=3)
            result["disclaimer"] = "This is for educational purposes only."
            logger.info("Diagnosis generated", symptoms=symptoms)
            return result
        except Exception as e:
            logger.error("Diagnosis failed", error=str(e), symptoms=symptoms)
            return {"error": str(e), "message": "Unable to process diagnosis."}
    
    @shared_task
    def analyze_xray(self, image_path):
        """
        Analyze X-ray images using MONAI and SAM-Med2D
        """
        if not self.xray_model or not self.sam_model:
            logger.error("X-ray models not initialized")
            return {"error": "Models not initialized", "message": "X-ray analysis unavailable."}
        
        try:
            transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                Resize((224, 224)),
            ])
            image = transforms(image_path)
            image = image.unsqueeze(0).to(self.device)
            
            # MONAI classification
            with torch.no_grad():
                output = self.xray_model(image)
                probabilities = torch.softmax(output, dim=1)
                abnormal_prob = probabilities[0, 1].item()
            
            # SAM-Med2D segmentation
            image_pil = Image.open(image_path).convert('RGB')
            inputs = self.sam_processor(image_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
                masks = outputs.pred_masks.squeeze().cpu().numpy()
            
            findings = []
            if abnormal_prob > 0.7:
                findings.append("High-confidence abnormality detected")
            elif abnormal_prob > 0.4:
                findings.append("Moderate-confidence abnormality detected")
            else:
                findings.append("No significant abnormalities")
            
            result = {
                "analysis": "X-ray analysis completed",
                "findings": findings,
                "confidence": abnormal_prob,
                "masks": masks.tolist(),
                "related_research": self.get_medical_research("X-ray abnormalities diagnosis", limit=2),
                "disclaimer": "This is an AI interpretation; confirm with a radiologist."
            }
            logger.info("X-ray analysis completed", image_path=image_path)
            return result
        except Exception as e:
            logger.error("X-ray analysis failed", error=str(e), image_path=image_path)
            return {"error": str(e), "message": "Unable to process X-ray."}
    
    @shared_task
    @circuit_breaker_with_fallback(pubmed_breaker, pubmed_fallback)
    @cache_result(timeout=60*60*24, key_prefix='research')
    def get_medical_research(self, query, limit=5):
        """
        Retrieve relevant medical research from PubMed
        """
        try:
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp])"
            results = self.pubmed_client.query(search_term, max_results=limit)
            articles = [{
                "title": article.title,
                "authors": ", ".join([author['lastname'] + ' ' + author['firstname'][0] for author in article.authors]) if article.authors else "Unknown",
                "journal": article.journal,
                "publication_date": str(article.publication_date) if hasattr(article, 'publication_date') else "Unknown",
                "doi": article.doi if hasattr(article, 'doi') else None,
                "abstract": article.abstract if hasattr(article, 'abstract') else "No abstract available"
            } for article in results]
            logger.info("Research fetched", query=query)
            return articles
        except Exception as e:
            logger.error("Research query failed", error=str(e), query=query)
            raise  # Let the circuit breaker handle this
    
    @shared_task
    @cache_result(timeout=60*60*24, key_prefix='diet')
    def analyze_diet(self, diet_info, user_profile=None):
        """
        Analyze diet information and provide nutritional recommendations
        """
        context = self._build_user_context(user_profile)
        prompt = f"""You are Praxia, a medical AI assistant. {self.identity}
        
        {context}
        
        Diet: {diet_info}
        
        Provide a nutritional analysis in JSON:
        ```json
        {
            "balance": "",
            "deficiencies": [],
            "recommendations": [],
            "foods": {"add": [], "remove": []}
        }
        ```
        """
        
        try:
            response = self._call_together_ai(prompt)
            result = json.loads(response)
            result["disclaimer"] = "This is for educational purposes only."
            logger.info("Diet analysis generated", diet_info=diet_info)
            return result
        except Exception as e:
            logger.error("Diet analysis failed", error=str(e), diet_info=diet_info)
            return {"error": str(e), "message": "Unable to process diet analysis."}
    
    @circuit_breaker_with_fallback(together_ai_breaker, together_ai_fallback)
    @retry_with_backoff(max_retries=3, initial_backoff=1, backoff_factor=2)
    def _call_together_ai(self, prompt):
        """Call Together AI API"""
        url = "https://api.together.xyz/v1/completions"
        payload = {
            "model": self.together_model,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "stop": ["<human>", "<assistant>"]
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.together_api_key}"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    
    @circuit_breaker_with_fallback(who_breaker, who_api_fallback)
    @cache_result(timeout=60*60*24, key_prefix='who')
    def _fetch_who_guidelines(self, query):
        """Fetch WHO guidelines for a disease or symptom"""
        response = requests.get(
            "https://ghoapi.azureedge.net/api/Indicator",
            params={"$filter": f"contains(IndicatorName, '{query}')"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()["value"]
        guidelines = " ".join([item["IndicatorName"] for item in data[:3]])
        logger.info("WHO guidelines fetched", query=query)
        return guidelines
    
    @circuit_breaker_with_fallback(mayo_breaker, mayo_clinic_fallback)
    @cache_result(timeout=60*60*24, key_prefix='mayo')
    def _scrape_mayo_clinic(self, query):
        """Scrape Mayo Clinic for symptom or disease info"""
        url = f"https://www.mayoclinic.org/diseases-conditions/{query.replace(' ', '-')}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="content")
        info = content.text.strip() if content else "No data found."
        logger.info("Mayo Clinic data scraped", query=query)
        return info
    
    def _build_user_context(self, user_profile):
        """Build context from user profile"""
        if not user_profile:
            return ""
        context = f"Patient: Age {user_profile.get('age', 'unknown')}, Weight {user_profile.get('weight', 'unknown')}kg, "
        context += f"Height {user_profile.get('height', 'unknown')}cm, Country: {user_profile.get('country', 'unknown')}. "
        if user_profile.get('allergies'):
            context += f"Allergies: {user_profile.get('allergies')}. "
        return context

# Add health check functions for Celery tasks
@shared_task
def scheduled_health_check():
    """Scheduled health check to ensure all services are operational"""
    from ..circuit_breaker import check_circuit_breakers
    
    results = {
        "timestamp": str(datetime.now()),
        "status": "operational",
        "services": {}
    }
    
    # Check database connection
    try:
        from django.db import connections
        connections['default'].cursor()
        results["services"]["database"] = "operational"
    except Exception as e:
        results["services"]["database"] = f"error: {str(e)}"
        results["status"] = "degraded"
    
    # Check Redis connection
    try:
        from django.core.cache import cache
        cache.set('health_check', 'ok', 10)
        assert cache.get('health_check') == 'ok'
        results["services"]["redis"] = "operational"
    except Exception as e:
        results["services"]["redis"] = f"error: {str(e)}"
        results["status"] = "degraded"
    
        # Check circuit breakers
    circuit_breaker_status = check_circuit_breakers()
    results["services"]["circuit_breakers"] = circuit_breaker_status
    
    # Check external services
    external_services = {
        "who_api": who_breaker,
        "mayo_clinic": mayo_breaker,
        "together_ai": together_ai_breaker,
        "pubmed": pubmed_breaker
    }
    
    for name, breaker in external_services.items():
        if breaker.current_state == pybreaker.STATE_CLOSED:
            results["services"][name] = "operational"
        else:
            results["services"][name] = "degraded"
            results["status"] = "degraded"
    
    # Check AI models
    try:
        praxia = PraxiaAI()
        if settings.INITIALIZE_XRAY_MODEL:
            results["services"]["xray_model"] = "operational" if praxia.xray_model else "not_loaded"
            results["services"]["sam_model"] = "operational" if praxia.sam_model else "not_loaded"
        else:
            results["services"]["xray_model"] = "disabled"
            results["services"]["sam_model"] = "disabled"
    except Exception as e:
        results["services"]["ai_models"] = f"error: {str(e)}"
        results["status"] = "degraded"
    
    # Check Celery workers
    try:
        insp = inspect()
        if not insp.ping():
            results["services"]["celery"] = "no_workers_online"
            results["status"] = "degraded"
        else:
            results["services"]["celery"] = "operational"
            
            # Check active tasks
            active = insp.active()
            if active:
                results["services"]["celery_active_tasks"] = sum(len(tasks) for tasks in active.values())
            
            # Check queue lengths
            from django.conf import settings
            import redis
            redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
            queue_length = redis_client.llen('celery')
            results["services"]["celery_queue_length"] = queue_length
            
            if queue_length > 100:  # Arbitrary threshold
                results["services"]["celery_queue_status"] = "backlogged"
                results["status"] = "degraded"
            else:
                results["services"]["celery_queue_status"] = "normal"
    except Exception as e:
        results["services"]["celery"] = f"error: {str(e)}"
        results["status"] = "degraded"
    
    # Store health check results in cache
    cache.set('health_check_results', results, 60 * 15)  # Cache for 15 minutes
    
    # Log health check results
    if results["status"] == "operational":
        logger.info("Health check passed", services=results["services"])
    else:
        logger.warning("Health check detected issues", services=results["services"])
    
    return results

@shared_task
def startup_health_check():
    """Health check to run at startup"""
    from datetime import datetime
    
    logger.info("Running startup health check")
    
    results = {
        "timestamp": str(datetime.now()),
        "status": "operational",
        "services": {}
    }
    
    # Check database connection
    try:
        from django.db import connections
        connections['default'].cursor()
        results["services"]["database"] = "operational"
        logger.info("Database connection successful")
    except Exception as e:
        results["services"]["database"] = f"error: {str(e)}"
        results["status"] = "degraded"
        logger.error("Database connection failed", error=str(e))
    
    # Check Redis connection
    try:
        from django.core.cache import cache
        cache.set('startup_health_check', 'ok', 10)
        assert cache.get('startup_health_check') == 'ok'
        results["services"]["redis"] = "operational"
        logger.info("Redis connection successful")
    except Exception as e:
        results["services"]["redis"] = f"error: {str(e)}"
        results["status"] = "degraded"
        logger.error("Redis connection failed", error=str(e))
    
    # Check AI model initialization
    try:
        if settings.INITIALIZE_XRAY_MODEL:
            praxia = PraxiaAI()
            results["services"]["xray_model"] = "operational" if praxia.xray_model else "not_loaded"
            results["services"]["sam_model"] = "operational" if praxia.sam_model else "not_loaded"
            logger.info(
                "AI models initialization check", 
                xray_model=results["services"]["xray_model"],
                sam_model=results["services"]["sam_model"]
            )
        else:
            results["services"]["xray_model"] = "disabled"
            results["services"]["sam_model"] = "disabled"
            logger.info("AI models disabled in settings")
    except Exception as e:
        results["services"]["ai_models"] = f"error: {str(e)}"
        results["status"] = "degraded"
        logger.error("AI model initialization check failed", error=str(e))
    
    # Store startup health check results
    cache.set('startup_health_check_results', results, 60 * 60 * 24)  # Cache for 24 hours
    
    # Log overall status
    if results["status"] == "operational":
        logger.info("Startup health check passed")
    else:
        logger.warning("Startup health check detected issues", services=results["services"])
    
    return results

# Add a WebSocket health check function
@shared_task
def websocket_health_check():
    """Check WebSocket server health"""
    from datetime import datetime
    import asyncio
    import websockets
    
    async def check_websocket():
        try:
            uri = f"ws://localhost:8000/ws/health/"
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                data = json.loads(response)
                return data.get("type") == "pong"
        except Exception as e:
            logger.error("WebSocket health check failed", error=str(e))
            return False
    
    try:
        result = asyncio.run(check_websocket())
        status = "operational" if result else "degraded"
        
        results = {
            "timestamp": str(datetime.now()),
            "service": "websocket",
            "status": status
        }
        
        cache.set('websocket_health_check_results', results, 60 * 15)  # Cache for 15 minutes
        
        if status == "operational":
            logger.info("WebSocket health check passed")
        else:
            logger.warning("WebSocket health check failed")
        
        return results
    except Exception as e:
        logger.error("WebSocket health check error", error=str(e))
        return {
            "timestamp": str(datetime.now()),
            "service": "websocket",
            "status": "error",
            "error": str(e)
        }
