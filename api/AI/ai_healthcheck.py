import os
import requests
import inspect
import pybreaker
import json
import pymed
import numpy as np
import torch
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from ..models import HealthCheckResult
from datetime import timedelta
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureChannelFirst, Resize
from celery import shared_task
from bs4 import BeautifulSoup
import structlog
from datetime import datetime
from PIL import Image
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
        self.densenet_model = None
        if settings.INITIALIZE_XRAY_MODEL:
            self._initialize_xray_model()
    
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
        """Initialize the MONAI DenseNet121 model for X-ray analysis"""
        try:
            from monai.networks.nets import DenseNet121
            self.densenet_model = DenseNet121(
                spatial_dims=2,
                in_channels=1,
                out_channels=3,  # Fracture, tumor, pneumonia
            ).to(self.device)
            
            model_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'densenet_xray.pth')
            if os.path.exists(model_path):
                self.densenet_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.densenet_model.eval()
            logger.info("DenseNet121 model initialized", device=self.device)
        except Exception as e:
            logger.error("Error initializing DenseNet121 model", error=str(e))
            self.densenet_model = None
    
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
        
        {
            "conditions": [],
            "next_steps": [],
            "urgent": [],
            "advice": []
        }
        
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
    def analyze_xray(self, image_data):
        """
        Analyze X-ray images using DenseNet121 model
        """
        if not hasattr(self, 'densenet_model') or self.densenet_model is None:
            logger.warning("DenseNet model not initialized")
            return {
                "error": "X-ray analysis model not initialized",
                "message": "The X-ray analysis model is not available."
            }
        
        try:
            # Cache key for results
            cache_key = f"xray_analysis_{hash(str(image_data))}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                logger.info("Returning cached X-ray analysis")
                return cached_result
                
            # Preprocess the image
            transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                Resize((224, 224)),
            ])
            
            # Handle different input types
            if isinstance(image_data, str):
                image = transforms(image_data)
            else:
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(image_data))
                image = np.array(image.convert('L'))  # Convert to grayscale
                image = transforms(image)
            
            # Add batch dimension
            image = image.unsqueeze(0).to(self.device)
            
            # Run inference with DenseNet model
            with torch.no_grad():
                densenet_output = self.densenet_model(image)
                densenet_probs = torch.softmax(densenet_output, dim=1)
                
                # Get probabilities
                fracture_prob = densenet_probs[0, 0].item()
                tumor_prob = densenet_probs[0, 1].item()
                pneumonia_prob = densenet_probs[0, 2].item()
                normal_prob = max(0, min(1.0, 1.0 - (fracture_prob + tumor_prob + pneumonia_prob)))
            
            # Determine findings
            findings = []
            confidence_scores = {
                "normal": round(normal_prob * 100, 2),
                "pneumonia": round(pneumonia_prob * 100, 2),
                "fracture": round(fracture_prob * 100, 2),
                "tumor": round(tumor_prob * 100, 2)
            }
            
            detected_conditions = {}
            for condition, score in confidence_scores.items():
                if score > 30:
                    confidence_level = "low"
                    if score > 70:
                        confidence_level = "high"
                    elif score > 50:
                        confidence_level = "moderate"
                    detected_conditions[condition] = confidence_level
                    findings.append(f"Potential {condition} detected with {confidence_level} confidence ({score}%)")
            
            if not detected_conditions or confidence_scores["normal"] > 60:
                findings.append("No significant abnormalities detected")
                detected_conditions["normal"] = "high" if confidence_scores["normal"] > 80 else "moderate"
            
            # Get related research
            research = self.get_medical_research("X-ray diagnosis " + " ".join(detected_conditions.keys()), limit=2)
            
            # Prepare prompt for detailed analysis
            prompt = f"""You are Praxia, a medical AI assistant specialized in X-ray analysis. {self.identity}
    
            I have analyzed an X-ray image with the following findings:
            - Detected conditions: {', '.join([f"{cond} ({level} confidence, {confidence_scores[cond]}%)" for cond, level in detected_conditions.items()])}
    
            Provide a detailed analysis in JSON format with these keys:
            1. "interpretation": Detailed interpretation of the X-ray findings
            2. "possible_conditions": List of possible conditions, ordered by likelihood
            3. "recommendations": Recommended next steps for the patient
            4. "limitations": Limitations of this AI analysis
    
            Your response must be valid JSON that can be parsed programmatically.
            """
            
            # Call Together AI API
            detailed_analysis_text = self._call_together_ai(prompt)
            
            # Extract JSON
            try:
                if "" in detailed_analysis_text and "" in detailed_analysis_text.split("", 1)[1]:
                    json_str = detailed_analysis_text.split("json", 1)[1].split("```", 1)[0].strip()
                    detailed_analysis = json.loads(json_str)
                else:
                    detailed_analysis = json.loads(detailed_analysis_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response for X-ray analysis")
                detailed_analysis = {
                    "interpretation": detailed_analysis_text,
                    "possible_conditions": list(detected_conditions.keys()),
                    "recommendations": ["Consult with a radiologist for professional interpretation"],
                    "limitations": ["AI analysis should be confirmed by a healthcare professional"]
                }
            
            result = {
                "analysis": "X-ray analysis completed successfully",
                "findings": findings,
                "confidence_scores": confidence_scores,
                "detected_conditions": detected_conditions,
                "detailed_analysis": detailed_analysis,
                "related_research": research,
                "disclaimer": "This is an AI interpretation and should be confirmed by a radiologist."
            }
            
            # Cache the result
            cache.set(cache_key, result, self.cache_timeout)
            
            logger.info("X-ray analysis completed successfully", 
                        conditions=list(detected_conditions.keys()),
                        confidence=confidence_scores)
            return result
        except Exception as e:
            logger.error("Error in X-ray analysis", error=str(e))
            return {"error": str(e), "message": "Unable to process X-ray at this time."}
    
    @shared_task
    @circuit_breaker_with_fallback(pubmed_breaker, pubmed_fallback)
    @cache_result(timeout=60*60*24, key_prefix='research')
    def get_medical_research(query, limit=5):
        """
        Retrieve relevant medical research from PubMed
        """
        try:
            # Create a PubMed client here instead of using self.pubmed_client
            pubmed_client = pymed.PubMed(tool="PraxiaAI", email="contact@praxia.ai")
        
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp])"
            results = pubmed_client.query(search_term, max_results=limit)
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
        
        {
            "balance": "",
            "deficiencies": [],
            "recommendations": [],
            "foods": {"add": [], "remove": []}
        }
        
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

@shared_task
def scheduled_health_check():
    """Scheduled health check to ensure all services are operational and gather latest data"""
    from ..circuit_breaker import check_circuit_breakers
    from .praxia_model import PraxiaAI as PraxiaModelAI  # Import the correct class
    
    # Check if we already have a recent health check (less than 6 hours old)
    six_hours_ago = timezone.now() - timedelta(hours=6)
    recent_check = HealthCheckResult.objects.filter(timestamp__gte=six_hours_ago).first()
    
    if recent_check:
        logger.info("Using recent health check", check_id=recent_check.id)
        return {
            "timestamp": str(recent_check.timestamp),
            "status": recent_check.status,
            "services": recent_check.services_status,
            "external_data": recent_check.external_data
        }
    
    results = {
        "timestamp": str(timezone.now()),
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
            results["services"]["densenet_model"] = "operational" if praxia.densenet_model else "not_loaded"
        else:
            results["services"]["densenet_model"] = "disabled"
    except Exception as e:
        results["services"]["ai_models"] = f"error: {str(e)}"
        results["status"] = "degraded"
    
    # Check Celery workers
    try:
        from celery.app.control import Inspect
        from praxia_backend.celery import app
        
        insp = Inspect(app=app)
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
    
    # Gather external data for AI context
    external_data = {}
    
    # Get latest health news
    try:
        # Use the PraxiaAI from praxia_model.py instead
        praxia = PraxiaModelAI()
        news_articles = praxia._scrape_health_news(source='all', limit=5)
        external_data["health_news"] = news_articles
    except Exception as e:
        logger.error("Failed to gather health news", error=str(e))
        external_data["health_news"] = []
    
    # Get latest medical research trends
    try:
        research_topics = ["COVID-19", "cancer treatment", "heart disease", "diabetes", "mental health"]
        research_data = {}
        
        for topic in research_topics:
            try:
                # Make sure to use the instance method on the praxia object
                research_data[topic] = praxia.get_medical_research(query=topic, limit=2)
            except Exception as e:
                logger.error("Function call failed", function="get_medical_research", error=str(e))
                logger.info("Using PubMed fallback", limit=2, query=topic)
                from ..circuit_breaker import pubmed_fallback
                research_data[topic] = pubmed_fallback(topic, 2)
            
        external_data["research_trends"] = research_data
    except Exception as e:
        logger.error("Failed to gather research trends", error=str(e))
        external_data["research_trends"] = {}
    
    # Store the results in the database
    health_check = HealthCheckResult.objects.create(
        status=results["status"],
        services_status=results["services"],
        external_data=external_data
    )
    
    # Log health check results
    if results["status"] == "operational":
        logger.info("Health check passed", services=results["services"])
    else:
        logger.warning("Health check detected issues", services=results["services"])
    
    # Return combined results
    return {
        "timestamp": str(health_check.timestamp),
        "status": health_check.status,
        "services": health_check.services_status,
        "external_data": health_check.external_data
    }

@shared_task
def startup_health_check():
    """Health check to run at startup"""
    from datetime import datetime
    
    logger.info("Running startup health check")
    
    # Check if we have a recent health check (less than 6 hours old)
    six_hours_ago = timezone.now() - timedelta(hours=6)
    recent_check = HealthCheckResult.objects.filter(timestamp__gte=six_hours_ago).first()
    
    if recent_check:
        logger.info("Using recent health check for startup", check_id=recent_check.id)
        return {
            "timestamp": str(recent_check.timestamp),
            "status": recent_check.status,
            "services": recent_check.services_status,
            "external_data": recent_check.external_data
        }
    
    # If no recent check, run a full health check
    return scheduled_health_check()

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
            "timestamp": str(timezone.now()),
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
            "timestamp": str(timezone.now()),
            "service": "websocket",
            "status": "error",
            "error": str(e)
        }

class AIHealthCheck:
    """Class for checking AI system health"""
    
    def __init__(self):
        self.praxia_ai = PraxiaAI()
    
    def run_check(self):
        """Run health check on AI system"""
        try:
            # Check Together AI API key
            if not getattr(self.praxia_ai, 'together_api_key', None):
                logger.error("Together AI API key missing")
                return False

            # Only check X-ray model if explicitly enabled
            if getattr(settings, "INITIALIZE_XRAY_MODEL", False):
                if not hasattr(self.praxia_ai, 'densenet_model') or self.praxia_ai.densenet_model is None:
                    logger.error("X-ray model required but not loaded")
                    return False

            # If we reach here, AI system is healthy for current config
            return True
        except Exception as e:
            logger.error("AI health check failed", error=str(e))
            return False
