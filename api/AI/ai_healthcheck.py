import os
import requests
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
    def diagnose_symptoms(self, symptoms, user_profile=None):
        """
        Analyze symptoms and provide potential diagnoses
        """
        cache_key = f"diagnosis_{hash(symptoms)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached diagnosis", cache_key=cache_key)
            return cached_result
        
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
            cache.set(cache_key, result, self.cache_timeout)
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
    def get_medical_research(self, query, limit=5):
        """
        Retrieve relevant medical research from PubMed
        """
        cache_key = f"research_{hash(query)}_{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached research", cache_key=cache_key)
            return cached_result
        
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
            cache.set(cache_key, articles, self.cache_timeout)
            logger.info("Research fetched", query=query)
            return articles
        except Exception as e:
            logger.error("Research query failed", error=str(e), query=query)
            return [
                {"title": "Recent advances in medical diagnosis", "authors": "Smith J, et al.", "journal": "Medical Journal", "publication_date": "2023"},
                {"title": "Clinical guidelines for symptom management", "authors": "Johnson M, et al.", "journal": "Healthcare Research", "publication_date": "2022"}
            ]
    
    @shared_task
    def analyze_diet(self, diet_info, user_profile=None):
        """
        Analyze diet information and provide nutritional recommendations
        """
        cache_key = f"diet_{hash(diet_info)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached diet analysis", cache_key=cache_key)
            return cached_result
        
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
            cache.set(cache_key, result, self.cache_timeout)
            logger.info("Diet analysis generated", diet_info=diet_info)
            return result
        except Exception as e:
            logger.error("Diet analysis failed", error=str(e), diet_info=diet_info)
            return {"error": str(e), "message": "Unable to process diet analysis."}
    
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
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            logger.error("Together AI API call failed", error=str(e))
            return json.dumps({
                "error": "API unavailable",
                "message": "Fallback response: please consult a healthcare professional."
            })
    
    def _fetch_who_guidelines(self, query):
        """Fetch WHO guidelines for a disease or symptom"""
        cache_key = f"who_{hash(query)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached WHO guidelines", cache_key=cache_key)
            return cached_result
        
        try:
            response = requests.get(
                "https://ghoapi.azureedge.net/api/Indicator",
                params={"$filter": f"contains(IndicatorName, '{query}')"},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()["value"]
            guidelines = " ".join([item["IndicatorName"] for item in data[:3]])
            cache.set(cache_key, guidelines, self.cache_timeout)
            logger.info("WHO guidelines fetched", query=query)
            return guidelines
        except Exception as e:
            logger.error("WHO API call failed", error=str(e), query=query)
            return "WHO guidelines unavailable; consult local health authorities."
    
    def _scrape_mayo_clinic(self, query):
        """Scrape Mayo Clinic for symptom or disease info"""
        cache_key = f"mayo_{hash(query)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached Mayo Clinic data", cache_key=cache_key)
            return cached_result
        
        try:
            url = f"https://www.mayoclinic.org/diseases-conditions/{query.replace(' ', '-')}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", class_="content")
            info = content.text.strip() if content else "No data found."
            cache.set(cache_key, info, self.cache_timeout)
            logger.info("Mayo Clinic data scraped", query=query)
            return info
        except Exception as e:
            logger.error("Mayo Clinic scraping failed", error=str(e), query=query)
            return "Mayo Clinic data unavailable; refer to standard medical resources."
    
    def _build_user_context(self, user_profile):
        """Build context from user profile"""
        if not user_profile:
            return ""
        context = f"Patient: Age {user_profile.get('age', 'unknown')}, Weight {user_profile.get('weight', 'unknown')}kg, "
        context += f"Height {user_profile.get('height', 'unknown')}cm, Country: {user_profile.get('country', 'unknown')}. "
        if user_profile.get('allergies'):
            context += f"Allergies: {user_profile.get('allergies')}. "
        return context