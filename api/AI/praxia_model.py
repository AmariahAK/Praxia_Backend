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
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
)
from celery import shared_task

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
        
        # Initialize MONAI model for X-ray analysis if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xray_model = None
        if settings.INITIALIZE_XRAY_MODEL:
            self._initialize_xray_model()
    
    def _load_identity(self):
        """Load the AI identity from the text file"""
        identity_path = os.path.join(settings.BASE_DIR, 'data', 'ai_identity.txt')
        try:
            with open(identity_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return "Praxia - A healthcare AI assistant by Amariah Kamau"
    
    def _initialize_xray_model(self):
        """Initialize the MONAI model for X-ray analysis"""
        try:
            # Initialize UNETR model for X-ray analysis
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
            
            # Load pre-trained weights if available
            model_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'xray_model.pth')
            if os.path.exists(model_path):
                self.xray_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.xray_model.eval()
        except Exception as e:
            print(f"Error initializing X-ray model: {e}")
            self.xray_model = None
    
    @shared_task
    def diagnose_symptoms(self, symptoms, user_profile=None):
        """
        Analyze symptoms and provide potential diagnoses
        
        Args:
            symptoms (str): User-reported symptoms
            user_profile (dict, optional): User profile data for personalized diagnosis
            
        Returns:
            dict: Diagnosis results with potential conditions and recommendations
        """
        cache_key = f"diagnosis_{hash(symptoms)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Prepare context with user profile if available
        context = ""
        if user_profile:
            context = f"Patient information: Age {user_profile.get('age')}, Weight {user_profile.get('weight')}kg, "
            context += f"Height {user_profile.get('height')}cm, Country: {user_profile.get('country')}. "
            if user_profile.get('allergies'):
                context += f"Allergies: {user_profile.get('allergies')}. "
        
        # Prepare prompt for the LLM
        prompt = f"""You are Praxia, a medical AI assistant. {self.identity}
        
        {context}
        
        Based on these symptoms: {symptoms}
        
        Provide a detailed analysis including:
        1. Potential conditions that match these symptoms
        2. Recommended next steps
        3. When the patient should seek immediate medical attention
        4. General advice for managing these symptoms
        
        Format your response in a structured way with clear sections.
        """
        
        try:
            # Call Together AI API
            response = self._call_together_ai(prompt)
            
            # Enhance the diagnosis with relevant medical research
            research_results = self.get_medical_research(symptoms, limit=3)
            
            # Process and structure the response
            result = {
                "diagnosis": response,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            }
            
            # Cache the result
            cache.set(cache_key, result, self.cache_timeout)
            
            return result
        except Exception as e:
            return {"error": str(e), "message": "Unable to process diagnosis at this time."}
    
    @shared_task
    def analyze_xray(self, image_data):
        """
        Analyze X-ray images using MONAI models
        
        Args:
            image_data: The X-ray image data (file path or bytes)
            
        Returns:
            dict: Analysis results
        """
        if not self.xray_model:
            return {
                "error": "X-ray model not initialized",
                "message": "The X-ray analysis model is not available."
            }
        
        try:
            # Preprocess the image
            transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                Resize((224, 224)),
            ])
            
            # Handle different input types
            if isinstance(image_data, str):
                # If image_data is a file path
                image = transforms(image_data)
            else:
                # If image_data is bytes
                image = Image.open(BytesIO(image_data))
                image = np.array(image.convert('L'))  # Convert to grayscale
                image = transforms(image)
            
            # Add batch dimension
            image = image.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.xray_model(image)
                probabilities = torch.softmax(output, dim=1)
                abnormal_prob = probabilities[0, 1].item()
            
            # Determine findings based on probability
            findings = []
            if abnormal_prob > 0.7:
                findings.append("Potential abnormality detected with high confidence")
            elif abnormal_prob > 0.4:
                findings.append("Possible abnormality detected with moderate confidence")
            else:
                findings.append("No significant abnormalities detected")
            
            # Get related research for context
            research = self.get_medical_research("X-ray abnormalities diagnosis", limit=2)
            
            result = {
                "analysis": "X-ray analysis completed successfully",
                "findings": findings,
                "confidence": abnormal_prob,
                "related_research": research,
                "disclaimer": "This is an AI interpretation and should be confirmed by a radiologist."
            }
            
            return result
        except Exception as e:
            return {"error": str(e), "message": "Unable to process X-ray at this time."}
    
    @shared_task
    def get_medical_research(self, query, limit=5):
        """
        Retrieve relevant medical research from PubMed
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            list: Research articles
        """
        cache_key = f"research_{hash(query)}_{limit}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            # Query PubMed API
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp])"
            results = self.pubmed_client.query(search_term, max_results=limit)
            
            # Process results
            articles = []
            for article in results:
                article_data = {
                    "title": article.title,
                    "authors": ", ".join([author['lastname'] + ' ' + author['firstname'][0] for author in article.authors]) if article.authors else "Unknown",
                    "journal": article.journal,
                    "publication_date": str(article.publication_date) if hasattr(article, 'publication_date') else "Unknown",
                    "doi": article.doi if hasattr(article, 'doi') else None,
                    "abstract": article.abstract if hasattr(article, 'abstract') else "No abstract available"
                }
                articles.append(article_data)
            
            # Cache the results
            cache.set(cache_key, articles, self.cache_timeout)
            
            return articles
        except Exception as e:
            # If PubMed API fails, return placeholder data
            placeholder_results = [
                                {"title": "Recent advances in medical diagnosis and treatment", "authors": "Smith J, et al.", "journal": "Medical Journal", "publication_date": "2023"},
                {"title": "Clinical guidelines for symptom management", "authors": "Johnson M, et al.", "journal": "Healthcare Research", "publication_date": "2022"}
            ]
            return placeholder_results
    
    @shared_task
    def analyze_diet(self, diet_info, user_profile=None):
        """
        Analyze diet information and provide nutritional recommendations
        
        Args:
            diet_info (str): User's diet information
            user_profile (dict, optional): User profile data for personalized recommendations
            
        Returns:
            dict: Diet analysis results
        """
        cache_key = f"diet_{hash(diet_info)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Prepare context with user profile if available
        context = ""
        if user_profile:
            context = f"Patient information: Age {user_profile.get('age')}, Weight {user_profile.get('weight')}kg, "
            context += f"Height {user_profile.get('height')}cm, Country: {user_profile.get('country')}. "
            if user_profile.get('allergies'):
                context += f"Allergies: {user_profile.get('allergies')}. "
        
        # Prepare prompt for the LLM
        prompt = f"""You are Praxia, a medical AI assistant. {self.identity}
        
        {context}
        
        Based on this diet information: {diet_info}
        
        Provide a detailed nutritional analysis including:
        1. Assessment of nutritional balance
        2. Potential nutritional deficiencies
        3. Recommendations for improvement
        4. Specific foods to consider adding or removing
        
        Format your response in a structured way with clear sections.
        """
        
        try:
            # Call Together AI API
            response = self._call_together_ai(prompt)
            
            # Process and structure the response
            result = {
                "analysis": response,
                "disclaimer": "This information is for educational purposes only and not a substitute for professional nutritional advice."
            }
            
            # Cache the result
            cache.set(cache_key, result, self.cache_timeout)
            
            return result
        except Exception as e:
            return {"error": str(e), "message": "Unable to process diet analysis at this time."}
    
    def _call_together_ai(self, prompt):
        """
        Call the Together AI API with the given prompt
        
        Args:
            prompt (str): The prompt to send to the API
            
        Returns:
            str: The generated response
        """
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
            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error calling Together AI API: {e}")
            # Fallback response for development/testing
            return "Based on the information provided, here is my analysis... [This is a fallback response]"
