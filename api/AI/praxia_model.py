import os
import requests
import json
import pymed
import numpy as np
import torch
import structlog
from datetime import datetime
from PIL import Image
from io import BytesIO
from django.conf import settings
from bs4 import BeautifulSoup
from django.core.cache import cache
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
)
from celery import shared_task

# Configure structured logging
logger = structlog.get_logger()

@shared_task
def scrape_health_news(source='who', limit=3):
    """
    Scrape health news from WHO and other sources, then summarize with AI
    
    Args:
        source (str): Source to scrape ('who', 'cdc', 'all')
        limit (int): Maximum number of articles to return
        
    Returns:
        list: News articles with summaries
    """
    # Create an instance of PraxiaAI to use its methods
    praxia = PraxiaAI()
    
    # Generate cache key
    cache_key = f"health_news_{source}_{limit}"
    cached_result = cache.get(cache_key)
    
    if cached_result:
        logger.info("Returning cached health news")
        return cached_result
    
    try:
        articles = []
        
        # Scrape WHO news
        if source in ['who', 'all']:
            who_articles = praxia._scrape_who_news(limit=limit if source == 'who' else max(1, limit // 2))
            articles.extend(who_articles)
        
        # Scrape CDC news
        if source in ['cdc', 'all']:
            cdc_articles = praxia._scrape_cdc_news(limit=limit if source == 'cdc' else max(1, limit // 2))
            articles.extend(cdc_articles)
        
        # Limit the total number of articles
        articles = articles[:limit]
        
        # Summarize articles with AI
        for article in articles:
            if 'content' in article and article['content']:
                article['summary'] = praxia._summarize_article(article['content'])
            else:
                article['summary'] = "No content available for summarization."
        
        # Cache the results
        cache.set(cache_key, articles, 60 * 60 * 12)  # Cache for 12 hours
        
        logger.info("Health news scraped successfully", source=source, count=len(articles))
        return articles
        
    except Exception as e:
        logger.error("Error scraping health news", error=str(e), source=source)
        return [
            {
                "title": "Unable to retrieve health news at this time",
                "source": source,
                "url": "#",
                "summary": "Please try again later.",
                "published_date": datetime.now().strftime("%Y-%m-%d")
            }
        ]

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
            # Skip UNETR initialization and only use DenseNet121
            self.xray_model = None
            
            # Initialize DenseNet121 for classification
            try:
                from monai.networks.nets import DenseNet121
                self.densenet_model = DenseNet121(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=3,  # Fracture, tumor, pneumonia
                ).to(self.device)
                
                densenet_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'densenet_xray.pth')
                if os.path.exists(densenet_path):
                    self.densenet_model.load_state_dict(torch.load(densenet_path, map_location=self.device))
                    self.densenet_model.eval()
                    logger.info("DenseNet model loaded successfully")
                else:
                    logger.warning("DenseNet model weights not found")
            except Exception as e:
                logger.error("Error initializing DenseNet model", error=str(e))
                self.densenet_model = None
                
        except Exception as e:
            logger.error("Error initializing X-ray model", error=str(e))
            self.densenet_model = None
    
    def _build_user_context(self, user_profile):
        """Build context string from user profile data"""
        if not user_profile:
            return ""
        
        context = f"Patient information: "
        
        if user_profile.get('gender'):
            context += f"Gender: {user_profile.get('gender')}, "
        
        context += f"Age {user_profile.get('age', 'unknown')}, "
        context += f"Weight {user_profile.get('weight', 'unknown')}kg, "
        context += f"Height {user_profile.get('height', 'unknown')}cm, "
        context += f"Country: {user_profile.get('country', 'unknown')}. "
        
        if user_profile.get('allergies'):
            context += f"Allergies: {user_profile.get('allergies')}. "
        
        if user_profile.get('medical_history'):
            context += f"Medical history: {user_profile.get('medical_history')}. "
        
        return context
    
    def _preprocess_symptoms(self, symptoms):
        """Preprocess and validate symptom input"""
        if not symptoms or len(symptoms.strip()) < 3:
            return "unspecified symptoms"
            
        # Remove any potentially harmful characters
        cleaned = symptoms.replace('<', '').replace('>', '')
        return cleaned
    
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
        # Preprocess symptoms
        processed_symptoms = self._preprocess_symptoms(symptoms)
        
        # Generate cache key
        cache_key = f"diagnosis_{hash(processed_symptoms)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached diagnosis", symptoms=processed_symptoms[:30])
            return cached_result
        
        # Build context with user profile
        context = self._build_user_context(user_profile)
        
        # Get relevant medical research first to incorporate into prompt
        research_results = self.get_medical_research(processed_symptoms, limit=2)
        research_context = ""
        if research_results:
            research_context = "Relevant medical research:\n"
            for i, article in enumerate(research_results):
                research_context += f"{i+1}. {article.get('title')} ({article.get('journal')}): "
                research_context += f"{article.get('abstract')[:200]}...\n"
        
        # Prepare prompt for the LLM with improved structure
        prompt = f"""You are Praxia, a medical AI assistant. {self.identity}

{context}

Based on these symptoms: {processed_symptoms}

{research_context}

WHO guidelines recommend careful assessment of symptoms and considering local disease prevalence.
Mayo Clinic emphasizes that symptom diagnosis should consider patient history and risk factors.

Analyze these symptoms and provide a response in JSON format with these keys:
1. "conditions": [List of potential conditions, ordered by likelihood, with confidence scores (0-100)]
2. "next_steps": [Recommended diagnostic steps or treatments]
3. "urgent": [Symptoms or conditions requiring immediate medical attention]
4. "advice": General advice for managing these symptoms
5. "clarification": Questions to ask if symptoms are ambiguous or incomplete

If symptoms are ambiguous, focus on the "clarification" section to gather more information.

Your response must be valid JSON that can be parsed programmatically.
"""
        
        try:
            # Call Together AI API
            response_text = self._call_together_ai(prompt)
            
            # Extract JSON from response
            try:
                # Find JSON content between triple backticks if present
                if "" in response_text and "" in response_text.split("", 1)[1]:
                    json_str = response_text.split("json", 1)[1].split("", 1)[0].strip()
                    diagnosis_data = json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    diagnosis_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: create structured response from unstructured text
                logger.warning("Failed to parse JSON response", response=response_text[:100])
                diagnosis_data = {
                    "conditions": ["Unable to parse conditions from response"],
                    "next_steps": ["Consult with a healthcare professional"],
                    "urgent": ["If symptoms are severe, seek immediate medical attention"],
                    "advice": response_text,
                    "clarification": ["Please provide more specific symptom details"]
                }
            
            # Add research results and disclaimer
            result = {
                "diagnosis": diagnosis_data,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            }
            
            # Cache the result
            cache.set(cache_key, result, self.cache_timeout)
            logger.info("Diagnosis completed successfully", symptoms=processed_symptoms[:30])
            
            return result
        except Exception as e:
            logger.error("Error in symptom diagnosis", error=str(e), symptoms=processed_symptoms[:30])
            return {
                "error": str(e), 
                "message": "Unable to process diagnosis at this time.",
                "disclaimer": "Please consult with a healthcare professional for medical advice."
            }
    
    @shared_task
    def analyze_xray(self, image_data):
        """
        Analyze X-ray images using DenseNet121 model
        
        Args:
            image_data: The X-ray image data (file path or bytes)
            
        Returns:
            dict: Analysis results with condition detection
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
                # If image_data is a file path
                image = transforms(image_data)
            else:
                # If image_data is bytes
                image = Image.open(BytesIO(image_data))
                image = np.array(image.convert('L'))  # Convert to grayscale
                image = transforms(image)
            
            # Add batch dimension
            image = image.unsqueeze(0).to(self.device)
            
            # Run inference with DenseNet model
            with torch.no_grad():
                densenet_output = self.densenet_model(image)
                densenet_probs = torch.softmax(densenet_output, dim=1)
                
                # Get probabilities from DenseNet
                fracture_prob = densenet_probs[0, 0].item()
                tumor_prob = densenet_probs[0, 1].item()
                pneumonia_prob = densenet_probs[0, 2].item()
                
                # Calculate normal probability (inverse of the sum of other probabilities)
                # Capped at 1.0 to ensure it's a valid probability
                normal_prob = max(0, min(1.0, 1.0 - (fracture_prob + tumor_prob + pneumonia_prob)))
            
            # Determine findings based on probabilities
            findings = []
            confidence_scores = {
                "normal": round(normal_prob * 100, 2),
                "pneumonia": round(pneumonia_prob * 100, 2),
                "fracture": round(fracture_prob * 100, 2),
                "tumor": round(tumor_prob * 100, 2)
            }
            
            # Determine detected conditions (those with >30% confidence)
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
            
            # Get related research for context
            research = self.get_medical_research("X-ray diagnosis " + " ".join(detected_conditions.keys()), limit=2)
            
            # Prepare prompt for detailed analysis
            prompt = f"""You are Praxia, a medical AI assistant specialized in X-ray analysis. {self.identity}

I have analyzed an X-ray image with the following findings:
- Detected conditions: {', '.join([f"{cond} ({level} confidence, {confidence_scores[cond]}%)" for cond, level in detected_conditions.items()])}

Based on these findings, provide a detailed analysis in JSON format with these keys:
1. "interpretation": Detailed interpretation of the X-ray findings
2. "possible_conditions": List of possible conditions, ordered by likelihood
3. "recommendations": Recommended next steps for the patient
4. "limitations": Limitations of this AI analysis

Your response must be valid JSON that can be parsed programmatically.
"""
            
            # Call Together AI API for detailed analysis
            detailed_analysis_text = self._call_together_ai(prompt)
            
            # Extract JSON from response
            try:
                # Find JSON content between triple backticks if present
                if "" in detailed_analysis_text and "" in detailed_analysis_text.split("", 1)[1]:
                    json_str = detailed_analysis_text.split("json", 1)[1].split("```", 1)[0].strip()
                    detailed_analysis = json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    detailed_analysis = json.loads(detailed_analysis_text)
            except json.JSONDecodeError:
                # Fallback: create structured response
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
            logger.info("Medical research retrieved successfully", query=query, count=len(articles))
            
            return articles
        except Exception as e:
            logger.warning("Error retrieving medical research", error=str(e), query=query)
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
        # Preprocess input
        if not diet_info or len(diet_info.strip()) < 5:
            return {
                "error": "Insufficient diet information",
                "message": "Please provide more details about your diet for analysis."
            }
            
        # Generate cache key
        cache_key = f"diet_{hash(diet_info)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached diet analysis")
            return cached_result
        
        # Build context with user profile
        context = self._build_user_context(user_profile)
        
        # Get relevant nutritional research
        research_results = self.get_medical_research(f"nutrition {diet_info}", limit=2)
        research_context = ""
        if research_results:
            research_context = "Relevant nutritional research:\n"
            for i, article in enumerate(research_results):
                research_context += f"{i+1}. {article.get('title')} ({article.get('journal')}): "
                research_context += f"{article.get('abstract')[:200]}...\n"
        
        # Prepare prompt for the LLM
        prompt = f"""You are Praxia, a medical AI assistant specializing in nutrition. {self.identity}

{context}

Based on this diet information: {diet_info}

{research_context}

Analyze this diet and provide a response in JSON format with these keys:
1. "assessment": Overall assessment of the diet's nutritional value
2. "deficiencies": [Potential nutritional deficiencies with confidence scores (0-100)]
3. "recommendations": [Specific foods or supplements to consider adding]
4. "concerns": [Potential health concerns related to this diet]
5. "positives": [Positive aspects of the current diet]

Your response must be valid JSON that can be parsed programmatically.
"""
        
        try:
            # Call Together AI API
            response_text = self._call_together_ai(prompt)
            
            # Extract JSON from response
            try:
                # Find JSON content between triple backticks if present
                if "" in response_text and "" in response_text.split("", 1)[1]:
                    json_str = response_text.split("json", 1)[1].split("", 1)[0].strip()
                    diet_data = json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    diet_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: create structured response from unstructured text
                logger.warning("Failed to parse JSON response for diet analysis", response=response_text[:100])
                diet_data = {
                    "assessment": "Unable to parse structured assessment from response",
                    "deficiencies": ["Unable to parse deficiencies from response"],
                    "recommendations": ["Consult with a nutritionist for personalized advice"],
                    "concerns": ["Unable to determine concerns from the provided information"],
                    "positives": ["Unable to determine positive aspects from the provided information"]
                }
            
            # Add research results and disclaimer
            result = {
                "analysis": diet_data,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only and not a substitute for professional nutritional advice."
            }
            
            # Cache the result
            cache.set(cache_key, result, self.cache_timeout)
            logger.info("Diet analysis completed successfully")
            
            return result
        except Exception as e:
            logger.error("Error in diet analysis", error=str(e))
            return {
                "error": str(e), 
                "message": "Unable to process diet analysis at this time.",
                "disclaimer": "Please consult with a nutritionist for professional advice."
            }
    
    @shared_task
    def analyze_medication(self, medication_info, user_profile=None):
        """
        Analyze medication information and provide insights
        
        Args:
            medication_info (str): User's medication information
            user_profile (dict, optional): User profile data for personalized analysis
            
        Returns:
            dict: Medication analysis results
        """
        # Preprocess input
        if not medication_info or len(medication_info.strip()) < 3:
            return {
                "error": "Insufficient medication information",
                "message": "Please provide more details about your medications for analysis."
            }
            
        # Generate cache key
        cache_key = f"medication_{hash(medication_info)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached medication analysis")
            return cached_result
        
        # Build context with user profile
        context = self._build_user_context(user_profile)
        
        # Get relevant research
        research_results = self.get_medical_research(f"medication {medication_info} interactions", limit=2)
        research_context = ""
        if research_results:
            research_context = "Relevant medication research:\n"
            for i, article in enumerate(research_results):
                research_context += f"{i+1}. {article.get('title')} ({article.get('journal')}): "
                research_context += f"{article.get('abstract')[:200]}...\n"
        
        # Prepare prompt for the LLM
        prompt = f"""You are Praxia, a medical AI assistant specializing in pharmacology. {self.identity}

{context}

Based on this medication information: {medication_info}

{research_context}

Analyze these medications and provide a response in JSON format with these keys:
1. "overview": General information about the medications mentioned
2. "interactions": [Potential interactions between medications, with severity levels]
3. "side_effects": [Common side effects to be aware of]
4. "precautions": [Important precautions when taking these medications]
5. "questions": [Questions the patient should ask their healthcare provider]

Your response must be valid JSON that can be parsed programmatically.
"""
        
        try:
            # Call Together AI API
            response_text = self._call_together_ai(prompt)
            
            # Extract JSON from response
            try:
                # Find JSON content between triple backticks if present
                if "json" in response_text and "" in response_text.split("json", 1)[1]:
                    json_str = response_text.split("", 1)[1].split("", 1)[0].strip()
                    medication_data = json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    medication_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: create structured response from unstructured text
                logger.warning("Failed to parse JSON response for medication analysis", response=response_text[:100])
                medication_data = {
                    "overview": response_text,
                    "interactions": ["Unable to parse interactions from response"],
                    "side_effects": ["Consult with a healthcare provider for side effect information"],
                    "precautions": ["Consult with a healthcare provider before changing any medication regimen"],
                    "questions": ["Ask your doctor about potential interactions with your current medications"]
                }
            
            # Add research results and disclaimer
            result = {
                "analysis": medication_data,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only. Always consult with a healthcare provider before making any changes to your medication regimen."
            }
            
            # Cache the result
            cache.set(cache_key, result, self.cache_timeout)
            logger.info("Medication analysis completed successfully")
            
            return result
        except Exception as e:
            logger.error("Error in medication analysis", error=str(e))
            return {
                "error": str(e), 
                "message": "Unable to process medication analysis at this time.",
                "disclaimer": "Please consult with a healthcare provider for professional advice."
            }
    
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
            response.raise_for_status()  # Raise exception for HTTP errors
            
            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error("Error calling Together AI API", error=str(e))
            raise Exception(f"API request failed: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Error parsing API response", error=str(e))
            raise Exception(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error in API call", error=str(e))
            raise Exception(f"Unexpected error: {str(e)}")
    
    def _scrape_health_news(self, source='who', limit=3):
        """
        Scrape health news from WHO and other sources, then summarize with AI
        
        Args:
            source (str): Source to scrape ('who', 'cdc', 'all')
            limit (int): Maximum number of articles to return
            
        Returns:
            list: News articles with summaries
        """
        # Generate cache key
        cache_key = f"health_news_{source}_{limit}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached health news")
            return cached_result
        
        try:
            articles = []
            
            # Scrape WHO news
            if source in ['who', 'all']:
                who_articles = self._scrape_who_news(limit=limit if source == 'who' else max(1, limit // 2))
                articles.extend(who_articles)
            
            # Scrape CDC news
            if source in ['cdc', 'all']:
                cdc_articles = self._scrape_cdc_news(limit=limit if source == 'cdc' else max(1, limit // 2))
                articles.extend(cdc_articles)
            
            # Limit the total number of articles
            articles = articles[:limit]
            
            # Summarize articles with AI
            for article in articles:
                if 'content' in article and article['content']:
                    article['summary'] = self._summarize_article(article['content'])
                else:
                    article['summary'] = "No content available for summarization."
            
            # Cache the results
            cache.set(cache_key, articles, 60 * 60 * 12)  # Cache for 12 hours
            
            logger.info("Health news scraped successfully", source=source, count=len(articles))
            return articles
            
        except Exception as e:
            logger.error("Error scraping health news", error=str(e), source=source)
            return [
                {
                    "title": "Unable to retrieve health news at this time",
                    "source": source,
                    "url": "#",
                    "summary": "Please try again later.",
                    "published_date": datetime.now().strftime("%Y-%m-%d")
                }
            ]
    
    def _scrape_who_news(self, limit=3):
        """Scrape news from WHO website"""
        try:
            url = "https://www.who.int/news"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.select('.list-view--item')
            
            articles = []
            for item in news_items[:limit]:
                try:
                    # Extract title and URL
                    title_elem = item.select_one('.heading')
                    title = title_elem.text.strip() if title_elem else "No title"
                    
                    link_elem = item.select_one('a')
                    url = "https://www.who.int" + link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
                    
                    # Extract date
                    date_elem = item.select_one('.timestamp')
                    published_date = date_elem.text.strip() if date_elem else None
                    
                    # Extract image
                    img_elem = item.select_one('img')
                    image_url = img_elem['src'] if img_elem and 'src' in img_elem.attrs else None
                    
                    # Get article content
                    content = self._get_article_content(url)
                    
                    articles.append({
                        "title": title,
                        "source": "WHO",
                        "url": url,
                        "content": content,
                        "image_url": image_url,
                        "published_date": published_date
                    })
                except Exception as e:
                    logger.warning(f"Error processing WHO news item: {str(e)}")
                    continue
            
            return articles
        except Exception as e:
            logger.error(f"Error scraping WHO news: {str(e)}")
            return []
    
    def _scrape_cdc_news(self, limit=3):
        """Scrape news from CDC website"""
        try:
            url = "https://www.cdc.gov/media/index.html"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.select('.feed-item')
            
            articles = []
            for item in news_items[:limit]:
                try:
                    # Extract title and URL
                    title_elem = item.select_one('a')
                    title = title_elem.text.strip() if title_elem else "No title"
                    
                    url = "https://www.cdc.gov" + title_elem['href'] if title_elem and 'href' in title_elem.attrs else "#"
                    
                    # Extract date
                    date_elem = item.select_one('.date')
                    published_date = date_elem.text.strip() if date_elem else None
                    
                    # Get article content
                    content = self._get_article_content(url)
                    
                    articles.append({
                        "title": title,
                        "source": "CDC",
                        "url": url,
                        "content": content,
                        "image_url": None,
                        "published_date": published_date
                    })
                except Exception as e:
                    logger.warning(f"Error processing CDC news item: {str(e)}")
                    continue
            
            return articles
        except Exception as e:
            logger.error(f"Error scraping CDC news: {str(e)}")
            return []
    
    def _get_article_content(self, url):
        """Get the content of an article from its URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try different content selectors (sites have different structures)
            content_selectors = [
                'article', '.content', '.main-content', 
                '#content', '.article-body', '.story-body'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem.select('script, style'):
                        script.extract()
                    
                    # Get text and clean it
                    content = content_elem.get_text(separator=' ', strip=True)
                    content = ' '.join(content.split())  # Normalize whitespace
                    
                    # Limit content length
                    if len(content) > 5000:
                        content = content[:5000] + "..."
                    
                    return content
            
            return "Content could not be extracted."
        except Exception as e:
            logger.warning(f"Error getting article content from {url}: {str(e)}")
            return "Content could not be retrieved."
    
    def _summarize_article(self, content, max_length=200):
        """
        Summarize article content using distilbart
        
        Args:
            content (str): Article content to summarize
            max_length (int): Maximum length of summary
            
        Returns:
            str: Summarized content
        """
        try:
            # For longer articles, use AI summarization
            if len(content) > 500:
                # Use Together AI for summarization
                prompt = f"""Summarize the following health news article in 3-4 sentences:

{content[:4000]}

Summary:"""
                
                summary = self._call_together_ai(prompt)
                
                # Clean up the summary
                summary = summary.strip()
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
                
                return summary
            else:
                # For short content, just return the first part
                return content[:max_length] + ("..." if len(content) > max_length else "")
                
        except Exception as e:
            logger.error(f"Error summarizing article: {str(e)}")
            return content[:max_length] + "..."
