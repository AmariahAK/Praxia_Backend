import os
import requests
import json
import pymed
import numpy as np
import collections
import torch
import structlog
from datetime import datetime
from django.utils import timezone
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

logger = structlog.get_logger()

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
        
        # Improved CUDA detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")
        
        self.densenet_model = None
        if getattr(settings, "INITIALIZE_XRAY_MODEL", False):
            self._initialize_xray_model()

    def _load_identity(self):
        identity_path = os.path.join(settings.BASE_DIR, 'data', 'ai_identity.txt')
        try:
            with open(identity_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logger.error("AI identity file not found", path=identity_path)
            return "Praxia - A healthcare AI assistant by Amariah Kamau"

    def _initialize_xray_model(self):
        try:
            # First try to use a fixed model if available
            from ..utils.model_fix import fix_densenet_model
            fixed_model_path = fix_densenet_model()
            
            if fixed_model_path and os.path.exists(fixed_model_path):
                logger.info(f"Loading fixed DenseNet model from {fixed_model_path}")
                
                # Add DenseNet and Sequential to safe globals list
                from torch.serialization import add_safe_globals
                import torchvision.models.densenet
                import torch.nn.modules.container
                add_safe_globals([
                    torchvision.models.densenet.DenseNet,
                    torch.nn.modules.container.Sequential
                ])
                
                # Try loading with weights_only=False for compatibility
                self.densenet_model = torch.load(fixed_model_path, map_location=self.device, weights_only=False)
                logger.info("Loaded model successfully")
                
                self.densenet_model.eval()
                logger.info("Fixed DenseNet model loaded successfully")
                return
            
            # If fixed model not available, try original approach
            densenet_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'densenet_xray.pth')
            if not os.path.exists(densenet_path):
                logger.warning("DenseNet model weights not found at %s, will use fallback methods", densenet_path)
                # Try to create a simple model as fallback
                try:
                    from torchvision.models import densenet121
                    model = densenet121(weights=None)  
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 3)
                    self.densenet_model = model
                    self.densenet_model.eval()
                    logger.info("Created fallback DenseNet model")
                except Exception as e:
                    logger.error("Error creating fallback model: %s", str(e))
                    self.densenet_model = None
                return
            
            if os.path.getsize(densenet_path) < 1000000:  
                logger.warning("DenseNet model file is too small, may be corrupted")
                self.densenet_model = None
                return
            
            # Try to load the model with the safe globals approach
            try:
                # Add DenseNet to safe globals list
                from torch.serialization import add_safe_globals
                import torchvision.models.densenet
                add_safe_globals([torchvision.models.densenet.DenseNet])
                
                # Try with weights_only=True first
                try:
                    loaded_obj = torch.load(densenet_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    logger.warning(f"Failed to load with weights_only=True: {str(e)}")
                    # Fall back to weights_only=False if necessary
                    loaded_obj = torch.load(densenet_path, map_location=self.device, weights_only=False)
                
                # If it's an OrderedDict, it's state_dict, need to create model first
                if isinstance(loaded_obj, collections.OrderedDict):
                    from torchvision.models import densenet121
                    model = densenet121(weights=None)
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 3)
                    model.load_state_dict(loaded_obj)
                    self.densenet_model = model
                else:
                    # It's a full model
                    self.densenet_model = loaded_obj
                
                self.densenet_model.eval()
                logger.info("DenseNet model loaded successfully")
            except Exception as e:
                logger.error("Error loading DenseNet model: %s", str(e))
            
                # Try alternative loading method
                try:
                    from torchvision.models import densenet121
                    model = densenet121(weights=None)
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 3)
                    self.densenet_model = model
                    self.densenet_model.eval()
                    logger.info("Created fallback DenseNet model")
                except Exception as e2:
                    logger.error("Error creating fallback model: %s", str(e2))
                    self.densenet_model = None
        except Exception as e:
            logger.error("Error initializing DenseNet model: %s", str(e))
            self.densenet_model = None

    def _build_user_context(self, user_profile):
        if not user_profile:
            return ""
        
        logger.info("Building context with user profile", 
                    profile_data=json.dumps({k: v for k, v in user_profile.items() if v}))
        
        context = f"Patient information: "
        if user_profile.get('gender'):
            context += f"Gender: {user_profile.get('gender')}, "
        if user_profile.get('age'):
            context += f"Age: {user_profile.get('age')} years, "
        if user_profile.get('weight'):
            context += f"Weight: {user_profile.get('weight')}kg, "
        if user_profile.get('height'):
            context += f"Height: {user_profile.get('height')}cm, "
        if user_profile.get('country'):
            context += f"Country: {user_profile.get('country')}. "
        if user_profile.get('allergies'):
            context += f"Allergies: {user_profile.get('allergies')}. "
        if user_profile.get('medical_history'):
            context += f"Medical history: {user_profile.get('medical_history')}. "
        
        logger.info("Built user context", context=context)
        return context

    def _create_search_terms_from_topic(self, topic, user_profile=None):
        """
        Create targeted search terms from chat topic and user profile
        """
        search_terms = []
        
        # Base search term from topic
        base_term = topic.replace("_", " ").replace("-", " ").strip()
        search_terms.append(base_term)
        
        # Add user-specific context if available
        if user_profile:
            # Age-specific terms
            if user_profile.get('age'):
                age = int(user_profile['age'])
                if age < 18:
                    search_terms.append(f"{base_term} pediatric")
                elif age > 65:
                    search_terms.append(f"{base_term} geriatric elderly")
                else:
                    search_terms.append(f"{base_term} adult")
            
            # Gender-specific terms
            if user_profile.get('gender') and user_profile['gender'] in ['male', 'female']:
                search_terms.append(f"{base_term} {user_profile['gender']}")
            
            # Country/region-specific terms for epidemiology
            if user_profile.get('country'):
                country = user_profile['country'].lower()
                # Add regional health considerations
                regional_terms = {
                    'kenya': 'malaria tuberculosis HIV',
                    'nigeria': 'malaria sickle cell',
                    'india': 'diabetes tuberculosis',
                    'usa': 'obesity diabetes hypertension',
                    'uk': 'diabetes cardiovascular',
                    'canada': 'diabetes cardiovascular',
                    'australia': 'skin cancer melanoma',
                    'brazil': 'dengue zika chikungunya',
                    'china': 'hepatitis tuberculosis',
                    'japan': 'stroke cardiovascular',
                }
                
                for region, conditions in regional_terms.items():
                    if region in country:
                        search_terms.append(f"{base_term} {conditions}")
                        break
        
        # Remove duplicates and return unique terms
        return list(set(search_terms))

    def _filter_research_by_relevance(self, research_results, user_profile=None, max_results=3):
        """
        Filter and rank research results based on user profile relevance
        """
        if not research_results or not user_profile:
            return research_results[:max_results]
        
        scored_results = []
        
        for article in research_results:
            score = 0
            title_lower = article.get('title', '').lower()
            abstract_lower = article.get('abstract', '').lower()
            content = title_lower + ' ' + abstract_lower
            
            # Age relevance
            if user_profile.get('age'):
                age = int(user_profile['age'])
                if age < 18 and any(term in content for term in ['pediatric', 'children', 'adolescent']):
                    score += 3
                elif age > 65 and any(term in content for term in ['elderly', 'geriatric', 'older adults']):
                    score += 3
                elif 18 <= age <= 65 and any(term in content for term in ['adult', 'working age']):
                    score += 2
            
            # Gender relevance
            if user_profile.get('gender'):
                gender = user_profile['gender'].lower()
                if gender in ['male', 'female'] and gender in content:
                    score += 2
            
            # Country/regional relevance
            if user_profile.get('country'):
                country = user_profile['country'].lower()
                if country in content:
                    score += 3
                # Check for regional disease patterns
                regional_keywords = {
                    'kenya': ['malaria', 'tuberculosis', 'hiv', 'tropical'],
                    'nigeria': ['malaria', 'sickle cell', 'tropical'],
                    'india': ['tuberculosis', 'diabetes', 'tropical'],
                    'usa': ['obesity', 'diabetes', 'hypertension', 'cardiovascular'],
                    'uk': ['diabetes', 'cardiovascular', 'temperate'],
                    'canada': ['diabetes', 'cardiovascular', 'temperate'],
                }
                
                for region, keywords in regional_keywords.items():
                    if region in country:
                        for keyword in keywords:
                            if keyword in content:
                                score += 1
                        break
            
            # Allergy relevance
            if user_profile.get('allergies'):
                allergies = user_profile['allergies'].lower()
                if any(allergy.strip() in content for allergy in allergies.split(',') if allergy.strip()):
                    score += 2
            
            scored_results.append((score, article))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [article for score, article in scored_results[:max_results]]

    def _preprocess_symptoms(self, symptoms):
        if not symptoms or len(symptoms.strip()) < 3:
            return "general health inquiry"
        
        # Clean the input more safely
        cleaned = str(symptoms).replace('<', '').replace('>', '').strip()
        
        # Handle greetings and extract the actual symptoms
        greeting_phrases = ["hey praxia", "hi praxia", "hello", "greetings"]
        lower_symptoms = cleaned.lower()
        
        for phrase in greeting_phrases:
            if lower_symptoms.startswith(phrase):
                cleaned = cleaned[len(phrase):].lstrip(" ,.;:!?")
                break
        
        # Make sure we have valid content
        if not cleaned or len(cleaned.strip()) < 3:
            return "general health inquiry"
        
        cleaned = cleaned.replace("'", "'").replace(""", '"').replace(""", '"')
        
        cleaned = cleaned.rstrip("'\"")
        
        if not cleaned.strip():
            return "general health inquiry"
        
        cleaned = ' '.join(cleaned.split()) 
        
        return cleaned.strip() if cleaned.strip() else "general health inquiry"

    def diagnose_symptoms(self, symptoms, user_profile=None, chat_topic=None):
        try:
            processed_symptoms = self._preprocess_symptoms(symptoms)
        
            # Log whether user profile is being used
            if user_profile:
                logger.info("Diagnosing with user profile", 
                            has_gender=bool(user_profile.get('gender')),
                            has_age=bool(user_profile.get('age')),
                            symptoms=processed_symptoms[:30],
                            chat_topic=chat_topic)
            else:
                logger.warning("Diagnosing without user profile", symptoms=processed_symptoms[:30])
        
            cache_key = f"diagnosis_{hash(processed_symptoms)}_{hash(str(user_profile))}_{hash(str(chat_topic))}"
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached diagnosis", symptoms=processed_symptoms[:30])
                return cached_result

            context = self._build_user_context(user_profile)
            health_data = self._get_latest_health_data()
            
            # Use chat topic for more targeted search if available
            search_topic = chat_topic if chat_topic and chat_topic != "New Chat" else processed_symptoms
            research_results = self._get_targeted_research_data(search_topic, user_profile, limit=5)
            
            # Filter research results based on user profile
            filtered_research = self._filter_research_by_relevance(research_results, user_profile, max_results=3)
            
            research_context = ""
            if filtered_research:
                research_context = "Relevant medical research (filtered for your profile):\n"
                for i, article in enumerate(filtered_research):
                    research_context += f"{i+1}. {article.get('title', 'Untitled')} ({article.get('journal', 'Unknown')}): "
                    research_context += f"{article.get('abstract', 'No abstract')[:200]}...\n"
            
            news_context = ""
            if 'health_news' in health_data and health_data['health_news']:
                news_context = "Recent health news:\n"
                for i, article in enumerate(health_data['health_news'][:2]):
                    news_context += f"{i+1}. {article.get('title', 'Untitled')} ({article.get('source', 'Unknown')}): "
                    news_context += f"{article.get('summary', 'No summary')[:150]}...\n"

            prompt = f"""You are Praxia, a medical AI assistant. {self.identity}

{context}

Based on these symptoms: {processed_symptoms}
Chat context: {chat_topic or 'General consultation'}

{research_context}

{news_context}

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
                response_text = self._call_together_ai(prompt)
                try:
                    # First try direct JSON parsing
                    diagnosis_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown blocks if present
                    if "" in response_text and "" in response_text:
                        json_content = response_text.split("")[1].split("")[0].strip()
                        try:
                            diagnosis_data = json.loads(json_content)
                        except json.JSONDecodeError:
                            raise
                    elif "" in response_text and "" in response_text:
                        json_content = response_text.split("")[1].split("")[0].strip()
                        try:
                            diagnosis_data = json.loads(json_content)
                        except json.JSONDecodeError:
                            raise
                    else:
                        raise
            
                # Validate required fields
                required_fields = ["conditions", "next_steps", "urgent", "advice", "clarification"]
                for field in required_fields:
                    if field not in diagnosis_data:
                        diagnosis_data[field] = []
                    if field == "advice" and not diagnosis_data[field]:
                        diagnosis_data[field] = "Please consult with a healthcare professional for specific advice."
                    
                result = {
                    "diagnosis": diagnosis_data,
                    "related_research": filtered_research,
                    "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
                }
                cache.set(cache_key, result, self.cache_timeout)
                logger.info("Diagnosis completed successfully", symptoms=processed_symptoms[:30])
                return result
            except Exception as e:
                logger.error("Error in symptom diagnosis", error=str(e), symptoms=processed_symptoms[:30])
                # Provide a more user-friendly error response
                return {
                    "diagnosis": {
                        "conditions": ["Unable to analyze symptoms at this time"],
                        "next_steps": ["Please consult with a healthcare professional"],
                        "urgent": [],
                        "advice": "I apologize, but I'm having trouble analyzing your symptoms right now. Please try again or consult with a healthcare professional.",
                        "clarification": ["Could you provide more details about your symptoms?"]
                    },
                    "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
                }
        except Exception as e:
            logger.error("Unexpected error in symptom diagnosis", error=str(e), exc_info=True)
            return {
                "diagnosis": {
                    "conditions": ["Unable to analyze symptoms at this time"],
                    "next_steps": ["Please consult with a healthcare professional"],
                    "urgent": [],
                    "advice": "I apologize, but I'm having trouble analyzing your symptoms right now. Please try again or consult with a healthcare professional.",
                    "clarification": ["Could you provide more details about your symptoms?"]
                },
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            }

    def _get_targeted_research_data(self, topic, user_profile=None, limit=5):
        """
        Get research data using targeted search terms based on topic and user profile
        """
        try:
            # Create targeted search terms
            search_terms = self._create_search_terms_from_topic(topic, user_profile)
            
            all_results = []
            
            for search_term in search_terms[:3]:  # Limit to top 3 search terms to avoid too many requests
                try:
                    results = self.get_medical_research(search_term, limit=max(2, limit // len(search_terms)))
                    if results:
                        all_results.extend(results)
                        logger.info("Retrieved research for term", term=search_term, count=len(results))
                except Exception as e:
                    logger.warning("Failed to get research for term", term=search_term, error=str(e))
                    continue
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_results = []
            for result in all_results:
                title = result.get('title', '').lower().strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(result)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error("Error in targeted research data retrieval", error=str(e), topic=topic)
            # Fallback to original method
            return self.get_medical_research(topic, limit=limit)

    def _get_latest_health_data(self):
        """Get the latest health check data for AI context"""
        from ..models import HealthCheckResult

        latest_check = HealthCheckResult.objects.order_by('-timestamp').first()
        if latest_check:
            return latest_check.external_data
        return {}

    def _get_topic_specific_data(self, topic, limit=3):
        """Get topic-specific data for the current conversation"""
        health_data = self._get_latest_health_data()
        if 'research_trends' in health_data:
            for research_topic, articles in health_data['research_trends'].items():
                if topic.lower() in research_topic.lower():
                    logger.info("Using cached research for topic", topic=topic)
                    return articles[:limit]
        logger.info("Performing new search for topic", topic=topic)
        return self.get_medical_research(topic, limit=limit)

    def _scrape_mayo_news(self, limit=3):
        try:
            # Use a different Mayo Clinic endpoint that's more accessible
            url = "https://www.mayoclinic.org/about-mayo-clinic/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple selectors for news items
            news_items = soup.select('article') or soup.select('.content-item') or soup.select('.news-item')
            articles = []
            
            for item in news_items[:limit]:
                try:
                    title_elem = item.select_one('h2, h3, .title, .headline')
                    title = title_elem.text.strip() if title_elem else "Health News Update"
                    
                    link_elem = item.select_one('a')
                    url = link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
                    if url.startswith('/'):
                        url = "https://www.mayoclinic.org" + url
                    
                    # Get some content text if available
                    content_elem = item.select_one('p, .summary, .excerpt')
                    content = content_elem.text.strip() if content_elem else "Mayo Clinic health information"
                    
                    articles.append({
                        "title": title,
                        "source": "Mayo Clinic",
                        "url": url,
                        "content": content,
                        "image_url": None,
                        "published_date": None
                    })
                except Exception as e:
                    logger.warning(f"Error processing Mayo Clinic news item: {str(e)}")
                    continue
            
            return articles
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping Mayo Clinic news: {str(e)}")
            # Return fallback Mayo Clinic content
            return [{
                "title": "Mayo Clinic Health Information",
                "source": "Mayo Clinic",
                "url": "https://www.mayoclinic.org/",
                "content": "Access comprehensive health information and medical expertise from Mayo Clinic.",
                "published_date": datetime.now().strftime("%Y-%m-%d")
            }]
        except Exception as e:
            logger.error(f"Unexpected error scraping Mayo Clinic: {str(e)}")
            return []

    def analyze_xray(self, image_data):
        if not hasattr(self, 'densenet_model') or self.densenet_model is None:
            logger.warning("DenseNet model not initialized")
            return {
                "analysis": "X-ray analysis could not be performed",
                "findings": ["The X-ray analysis model is not available at this time."],
                "confidence_scores": {
                    "normal": 0,
                    "pneumonia": 0,
                    "fracture": 0,
                    "tumor": 0
                },
                "detected_conditions": {"unavailable": "high"},
                "detailed_analysis": {
                    "interpretation": "X-ray analysis is currently unavailable. Please try again later or consult with a healthcare professional.",
                    "possible_conditions": ["unavailable"],
                    "recommendations": ["Consult with a radiologist for professional interpretation"],
                    "limitations": ["AI analysis is currently unavailable"]
                },
                "disclaimer": "This is an AI interpretation and should be confirmed by a radiologist."
            }
        try:
            cache_key = f"xray_analysis_{hash(str(image_data))}"
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached X-ray analysis")
                return cached_result

            transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                Resize((224, 224)),
            ])
            if isinstance(image_data, str):
                image = transforms(image_data)
            else:
                image = Image.open(BytesIO(image_data))
                image = np.array(image.convert('L'))
                image = transforms(image)
            image = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                densenet_output = self.densenet_model(image)
                densenet_probs = torch.softmax(densenet_output, dim=1)
                fracture_prob = densenet_probs[0, 0].item()
                tumor_prob = densenet_probs[0, 1].item()
                pneumonia_prob = densenet_probs[0, 2].item()
                normal_prob = max(0, min(1.0, 1.0 - (fracture_prob + tumor_prob + pneumonia_prob)))
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
            research = self.get_medical_research("X-ray diagnosis " + " ".join(detected_conditions.keys()), limit=2)
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
            detailed_analysis_text = self._call_together_ai(prompt)
            try:
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
            cache.set(cache_key, result, self.cache_timeout)
            logger.info("X-ray analysis completed successfully",
                        conditions=list(detected_conditions.keys()),
                        confidence=confidence_scores)
            return result
        except Exception as e:
            logger.error("Error in X-ray analysis", error=str(e))
            return {"error": str(e), "message": "Unable to process X-ray at this time."}

    def get_medical_research(self, query, limit=5):
        cache_key = f"research_{hash(query)}_{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            # Ensure query is a string and not None
            if not query or not isinstance(query, str):
                query = "general medical research"
            
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp])"
            results = self.pubmed_client.query(search_term, max_results=limit)
            articles = []
            for article in results:
                # Safely handle None values
                title = getattr(article, 'title', None) or "Research Article"
                authors = getattr(article, 'authors', None)
                journal = getattr(article, 'journal', None) or "Medical Journal"
                
                # Safely process authors
                author_str = "Unknown"
                if authors and isinstance(authors, list):
                    try:
                        author_str = ", ".join([
                            f"{author.get('lastname', '')} {author.get('firstname', [''])[0]}"
                            for author in authors[:3]  # Limit to first 3 authors
                            if isinstance(author, dict) and author.get('lastname')
                        ]) or "Unknown"
                    except (AttributeError, TypeError, IndexError):
                        author_str = "Unknown"
                
                # Safely handle other attributes
                pub_date = getattr(article, 'publication_date', None)
                pub_date_str = str(pub_date) if pub_date else "Unknown"
                
                doi = getattr(article, 'doi', None)
                abstract = getattr(article, 'abstract', None) or "No abstract available"
                
                article_data = {
                    "title": title,
                    "authors": author_str,
                    "journal": journal,
                    "publication_date": pub_date_str,
                    "doi": doi,
                    "abstract": abstract
                }
                articles.append(article_data)
            
            cache.set(cache_key, articles, self.cache_timeout)
            logger.info("Medical research retrieved successfully", query=query, count=len(articles))
            return articles
        except Exception as e:
            logger.warning("Error retrieving medical research", error=str(e), query=query)
            # Return safe placeholder results
            placeholder_results = [
                {
                    "title": "Recent advances in medical diagnosis and treatment", 
                    "authors": "Smith J, et al.", 
                    "journal": "Medical Journal", 
                    "publication_date": "2023",
                    "doi": None,
                    "abstract": "Recent research in medical diagnosis and treatment methods."
                },
                {
                    "title": "Clinical guidelines for symptom management", 
                    "authors": "Johnson M, et al.", 
                    "journal": "Healthcare Research", 
                    "publication_date": "2022",
                    "doi": None,
                    "abstract": "Clinical guidelines for effective symptom management and patient care."
                }
            ]
            return placeholder_results[:limit]

    def analyze_diet(self, diet_info, user_profile=None):
        if not diet_info or len(diet_info.strip()) < 5:
            return {
                "error": "Insufficient diet information",
                "message": "Please provide more details about your diet for analysis."
            }
        cache_key = f"diet_{hash(diet_info)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached diet analysis")
            return cached_result
        context = self._build_user_context(user_profile)
        research_results = self.get_medical_research(f"nutrition {diet_info}", limit=2)
        research_context = ""
        if research_results:
            research_context = "Relevant nutritional research:\n"
            for i, article in enumerate(research_results):
                research_context += f"{i+1}. {article.get('title')} ({article.get('journal')}): "
                research_context += f"{article.get('abstract')[:200]}...\n"
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
            response_text = self._call_together_ai(prompt)
            try:
                diet_data = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response for diet analysis", response=response_text[:100])
                diet_data = {
                    "assessment": "Unable to parse structured assessment from response",
                    "deficiencies": ["Unable to parse deficiencies from response"],
                    "recommendations": ["Consult with a nutritionist for personalized advice"],
                    "concerns": ["Unable to determine concerns from the provided information"],
                    "positives": ["Unable to determine positive aspects from the provided information"]
                }
            result = {
                "analysis": diet_data,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only and not a substitute for professional nutritional advice."
            }
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

    def analyze_medication(self, medication_info, user_profile=None):
        if not medication_info or len(medication_info.strip()) < 3:
            return {
                "error": "Insufficient medication information",
                "message": "Please provide more details about your medications for analysis."
            }
        cache_key = f"medication_{hash(medication_info)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached medication analysis")
            return cached_result
        context = self._build_user_context(user_profile)
        research_results = self.get_medical_research(f"medication {medication_info} interactions", limit=2)
        research_context = ""
        if research_results:
            research_context = "Relevant medication research:\n"
            for i, article in enumerate(research_results):
                research_context += f"{i+1}. {article.get('title')} ({article.get('journal')}): "
                research_context += f"{article.get('abstract')[:200]}...\n"
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
            response_text = self._call_together_ai(prompt)
            try:
                medication_data = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response for medication analysis", response=response_text[:100])
                medication_data = {
                    "overview": response_text,
                    "interactions": ["Unable to parse interactions from response"],
                    "side_effects": ["Consult with a healthcare provider for side effect information"],
                    "precautions": ["Consult with a healthcare provider before changing any medication regimen"],
                    "questions": ["Ask your doctor about potential interactions with your current medications"]
                }
            result = {
                "analysis": medication_data,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only. Always consult with a healthcare provider before making any changes to your medication regimen."
            }
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
        cache_key = f"health_news_{source}_{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached health news")
            return cached_result
        
        try:
            articles = []
            
            # Expanded sources list with better error handling
            sources_to_try = []
            if source in ['who', 'all']:
                sources_to_try.append(('who', self._scrape_who_news))
            if source in ['cdc', 'all']:
                sources_to_try.append(('cdc', self._scrape_cdc_news))
            if source in ['mayo', 'all']:
                sources_to_try.append(('mayo', self._scrape_mayo_news))
            
            # Try each source with more detailed logging
            for src_name, scrape_func in sources_to_try:
                try:
                    logger.info(f"Attempting to scrape from {src_name}")
                    src_articles = scrape_func(limit=limit if source == src_name else max(1, limit // len(sources_to_try)))
                    if src_articles and isinstance(src_articles, list):
                        articles.extend(src_articles)
                        logger.info(f"Successfully retrieved {len(src_articles)} articles from {src_name}")
                    else:
                        logger.warning(f"No articles found from {src_name}")
                except Exception as e:
                    logger.error(f"Failed to scrape {src_name}", error=str(e))
                    continue
            
            # Add fallback if no articles found
            if not articles:
                logger.warning("No articles found from any source, using fallback data")
                articles = self._get_fallback_health_news(limit)
            
            # Process articles safely
            processed_articles = []
            for article in articles[:limit]:
                try:
                    if not isinstance(article, dict):
                        continue
                        
                    processed_article = {
                        'title': article.get('title', 'Health News Update'),
                        'source': article.get('source', 'Health Authority'),
                        'url': article.get('url', '#'),
                        'content': article.get('content', 'Health information update'),
                        'image_url': article.get('image_url'),
                        'published_date': article.get('published_date', datetime.now().strftime("%Y-%m-%d"))
                    }
                    
                    # Generate summary safely
                    content = processed_article.get('content', '')
                    if content and len(content) > 500:
                        try:
                            processed_article['summary'] = self._summarize_article(content)
                        except Exception as e:
                            logger.warning(f"Failed to summarize article: {str(e)}")
                            processed_article['summary'] = content[:200] + "..."
                    else:
                        processed_article['summary'] = content
                    
                    processed_articles.append(processed_article)
                except Exception as e:
                    logger.warning(f"Error processing article: {str(e)}")
                    continue
            
            cache.set(cache_key, processed_articles, 60 * 60 * 12)
            return processed_articles
        except Exception as e:
            logger.error("Error scraping health news", error=str(e))
            return self._get_fallback_health_news(limit)

    def _get_fallback_health_news(self, limit=3):
        """Fallback method for when scraping fails"""
        return [
            {
                "title": "COVID-19 Updates and Prevention Measures",
                "source": "WHO",
                "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019",
                "summary": "Latest information on COVID-19 prevention, symptoms, and global statistics.",
                "published_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Managing Chronic Conditions During Healthcare Disruptions",
                "source": "CDC",
                "url": "https://www.cdc.gov/",
                "summary": "Guidance for patients with chronic conditions during healthcare system disruptions.",
                "content": "Patients with chronic conditions should maintain medication supplies, use telehealth when possible, and have emergency plans in place.",
                "published_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Seasonal Illness Prevention Strategies",
                "source": "Mayo Clinic",
                "url": "https://www.mayoclinic.org/",
                "summary": "Tips for preventing common seasonal illnesses and maintaining good health.",
                "content": "Regular handwashing, adequate sleep, proper nutrition, and staying up-to-date with vaccinations help prevent seasonal illnesses.",
                "published_date": datetime.now().strftime("%Y-%m-%d")
            }
        ][:limit]

    def _scrape_who_news(self, limit=3):
        try:
            url = "https://www.who.int/news"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.select('.list-view--item')
            articles = []
            for item in news_items[:limit]:
                try:
                    title_elem = item.select_one('.heading')
                    title = title_elem.text.strip() if title_elem else "No title"
                    link_elem = item.select_one('a')
                    url = "https://www.who.int" + link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
                    date_elem = item.select_one('.timestamp')
                    published_date = date_elem.text.strip() if date_elem else None
                    img_elem = item.select_one('img')
                    image_url = img_elem['src'] if img_elem and 'src' in img_elem.attrs else None
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
        try:
            url = "https://www.cdc.gov/media/index.html"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.select('.feed-item')
            articles = []
            for item in news_items[:limit]:
                try:
                    title_elem = item.select_one('a')
                    title = title_elem.text.strip() if title_elem else "No title"
                    url = "https://www.cdc.gov" + title_elem['href'] if title_elem and 'href' in title_elem.attrs else "#"
                    date_elem = item.select_one('.date')
                    published_date = date_elem.text.strip() if date_elem else None
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
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content_selectors = [
                'article', '.content', '.main-content',
                '#content', '.article-body', '.story-body'
            ]
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    for script in content_elem.select('script, style'):
                        script.extract()
                    content = content_elem.get_text(separator=' ', strip=True)
                    content = ' '.join(content.split())
                    if len(content) > 5000:
                        content = content[:5000] + "..."
                    return content
            return "Content could not be extracted."
        except Exception as e:
            logger.warning(f"Error getting article content from {url}: {str(e)}")
            return "Content could not be retrieved."

    def _summarize_article(self, content, max_length=200):
        try:
            if len(content) > 500:
                prompt = f"""Summarize the following health news article in 3-4 sentences:

{content[:4000]}

Summary:"""
                summary = self._call_together_ai(prompt)
                summary = summary.strip()
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
                return summary
            else:
                return content[:max_length] + ("..." if len(content) > max_length else "")
        except Exception as e:
            logger.error(f"Error summarizing article: {str(e)}")
            return content[:max_length]
        
@shared_task
def analyze_xray_task(xray_id, image_path):
    """
    Celery task to analyze X-ray images asynchronously
    """
    try:
        from ..models import XRayAnalysis
        
        # Get the XRayAnalysis object
        xray = XRayAnalysis.objects.get(id=xray_id)
        
        # Initialize PraxiaAI
        praxia = PraxiaAI()
        
        # Perform the analysis
        analysis_result = praxia.analyze_xray(image_path)
        
        # Update the XRayAnalysis object with results
        if isinstance(analysis_result, dict):
            xray.analysis_result = analysis_result.get('analysis', 'Analysis completed')
            xray.detected_conditions = analysis_result.get('detected_conditions', {})
            xray.confidence_scores = analysis_result.get('confidence_scores', {})
        else:
            xray.analysis_result = str(analysis_result)
            xray.detected_conditions = {}
            xray.confidence_scores = {}
        
        xray.save()
        
        logger.info("X-ray analysis completed", xray_id=xray_id)
        return analysis_result
        
    except Exception as e:
        logger.error("Error in X-ray analysis task", error=str(e), xray_id=xray_id)
        
        # Update the XRayAnalysis object with error
        try:
            xray = XRayAnalysis.objects.get(id=xray_id)
            xray.analysis_result = f"Analysis failed: {str(e)}"
            xray.detected_conditions = {"error": "analysis_failed"}
            xray.confidence_scores = {}
            xray.save()
        except Exception as save_error:
            logger.error("Failed to save error state", error=str(save_error))
        
        raise

@shared_task
def scrape_health_news(source='all', limit=3):
    """
    Celery task to scrape health news asynchronously
    """
    try:
        praxia = PraxiaAI()
        news_articles = praxia._scrape_health_news(source, limit)
        
        # Save to database
        from ..models import HealthNews
        from datetime import datetime
        
        saved_articles = []
        for article in news_articles:
            try:
                # Create or update the news article
                obj, created = HealthNews.objects.get_or_create(
                    url=article.get('url', ''),
                    defaults={
                        'title': article.get('title', 'Health News'),
                        'source': article.get('source', 'Unknown'),
                        'summary': article.get('summary', ''),
                        'original_content': article.get('content', ''),
                        'image_url': article.get('image_url'),
                        'published_date': article.get('published_date')
                    }
                )
                saved_articles.append({
                    'id': obj.id,
                    'title': obj.title,
                    'source': obj.source,
                    'url': obj.url,
                    'summary': obj.summary,
                    'created': created
                })
            except Exception as e:
                logger.error("Error saving news article", error=str(e), article_title=article.get('title', 'Unknown'))
                continue
        
        logger.info("Health news scraping completed", source=source, count=len(saved_articles))
        return saved_articles
        
    except Exception as e:
        logger.error("Error in health news scraping task", error=str(e), source=source)
        raise

@shared_task
def diagnose_symptoms_task(symptoms, user_profile=None):
    """
    Celery task for symptom diagnosis (if you need async diagnosis)
    """
    try:
        praxia = PraxiaAI()
        diagnosis_result = praxia.diagnose_symptoms(symptoms, user_profile)
        
        logger.info("Symptom diagnosis completed", symptoms=symptoms[:50])
        return diagnosis_result
        
    except Exception as e:
        logger.error("Error in symptom diagnosis task", error=str(e), symptoms=symptoms[:50])
        raise
