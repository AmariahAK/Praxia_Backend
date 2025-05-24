import os
import requests
import json
import pymed
import numpy as np
import collections
import torch
import structlog
from datetime import datetime
import re
from typing import Dict, List, Any
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

    def _get_latest_health_data(self):
        """
        Get the latest health data from cache or external sources
        """
        try:
            # Try to get cached health data first
            cached_data = cache.get('latest_health_data')
            if cached_data:
                logger.info("Using cached health data")
                return cached_data
            
            # If no cached data, create a basic health data structure
            health_data = {
                'health_news': [],
                'research_trends': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Try to get latest health news
            try:
                health_news = self._scrape_health_news(source='all', limit=3)
                health_data['health_news'] = health_news
            except Exception as e:
                logger.warning(f"Failed to get health news: {str(e)}")
                health_data['health_news'] = []
            
            # Try to get research trends for common topics
            try:
                research_topics = ["COVID-19", "heart disease", "diabetes"]
                research_data = {}
                for topic in research_topics:
                    try:
                        research_data[topic] = self.get_medical_research(query=topic, limit=1)
                    except Exception as e:
                        logger.warning(f"Failed to get research for {topic}: {str(e)}")
                        research_data[topic] = []
                health_data['research_trends'] = research_data
            except Exception as e:
                logger.warning(f"Failed to get research trends: {str(e)}")
                health_data['research_trends'] = {}
            
            # Cache the health data for 6 hours
            cache.set('latest_health_data', health_data, 60 * 60 * 6)
            logger.info("Health data updated and cached")
            return health_data
            
        except Exception as e:
            logger.error(f"Error getting latest health data: {str(e)}")
            # Return basic structure if everything fails
            return {
                'health_news': [],
                'research_trends': {},
                'last_updated': datetime.now().isoformat()
            }

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
        
        try:
            # Clean the input more safely
            cleaned = str(symptoms).replace('<', '').replace('>', '').strip()
            
            # Remove problematic characters that might cause separator issues
            cleaned = re.sub(r'[^\w\s.,;:!?()-]', ' ', cleaned)
            
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
            
            # Normalize whitespace and quotes
            cleaned = cleaned.replace("'", "'").replace(""", '"').replace(""", '"')
            cleaned = cleaned.rstrip("'\"")
            
            if not cleaned.strip():
                return "general health inquiry"
            
            # Remove excessive whitespace
            cleaned = ' '.join(cleaned.split()) 
            
            # Limit length to prevent overly long queries
            if len(cleaned) > 500:
                cleaned = cleaned[:500] + "..."
            
            return cleaned.strip() if cleaned.strip() else "general health inquiry"
            
        except Exception as e:
            logger.error("Error preprocessing symptoms", error=str(e), symptoms=str(symptoms)[:50])
            return "general health inquiry"

    def _extract_medical_topic(self, symptoms: str, user_profile: Dict[str, Any] = None) -> str:
        """
        Extract a concise medical topic from symptoms for targeted searches
        """
        try:
            # Clean and prepare the symptoms text
            cleaned_symptoms = re.sub(r'[^\w\s]', ' ', symptoms.lower())
            cleaned_symptoms = ' '.join(cleaned_symptoms.split())
            
            # Common medical terms and their categories
            symptom_keywords = {
                'respiratory': ['cough', 'breathing', 'chest pain', 'shortness of breath', 'wheezing', 'phlegm', 'runny nose', 'congestion'],
                'cardiovascular': ['chest pain', 'heart palpitations', 'dizziness', 'fainting'],
                'gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'stomach', 'digestive'],
                'neurological': ['headache', 'dizziness', 'numbness', 'tingling', 'memory', 'confusion'],
                'musculoskeletal': ['joint pain', 'muscle pain', 'back pain', 'stiffness', 'swelling'],
                'dermatological': ['rash', 'itching', 'skin', 'lesion', 'bump'],
                'systemic': ['fever', 'fatigue', 'weight loss', 'weight gain', 'night sweats']
            }
            
            # Find matching categories
            matched_categories = []
            for category, keywords in symptom_keywords.items():
                if any(keyword in cleaned_symptoms for keyword in keywords):
                    matched_categories.append(category)
            
            # Generate topic based on matched categories and key symptoms
            if matched_categories:
                primary_category = matched_categories[0]
                
                # Extract key symptoms for the topic
                key_symptoms = []
                for category, keywords in symptom_keywords.items():
                    if category in matched_categories:
                        for keyword in keywords:
                            if keyword in cleaned_symptoms:
                                key_symptoms.append(keyword)
                
                # Limit to top 3 most relevant symptoms
                topic_symptoms = key_symptoms[:3]
                topic = f"{primary_category} {' '.join(topic_symptoms)}"
                
                # Add demographic context if available
                if user_profile:
                    if user_profile.get('age'):
                        age = int(user_profile['age'])
                        if age < 18:
                            topic = f"pediatric {topic}"
                        elif age > 65:
                            topic = f"elderly {topic}"
                    
                    if user_profile.get('gender') in ['male', 'female']:
                        topic = f"{user_profile['gender']} {topic}"
                
                return topic
            else:
                # Fallback: extract key medical terms
                medical_terms = re.findall(r'\b(?:pain|ache|fever|cough|fatigue|nausea|dizziness|headache|rash|swelling)\b', cleaned_symptoms)
                if medical_terms:
                    return ' '.join(medical_terms[:3])
                else:
                    return "general medical consultation"
                    
        except Exception as e:
            logger.error("Error extracting medical topic", error=str(e))
            return "general medical consultation"

    def _create_targeted_search_queries(self, topic: str, user_profile: Dict[str, Any] = None, max_queries: int = 3) -> List[str]:
        """
        Create targeted search queries based on the extracted topic
        """
        queries = [topic]
        
        if user_profile:
            # Add demographic-specific queries
            if user_profile.get('country'):
                country = user_profile['country'].lower()
                # Regional health considerations
                regional_conditions = {
                    'kenya': ['malaria', 'tuberculosis'],
                    'nigeria': ['malaria', 'sickle cell'],
                    'india': ['tuberculosis', 'diabetes'],
                    'usa': ['diabetes', 'hypertension'],
                    'uk': ['diabetes', 'cardiovascular'],
                    'canada': ['diabetes', 'cardiovascular'],
                    'australia': ['skin cancer'],
                    'brazil': ['dengue', 'zika'],
                    'china': ['hepatitis', 'tuberculosis'],
                    'japan': ['stroke', 'cardiovascular'],
                }
                
                for region, conditions in regional_conditions.items():
                    if region in country:
                        for condition in conditions[:1]:  # Limit to 1 condition
                            queries.append(f"{topic} {condition}")
                        break
            
            # Add age-specific query
            if user_profile.get('age'):
                age = int(user_profile['age'])
                if age < 18:
                    queries.append(f"{topic} pediatric children")
                elif age > 65:
                    queries.append(f"{topic} elderly geriatric")
        
        return queries[:max_queries]

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

            # Extract medical topic first - THIS IS THE KEY TOPIC TO USE
            if chat_topic and chat_topic != "New Chat":
                medical_topic = chat_topic
            else:
                medical_topic = self._extract_medical_topic(processed_symptoms, user_profile)
            
            logger.info("Extracted medical topic", topic=medical_topic)
        
            # Use the medical topic as the PRIMARY research query
            cache_key = f"diagnosis_{hash(processed_symptoms)}_{hash(str(user_profile))}_{hash(medical_topic)}"
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached diagnosis", symptoms=processed_symptoms[:30])
                return cached_result

            context = self._build_user_context(user_profile)
            health_data = self._get_latest_health_data()
            
            # UPDATED: Use the medical topic directly for research queries
            # This makes it faster and more accurate as suggested
            primary_research_queries = [
                medical_topic,  # Use the exact topic first
                f"{medical_topic} treatment",  # Add treatment focus
                f"{medical_topic} diagnosis"   # Add diagnostic focus
            ]
            
            # Add user-specific context if available
            if user_profile:
                if user_profile.get('age'):
                    age = int(user_profile['age'])
                    if age < 18:
                        primary_research_queries.append(f"{medical_topic} pediatric")
                    elif age > 65:
                        primary_research_queries.append(f"{medical_topic} elderly")
                
                if user_profile.get('country'):
                    country = user_profile['country'].lower()
                    # Add regional considerations
                    regional_terms = {
                        'kenya': f"{medical_topic} malaria",
                        'nigeria': f"{medical_topic} tropical",
                        'india': f"{medical_topic} tuberculosis",
                        'usa': f"{medical_topic} guidelines",
                        'uk': f"{medical_topic} NHS"
                    }
                    if country in regional_terms:
                        primary_research_queries.append(regional_terms[country])
            
            # Get research data using the medical topic
            all_research = []
            for i, query in enumerate(primary_research_queries[:3]):  # Limit to 3 queries for speed
                try:
                    research = self.get_medical_research(query, limit=2)
                    all_research.extend(research)
                    logger.info("Retrieved research for topic-based query", 
                               query=query, count=len(research), priority=i+1)
                except Exception as e:
                    logger.warning("Failed to get research for topic-based query", 
                                  query=query, error=str(e))
                    continue
            
            # Remove duplicates and limit results
            seen_titles = set()
            filtered_research = []
            for article in all_research:
                title = article.get('title', '').lower().strip()
                if title and title not in seen_titles and len(filtered_research) < 5:
                    seen_titles.add(title)
                    filtered_research.append(article)
            
            # Filter research results based on user profile relevance
            final_research = self._filter_research_by_relevance(filtered_research, user_profile, max_results=3)
            
            # Build research context
            research_context = ""
            if final_research:
                research_context = f"Relevant medical research for {medical_topic}:\n"
                for i, article in enumerate(final_research):
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
Medical topic: {medical_topic}
Chat context: {chat_topic or 'General consultation'}

{research_context}

{news_context}

WHO guidelines recommend careful assessment of symptoms and considering local disease prevalence.
Mayo Clinic emphasizes that symptom diagnosis should consider patient history and risk factors.

Analyze these symptoms and provide a response in JSON format with these keys:
1. "conditions": [List of potential conditions, ordered by likelihood]
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
                    if "```json" in response_text and "```" in response_text:
                        json_content = response_text.split("```json")[1].split("```")[0].strip()
                        try:
                            diagnosis_data = json.loads(json_content)
                        except json.JSONDecodeError:
                            raise
                    elif "```" in response_text and "```" in response_text:
                        json_content = response_text.split("```")[1].split("```")[0].strip()
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
                    "related_research": final_research,
                    "medical_topic": medical_topic,
                    "search_queries_used": primary_research_queries[:3],  # Show which queries were used
                    "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
                }
                cache.set(cache_key, result, self.cache_timeout)
                logger.info("Diagnosis completed successfully using topic-based research", 
                           symptoms=processed_symptoms[:30], 
                                                      topic=medical_topic,
                           research_count=len(final_research))
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
                    "medical_topic": medical_topic,
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
                "medical_topic": "general medical consultation",
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            }

    def get_medical_research(self, query, limit=5):
        """
        Updated to handle shorter, more targeted queries
        """
        cache_key = f"research_{hash(query)}_{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            # Ensure query is a string and not None
            if not query or not isinstance(query, str):
                query = "general medical research"
            
            # Clean and limit query length
            query = query.strip()
            if len(query) > 100:  # Limit query length
                query = query[:100]
            
            # Create a more targeted PubMed search
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp] OR Case Reports[ptyp])"
            
            logger.info("Medical research query", search_term=search_term)
            
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
            logger.error("Error retrieving medical research", error=str(e), query=query)
            # Return safe placeholder results
            placeholder_results = [
                {
                    "title": f"Recent advances in {query[:50]}", 
                    "authors": "Medical Research Team", 
                    "journal": "Medical Journal", 
                    "publication_date": "2023",
                    "doi": None,
                    "abstract": f"Recent research in {query[:50]} and related medical conditions."
                }
            ]
            return placeholder_results[:limit]

    def analyze_diet(self, dietary_info, user_profile=None):
        """
        Analyze user's dietary information and provide recommendations
        """
        try:
            cache_key = f"diet_analysis_{hash(dietary_info)}_{hash(str(user_profile))}"
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result

            context = self._build_user_context(user_profile)
            
            prompt = f"""You are Praxia, a medical AI assistant specialized in nutrition. {self.identity}

{context}

Analyze this dietary information: {dietary_info}

Provide a comprehensive dietary analysis in JSON format with these keys:
1. "nutritional_assessment": Overall assessment of the diet
2. "strengths": Positive aspects of the current diet
3. "areas_for_improvement": Areas that need attention
4. "recommendations": Specific dietary recommendations
5. "health_considerations": Health implications based on user profile
6. "meal_suggestions": Suggested meal ideas

Your response must be valid JSON that can be parsed programmatically.
"""
            
            response_text = self._call_together_ai(prompt)
            
            try:
                diet_analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown blocks
                if "```json" in response_text:
                    json_content = response_text.split("```json")[1].split("```")[0].strip()
                    diet_analysis = json.loads(json_content)
                else:
                    raise
            
            result = {
                "diet_analysis": diet_analysis,
                "disclaimer": "This dietary advice is for informational purposes only. Consult with a registered dietitian for personalized nutrition plans."
            }
            
            cache.set(cache_key, result, self.cache_timeout)
            return result
            
        except Exception as e:
            logger.error("Error in diet analysis", error=str(e))
            return {
                "diet_analysis": {
                    "nutritional_assessment": "Unable to analyze diet at this time",
                    "recommendations": ["Consult with a registered dietitian for personalized advice"],
                    "health_considerations": ["Maintain a balanced diet with variety"]
                },
                "disclaimer": "This dietary advice is for informational purposes only."
            }

    def analyze_medication(self, medication_info, user_profile=None):
        """
        Analyze medication information and check for potential interactions
        """
        try:
            cache_key = f"medication_analysis_{hash(medication_info)}_{hash(str(user_profile))}"
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result

            context = self._build_user_context(user_profile)
            
            prompt = f"""You are Praxia, a medical AI assistant specialized in medication safety. {self.identity}

{context}

Analyze this medication information: {medication_info}

Provide a medication analysis in JSON format with these keys:
1. "medication_review": Overview of the medications mentioned
2. "potential_interactions": Potential drug interactions to be aware of
3. "side_effects": Common side effects to monitor
4. "precautions": Important precautions based on user profile
5. "recommendations": Recommendations for safe medication use
6. "when_to_consult": When to contact healthcare provider

Your response must be valid JSON that can be parsed programmatically.
"""
            
            response_text = self._call_together_ai(prompt)
            
            try:
                medication_analysis = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_content = response_text.split("```json")[1].split("```")[0].strip()
                    medication_analysis = json.loads(json_content)
                else:
                    raise
            
            result = {
                "medication_analysis": medication_analysis,
                "disclaimer": "This information is not a substitute for professional medical advice. Always consult your healthcare provider about medications."
            }
            
            cache.set(cache_key, result, self.cache_timeout)
            return result
            
        except Exception as e:
            logger.error("Error in medication analysis", error=str(e))
            return {
                "medication_analysis": {
                    "medication_review": "Unable to analyze medications at this time",
                    "recommendations": ["Always consult your healthcare provider or pharmacist about medication concerns"],
                    "when_to_consult": ["Contact your healthcare provider immediately if you experience adverse effects"]
                },
                "disclaimer": "This information is not a substitute for professional medical advice."
            }

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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            url = "https://www.who.int/news"
            response = requests.get(url, timeout=10, headers=headers)
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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            url = "https://www.cdc.gov/media/index.html"
            response = requests.get(url, timeout=10, headers=headers)
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

    def _scrape_mayo_news(self, limit=3):
        """Mayo Clinic news scraper with better error handling"""
        try:
            # Add headers to avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Try alternative Mayo Clinic news sources
            mayo_urls = [
                "https://newsnetwork.mayoclinic.org/",  # Alternative URL
                "https://www.mayoclinic.org/about-mayo-clinic/newsnetwork",  # Original URL
            ]
            
            for url in mayo_urls:
                try:
                    response = requests.get(url, timeout=10, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Try multiple selectors
                        selectors = [
                            '.content-item',
                            '.news-item',
                            '.article-item',
                            'article',
                            '.card'
                        ]
                        
                        news_items = []
                        for selector in selectors:
                            news_items = soup.select(selector)
                            if news_items:
                                break
                        
                        if not news_items:
                            continue
                            
                        articles = []
                        for item in news_items[:limit]:
                            try:
                                # Try multiple title selectors
                                title_elem = item.select_one('h3 a, h2 a, .title a, a.headline')
                                title = title_elem.text.strip() if title_elem else "No title"
                                
                                article_url = "#"
                                if title_elem and 'href' in title_elem.attrs:
                                    href = title_elem['href']
                                    if href.startswith('/'):
                                        article_url = f"https://www.mayoclinic.org{href}"
                                    elif href.startswith('http'):
                                        article_url = href
                                
                                # Try multiple date selectors
                                date_elem = item.select_one('.date, .publish-date, time')
                                published_date = date_elem.text.strip() if date_elem else None
                                
                                content = self._get_article_content(article_url)
                                
                                articles.append({
                                    "title": title,
                                    "source": "Mayo Clinic",
                                    "url": article_url,
                                    "content": content,
                                    "image_url": None,
                                    "published_date": published_date
                                })
                            except Exception as e:
                                logger.warning(f"Error processing Mayo news item: {str(e)}")
                                continue
                        
                        if articles:
                            return articles
                            
                    elif response.status_code == 403:
                        logger.warning(f"403 Forbidden for {url}, trying next URL")
                        continue
                    else:
                        logger.warning(f"HTTP {response.status_code} for {url}")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request error for {url}: {str(e)}")
                    continue
            
            # If all URLs fail, return fallback content
            logger.warning("All Mayo Clinic URLs failed, returning fallback content")
            return [{
                "title": "Mayo Clinic Health Information",
                "source": "Mayo Clinic",
                "url": "https://www.mayoclinic.org/",
                "content": "Mayo Clinic provides comprehensive health information and medical expertise.",
                "image_url": None,
                "published_date": None
            }]
            
        except Exception as e:
            logger.error(f"Error scraping Mayo news: {str(e)}")
            return []

    def _get_article_content(self, url):
        try:
            # Add headers to avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content_selectors = [
                'article', '.content', '.main-content',
                '#content', '.article-body', '.story-body',
                '.entry-content', '.post-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove unwanted elements
                    for script in content_elem.select('script, style, nav, footer, aside'):
                        script.extract()
                    
                    content = content_elem.get_text(separator=' ', strip=True)
                    content = ' '.join(content.split())
                    
                    if len(content) > 5000:
                        content = content[:5000] + "..."
                    
                    if len(content) > 100:  # Only return if we got substantial content
                        return content
            
            return "Content could not be extracted."
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error getting article content from {url}: {str(e)}")
            return "Content could not be retrieved due to access restrictions."
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
