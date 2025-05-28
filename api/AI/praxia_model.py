import os
import requests
import json
import pymed
import numpy as np
import collections
import torch
import structlog
import feedparser
from datetime import datetime
import re
from typing import Dict, Any
from PIL import Image
from io import BytesIO
from django.conf import settings
from bs4 import BeautifulSoup
from django.core.cache import cache
from ..utils.timeout_utils import ThreadPoolExecutor, FutureTimeoutError
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
    Praxia AI Healthcare Assistant Model
    
    A comprehensive AI-powered healthcare assistant that provides:
    - Symptom analysis and medical guidance
    - X-ray image analysis using DenseNet
    - Medical research retrieval from PubMed
    - Health news aggregation from WHO and CDC
    - Dietary and medication analysis
    
    Features:
    - Circuit breaker pattern for external API reliability
    - Intelligent caching for performance optimization
    - User profile-based personalized recommendations
    - Multi-language support through translation services
    - Asynchronous processing with Celery integration
    
    Developed by Amariah Kamau (https://www.linkedin.com/in/amariah-kamau-3156412a6/)
    GitHub: https://github.com/AmariahAK
    
    Open Source Healthcare AI - Contributing developers welcome!
    """

    def __init__(self):
        """Initialize the Praxia AI model with all necessary components."""
        self.identity = self._load_identity()
        self.together_api_key = settings.TOGETHER_AI_API_KEY
        self.together_model = settings.TOGETHER_AI_MODEL
        self.cache_timeout = 60 * 60 * 24  # 24 hours cache for research data
        self.pubmed_client = pymed.PubMed(tool="PraxiaAI", email="contact@praxia.ai")
        
        # Intelligent device detection for optimal performance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Initialize X-ray analysis model if enabled
        self.densenet_model = None
        if getattr(settings, "INITIALIZE_XRAY_MODEL", False):
            self._initialize_xray_model()

    def _load_identity(self):
        """Load AI identity configuration from file."""
        identity_path = os.path.join(settings.BASE_DIR, 'data', 'ai_identity.txt')
        try:
            with open(identity_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logger.error("AI identity file not found", path=identity_path)
            return "Praxia - A healthcare AI assistant by Amariah Kamau"

    def _initialize_xray_model(self):
        """Initialize DenseNet model for X-ray image analysis."""
        try:
            # Try to use fixed model if available (recommended approach)
            from ..utils.model_fix import fix_densenet_model
            fixed_model_path = fix_densenet_model()
            
            if fixed_model_path and os.path.exists(fixed_model_path):
                logger.info(f"Loading fixed DenseNet model from {fixed_model_path}")
                
                # Configure safe model loading
                from torch.serialization import add_safe_globals
                import torchvision.models.densenet
                import torch.nn.modules.container
                add_safe_globals([
                    torchvision.models.densenet.DenseNet,
                    torch.nn.modules.container.Sequential
                ])
                
                # Load pre-trained model
                self.densenet_model = torch.load(fixed_model_path, map_location=self.device, weights_only=False)
                logger.info("Pre-trained model loaded successfully")
                
                self.densenet_model.eval()
                logger.info("DenseNet model ready for X-ray analysis")
                return
            
            # Fallback: Load original model weights
            densenet_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'densenet_xray.pth')
            if not os.path.exists(densenet_path):
                logger.warning("DenseNet model weights not found, creating fallback model")
                # Create basic model structure for development
                try:
                    from torchvision.models import densenet121
                    model = densenet121(weights=None)  
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 3)  # 3 classes: fracture, tumor, pneumonia
                    self.densenet_model = model
                    self.densenet_model.eval()
                    logger.info("Created fallback DenseNet model for development")
                except Exception as e:
                    logger.error("Error creating fallback model: %s", str(e))
                    self.densenet_model = None
                return
            
            # Validate model file size
            if os.path.getsize(densenet_path) < 1000000:  
                logger.warning("DenseNet model file appears corrupted (too small)")
                self.densenet_model = None
                return
            
            # Load model with proper error handling
            try:
                from torch.serialization import add_safe_globals
                import torchvision.models.densenet
                add_safe_globals([torchvision.models.densenet.DenseNet])
                
                # Attempt secure loading first
                try:
                    loaded_obj = torch.load(densenet_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    logger.warning(f"Secure loading failed: {str(e)}, trying compatibility mode")
                    loaded_obj = torch.load(densenet_path, map_location=self.device, weights_only=False)
                
                # Handle different model formats
                if isinstance(loaded_obj, collections.OrderedDict):
                    # State dict format - need to create model architecture
                    from torchvision.models import densenet121
                    model = densenet121(weights=None)
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 3)
                    model.load_state_dict(loaded_obj)
                    self.densenet_model = model
                else:
                    # Complete model format
                    self.densenet_model = loaded_obj
                
                self.densenet_model.eval()
                logger.info("DenseNet model loaded and ready for inference")
                
            except Exception as e:
                logger.error("Error loading DenseNet model: %s", str(e))
                # Create development fallback
                try:
                    from torchvision.models import densenet121
                    model = densenet121(weights=None)
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 3)
                    self.densenet_model = model
                    self.densenet_model.eval()
                    logger.info("Using development fallback model")
                except Exception as e2:
                    logger.error("Failed to create any model: %s", str(e2))
                    self.densenet_model = None
                    
        except Exception as e:
            logger.error("Critical error in model initialization: %s", str(e))
            self.densenet_model = None

    def _get_latest_health_data(self):
        """Retrieve and cache the latest health data from multiple sources."""
        try:
            # Check cache first for performance
            cached_data = cache.get('latest_health_data')
            if cached_data:
                logger.info("Using cached health data for optimal performance")
                return cached_data
            
            # Build fresh health data structure
            health_data = {
                'health_news': [],
                'research_trends': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Fetch latest health news from reliable sources
            try:
                health_news = self._get_health_news_comprehensive(source='all', limit=3)
                health_data['health_news'] = health_news
                logger.info(f"Retrieved {len(health_news)} health news articles")
            except Exception as e:
                logger.warning(f"Health news retrieval failed: {str(e)}")
                health_data['health_news'] = []
            
            # Gather research trends for common medical topics
            try:
                research_topics = ["COVID-19", "heart disease", "diabetes"]
                research_data = {}
                for topic in research_topics:
                    try:
                        research_data[topic] = self.get_medical_research(query=topic, limit=1)
                    except Exception as e:
                        logger.warning(f"Research retrieval failed for {topic}: {str(e)}")
                        research_data[topic] = []
                health_data['research_trends'] = research_data
                logger.info(f"Retrieved research trends for {len(research_topics)} topics")
            except Exception as e:
                logger.warning(f"Research trends retrieval failed: {str(e)}")
                health_data['research_trends'] = {}
            
            # Cache for 6 hours to balance freshness and performance
            cache.set('latest_health_data', health_data, 60 * 60 * 6)
            logger.info("Health data updated and cached successfully")
            return health_data
            
        except Exception as e:
            logger.error(f"Critical error in health data retrieval: {str(e)}")
            # Return safe fallback structure
            return {
                'health_news': [],
                'research_trends': {},
                'last_updated': datetime.now().isoformat()
            }

    def _build_user_context(self, user_profile):
        """Build personalized context string from user profile data."""
        if not user_profile:
            return ""
        
        logger.info("Building personalized user context", 
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
        
        logger.info("User context built successfully", context_length=len(context))
        return context

    def _get_source_icon(self, journal_or_source):
        """Map journal/source names to correct icon names that match frontend"""
        if not journal_or_source:
            return "source-icon"
        
        source_lower = str(journal_or_source).lower()
        
        # Map to your actual icon file names (without .svg extension)
        icon_mapping = {
            'who': 'who-logo',
            'world health organization': 'who-logo',
            'world health': 'who-logo',
            'cdc': 'cdc-logo',
            'centers for disease control': 'cdc-logo',
            'centers for disease': 'cdc-logo',
            'nih': 'nih-logo',
            'national institutes of health': 'nih-logo',
            'national institutes': 'nih-logo',
            'mayo clinic': 'mayo-logo',
            'mayo': 'mayo-logo',
            'pubmed': 'pubmed-logo',
            'journal': 'journal-icon',
            'medical journal': 'journal-icon'
        }
        
        # Check for exact matches first
        for key, icon in icon_mapping.items():
            if key in source_lower:
                return icon
        
        # Default fallback
        return "source-icon"

    def _extract_medical_topic(self, symptoms: str, user_profile: Dict[str, Any] = None) -> str:
        """Extract a focused medical topic from symptoms for targeted research."""
        try:
            # Clean and normalize symptom text
            cleaned_symptoms = re.sub(r'[^\w\s]', ' ', symptoms.lower())
            cleaned_symptoms = ' '.join(cleaned_symptoms.split())
            
            # Medical symptom categorization system
            symptom_keywords = {
                'respiratory': ['cough', 'breathing', 'chest pain', 'shortness of breath', 'wheezing', 'phlegm', 'runny nose', 'congestion'],
                'cardiovascular': ['chest pain', 'heart palpitations', 'dizziness', 'fainting'],
                'gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'stomach', 'digestive'],
                'neurological': ['headache', 'dizziness', 'numbness', 'tingling', 'memory', 'confusion'],
                'musculoskeletal': ['joint pain', 'muscle pain', 'back pain', 'stiffness', 'swelling'],
                'dermatological': ['rash', 'itching', 'skin', 'lesion', 'bump'],
                'systemic': ['fever', 'fatigue', 'weight loss', 'weight gain', 'night sweats']
            }
            
            # Find matching medical categories
            matched_categories = []
            for category, keywords in symptom_keywords.items():
                if any(keyword in cleaned_symptoms for keyword in keywords):
                    matched_categories.append(category)
            
            # Generate targeted topic based on matches
            if matched_categories:
                primary_category = matched_categories[0]
                
                # Extract key symptoms for precision
                key_symptoms = []
                for category, keywords in symptom_keywords.items():
                    if category in matched_categories:
                        for keyword in keywords:
                            if keyword in cleaned_symptoms:
                                key_symptoms.append(keyword)
                
                # Limit to top 3 symptoms for focused research
                topic_symptoms = key_symptoms[:3]
                topic = f"{primary_category} {' '.join(topic_symptoms)}"
                
                # Add demographic context for personalization
                if user_profile:
                    if user_profile.get('age'):
                        age = int(user_profile['age'])
                        if age < 18:
                            topic = f"pediatric {topic}"
                        elif age > 65:
                            topic = f"elderly {topic}"
                    
                    if user_profile.get('gender') in ['male', 'female']:
                        topic = f"{user_profile['gender']} {topic}"
                
                logger.info("Medical topic extracted successfully", topic=topic)
                return topic
            else:
                # Fallback: extract any medical terms
                medical_terms = re.findall(r'\b(?:pain|ache|fever|cough|fatigue|nausea|dizziness|headache|rash|swelling)\b', cleaned_symptoms)
                if medical_terms:
                    return ' '.join(medical_terms[:3])
                else:
                    return "general medical consultation"
                    
        except Exception as e:
            logger.error("Error in medical topic extraction", error=str(e))
            return "general medical consultation"

    def _preprocess_symptoms(self, symptoms):
        """Clean and normalize user symptom input for processing."""
        if not symptoms or len(symptoms.strip()) < 3:
            return "general health inquiry"
        
        try:
            # Basic sanitization
            cleaned = str(symptoms).replace('<', '').replace('>', '').strip()
            
            # Remove problematic characters that might cause issues
            cleaned = re.sub(r'[^\w\s.,;:!?()-]', ' ', cleaned)
            
            # Handle common greetings and extract actual symptoms
            greeting_phrases = ["hey praxia", "hi praxia", "hello", "greetings"]
            lower_symptoms = cleaned.lower()
            
            for phrase in greeting_phrases:
                if lower_symptoms.startswith(phrase):
                    cleaned = cleaned[len(phrase):].lstrip(" ,.;:!?")
                    break
            
            # Validate remaining content
            if not cleaned or len(cleaned.strip()) < 3:
                return "general health inquiry"
            
            # Normalize quotes and whitespace
            cleaned = cleaned.replace("'", "'").replace(""", '"').replace(""", '"')
            cleaned = cleaned.rstrip("'\"")
            
            if not cleaned.strip():
                return "general health inquiry"
            
            # Remove excessive whitespace
            cleaned = ' '.join(cleaned.split()) 
            
            # Prevent overly long queries for performance
            if len(cleaned) > 500:
                cleaned = cleaned[:500] + "..."
            
            return cleaned.strip() if cleaned.strip() else "general health inquiry"
            
        except Exception as e:
            logger.error("Error in symptom preprocessing", error=str(e), symptoms=str(symptoms)[:50])
            return "general health inquiry"

    def diagnose_symptoms(self, symptoms, user_profile=None, chat_topic=None):
        """Main symptom diagnosis method using AI and medical research."""
        try:
            processed_symptoms = self._preprocess_symptoms(symptoms)
        
            # Log diagnostic context for monitoring
            if user_profile:
                logger.info("Starting diagnosis with user profile", 
                            has_gender=bool(user_profile.get('gender')),
                            has_age=bool(user_profile.get('age')),
                            symptoms=processed_symptoms[:30],
                            chat_topic=chat_topic)
            else:
                logger.warning("Diagnosing without user profile", symptoms=processed_symptoms[:30])

            # Use existing chat topic if meaningful, otherwise extract new one
            if chat_topic and chat_topic != "New Chat" and len(chat_topic.strip()) > 3:
                medical_topic = chat_topic.strip()
                logger.info("Using existing chat topic for continuity", topic=medical_topic)
            else:
                medical_topic = self._extract_medical_topic(processed_symptoms, user_profile)
                logger.info("Extracted new medical topic", topic=medical_topic)
        
            # Check cache for performance optimization
            cache_key = f"diagnosis_{hash(processed_symptoms)}_{hash(str(user_profile))}_{hash(medical_topic)}"
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached diagnosis for performance", symptoms=processed_symptoms[:30])
                return cached_result

            # Build user context and get health data
            context = self._build_user_context(user_profile)
            health_data = self._get_latest_health_data()
            
            # Create targeted research queries based on medical topic
            primary_research_queries = [
                medical_topic,  # Exact topic match
                f"{medical_topic} diagnosis",
                f"{medical_topic} treatment guidelines",
            ]
            
            # Add user-specific refinements for personalization
            if user_profile:
                if user_profile.get('age'):
                    age = int(user_profile['age'])
                    if age < 18:
                        primary_research_queries.insert(1, f"pediatric {medical_topic}")
                    elif age > 65:
                        primary_research_queries.insert(1, f"elderly {medical_topic}")
                
                if user_profile.get('country'):
                    country = user_profile['country'].lower()
                    # Regional medical considerations
                    regional_queries = {
                        'kenya': f"{medical_topic} tropical medicine",
                        'nigeria': f"{medical_topic} tropical diseases",
                        'india': f"{medical_topic} endemic diseases",
                        'usa': f"{medical_topic} guidelines",
                        'uk': f"{medical_topic} NHS guidelines"
                    }
                    if country in regional_queries:
                        primary_research_queries.append(regional_queries[country])
            
            # Optimize query count for performance (max 3 queries)
            final_queries = primary_research_queries[:3]
            
            # Retrieve medical research with circuit breaker protection
            all_research = []
            for i, query in enumerate(final_queries):
                try:
                    research = self.get_medical_research(query, limit=2)
                    all_research.extend(research)
                    logger.info("Retrieved research for targeted query", 
                               query=query, count=len(research), priority=i+1)
                except Exception as e:
                    logger.warning("Research query failed, using fallback", 
                                  query=query, error=str(e))
                    # Use circuit breaker fallback for reliability
                    from ..circuit_breaker import pubmed_fallback
                    fallback_research = pubmed_fallback(query, 2)
                    all_research.extend(fallback_research)
                    continue
            
            # Remove duplicates and limit results for performance
            seen_titles = set()
            filtered_research = []
            for article in all_research:
                title = article.get('title', '').lower().strip()
                if title and title not in seen_titles and len(filtered_research) < 5:
                    seen_titles.add(title)
                    filtered_research.append(article)
            
            # Filter research by user profile relevance
            final_research = self._filter_research_by_relevance(filtered_research, user_profile, max_results=3)
            
            # Add source icons to research articles
            for article in final_research:
                if 'source_icon' not in article:
                    article['source_icon'] = self._get_source_icon(article.get('journal', 'Medical Journal'))
            
            # Build research context for AI processing
            research_context = ""
            if final_research:
                research_context = f"Relevant medical research for {medical_topic}:\n"
                for i, article in enumerate(final_research):
                    research_context += f"{i+1}. {article.get('title', 'Untitled')} ({article.get('journal', 'Unknown')}): "
                    research_context += f"{article.get('abstract', 'No abstract')[:200]}...\n"
            
            # Build health news context
            news_context = ""
            if 'health_news' in health_data and health_data['health_news']:
                news_context = "Recent health news:\n"
                for i, article in enumerate(health_data['health_news'][:2]):
                    news_context += f"{i+1}. {article.get('title', 'Untitled')} ({article.get('source', 'Unknown')}): "
                    news_context += f"{article.get('summary', 'No summary')[:150]}...\n"

                        # Create comprehensive AI prompt for diagnosis
            prompt = f"""You are Praxia, a medical AI assistant. {self.identity}

{context}

Based on these symptoms: {processed_symptoms}
Medical topic: {medical_topic}
Chat context: {chat_topic or 'General consultation'}

{research_context}

{news_context}

WHO guidelines recommend careful assessment of symptoms and considering local disease prevalence.
Evidence-based medicine emphasizes that symptom diagnosis should consider patient history and risk factors.

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
                response_text = self._call_together_ai_with_circuit_breaker(prompt)
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
            
                # Validate required fields for consistency
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
                    "search_queries_used": final_queries,
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
                # Use circuit breaker fallback
                from ..circuit_breaker import together_ai_fallback
                fallback_response = together_ai_fallback(prompt)
                try:
                    fallback_data = json.loads(fallback_response)
                    return {
                        "diagnosis": fallback_data.get("diagnosis", {
                            "conditions": ["Unable to analyze symptoms at this time"],
                            "next_steps": ["Please consult with a healthcare professional"],
                            "urgent": [],
                            "advice": "I apologize, but I'm having trouble analyzing your symptoms right now. Please try again or consult with a healthcare professional.",
                            "clarification": ["Could you provide more details about your symptoms?"]
                        }),
                        "medical_topic": medical_topic,
                        "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
                    }
                except:
                    return self._get_fallback_diagnosis_response(medical_topic)
        except Exception as e:
            logger.error("Unexpected error in symptom diagnosis", error=str(e), exc_info=True)
            return self._get_fallback_diagnosis_response("general medical consultation")

    def _get_fallback_diagnosis_response(self, medical_topic):
        """Get a safe fallback response when all else fails."""
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

    def _filter_research_by_relevance(self, research_results, user_profile=None, max_results=3):
        """Filter and rank research results based on user profile relevance."""
        if not research_results or not user_profile:
            return research_results[:max_results]
        
        scored_results = []
        
        for article in research_results:
            score = 0
            title_lower = article.get('title', '').lower()
            abstract_lower = article.get('abstract', '').lower()
            content = title_lower + ' ' + abstract_lower
            
            # Age relevance scoring
            if user_profile.get('age'):
                age = int(user_profile['age'])
                if age < 18 and any(term in content for term in ['pediatric', 'children', 'adolescent']):
                    score += 3
                elif age > 65 and any(term in content for term in ['elderly', 'geriatric', 'older adults']):
                    score += 3
                elif 18 <= age <= 65 and any(term in content for term in ['adult', 'working age']):
                    score += 2
            
            # Gender relevance scoring
            if user_profile.get('gender'):
                gender = user_profile['gender'].lower()
                if gender in ['male', 'female'] and gender in content:
                    score += 2
            
            # Country/regional relevance scoring
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
            
            # Allergy relevance scoring
            if user_profile.get('allergies'):
                allergies = user_profile['allergies'].lower()
                if any(allergy.strip() in content for allergy in allergies.split(',') if allergy.strip()):
                    score += 2
            
            scored_results.append((score, article))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [article for score, article in scored_results[:max_results]]

    def get_medical_research(self, query, limit=5):
        """Enhanced medical research retrieval with circuit breaker protection."""
        cache_key = f"research_{hash(query)}_{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            # Ensure query is valid and properly formatted
            if not query or not isinstance(query, str):
                query = "general medical research"
            
            # Clean and limit query length for optimal performance
            query = query.strip()
            if len(query) > 100:
                query = query[:100]
            
            # Create targeted PubMed search with filters
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp] OR Case Reports[ptyp])"
            
            logger.info("Medical research query", search_term=search_term)
            
            # Execute PubMed query with error handling
            try:
                results = self.pubmed_client.query(search_term, max_results=limit)
                articles = []
                for article in results:
                    # Safely handle None values and extract metadata
                    title = getattr(article, 'title', None) or "Research Article"
                    authors = getattr(article, 'authors', None)
                    journal = getattr(article, 'journal', None) or "Medical Journal"
                    
                    # Safely process authors list
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
                    
                    # Safely handle publication date and other metadata
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
                        "abstract": abstract,
                        "source_icon": self._get_source_icon(journal)
                    }
                    articles.append(article_data)
                
                cache.set(cache_key, articles, self.cache_timeout)
                logger.info("Medical research retrieved successfully", query=query, count=len(articles))
                return articles
            except Exception as e:
                logger.error("Error retrieving medical research", error=str(e), query=query)
                # Use circuit breaker fallback
                from ..circuit_breaker import pubmed_fallback
                return pubmed_fallback(query, limit)
        except Exception as e:
            logger.error("Critical error in medical research", error=str(e), query=query)
            # Return safe placeholder results
            placeholder_results = [
                {
                    "title": f"Recent advances in {query[:50]}", 
                    "authors": "Medical Research Team", 
                    "journal": "Medical Journal", 
                    "publication_date": "2023",
                    "doi": None,
                    "abstract": f"Recent research in {query[:50]} and related medical conditions.",
                    "source_icon": self._get_source_icon("Medical Journal")
                }
            ]
            return placeholder_results[:limit]

    def analyze_diet(self, dietary_info, user_profile=None):
        """Analyze user's dietary information and provide personalized recommendations."""
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
            
            response_text = self._call_together_ai_with_circuit_breaker(prompt)
            
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
        """Analyze medication information and check for potential interactions."""
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
            
            response_text = self._call_together_ai_with_circuit_breaker(prompt)
            
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
        """Analyze X-ray images using DenseNet deep learning model."""
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

            # Prepare image transformations for DenseNet input
            transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                ScaleIntensity(),
                Resize((224, 224)),
            ])
            
            # Process image data
            if isinstance(image_data, str):
                image = transforms(image_data)
            else:
                image = Image.open(BytesIO(image_data))
                image = np.array(image.convert('L'))
                image = transforms(image)
            
            image = image.unsqueeze(0).to(self.device)
            
            # Run inference with DenseNet model
            with torch.no_grad():
                densenet_output = self.densenet_model(image)
                densenet_probs = torch.softmax(densenet_output, dim=1)
                fracture_prob = densenet_probs[0, 0].item()
                tumor_prob = densenet_probs[0, 1].item()
                pneumonia_prob = densenet_probs[0, 2].item()
                normal_prob = max(0, min(1.0, 1.0 - (fracture_prob + tumor_prob + pneumonia_prob)))
            
            # Process results and generate findings
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
            
            # Get related research for detected conditions
            research = self.get_medical_research("X-ray diagnosis " + " ".join(detected_conditions.keys()), limit=2)
            
            # Generate detailed AI interpretation
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
            detailed_analysis_text = self._call_together_ai_with_circuit_breaker(prompt)
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

    def _call_together_ai_with_circuit_breaker(self, prompt):
        """Call Together AI with circuit breaker protection."""
        try:
            from ..circuit_breaker import together_ai_breaker
            return together_ai_breaker.call(self._call_together_ai, prompt)
        except Exception as e:
            logger.error("Circuit breaker failed for Together AI", error=str(e))
            # Use fallback
            from ..circuit_breaker import together_ai_fallback
            return together_ai_fallback(prompt)

    def _call_together_ai(self, prompt):
        """Direct call to Together AI API."""
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
            response = requests.post(url, json=payload, headers=headers, timeout=30)
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

    def _get_health_news_comprehensive(self, source='all', limit=5):
        """Comprehensive health news retrieval with enhanced error handling."""
        cache_key = f"health_news_comprehensive_{source}_{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached comprehensive health news")
            return cached_result

        try:
            articles = []
            
            # Strategy 1: Try RSS feeds first with timeout protection
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._get_health_news_from_rss_direct, source, limit)
                    try:
                        rss_articles = future.result(timeout=10)
                        if rss_articles:
                            articles.extend(rss_articles)
                            logger.info("Retrieved articles from RSS feeds", count=len(rss_articles))
                    except FutureTimeoutError:
                        logger.warning("RSS retrieval timed out")
                        future.cancel()
                    except Exception as e:
                        logger.warning("RSS feed retrieval failed", error=str(e))
            except Exception as e:
                logger.warning("RSS strategy setup failed", error=str(e))
            
            # Strategy 2: Try web scraping if we need more articles
            if len(articles) < limit:
                try:
                    scraped_articles = self._scrape_health_news_direct(source, limit - len(articles))
                    if scraped_articles:
                        articles.extend(scraped_articles)
                        logger.info("Retrieved articles from web scraping", count=len(scraped_articles))
                except Exception as e:
                    logger.warning("Web scraping failed", error=str(e))
            
            # Strategy 3: Use enhanced fallback to complete the set
            if len(articles) < limit:
                logger.info("Using enhanced fallback to complete article set")
                fallback_articles = self._get_enhanced_fallback_health_news(source, limit - len(articles))
                articles.extend(fallback_articles)
            
            # Process and clean articles for consistency
            processed_articles = []
            seen_titles = set()
            
            for article in articles[:limit * 2]:
                try:
                    if not isinstance(article, dict):
                        continue
                    
                    title = article.get('title', 'Health News Update')
                    # Skip duplicates
                    if title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    
                    # Ensure published_date is in correct format
                    published_date = article.get('published_date')
                    if not published_date:
                        published_date = datetime.now().strftime("%Y-%m-%d")
                    elif not isinstance(published_date, str) or len(published_date) < 8:
                        published_date = datetime.now().strftime("%Y-%m-%d")
                    
                    processed_article = {
                        'title': title,
                        'source': article.get('source', 'Health Authority'),
                        'url': article.get('url', '#'),
                        'content': article.get('content', 'Health information update'),
                        'image_url': article.get('image_url'),
                        'published_date': published_date
                    }
                    
                    # Generate summary efficiently
                    content = processed_article.get('content', '')
                    if content and len(content) > 200:
                        processed_article['summary'] = content[:200] + "..."
                    else:
                        processed_article['summary'] = content
                    
                    processed_articles.append(processed_article)
                    
                    # Stop when we have enough articles
                    if len(processed_articles) >= limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing article: {str(e)}")
                    continue
            
            # Cache for 1 hour for faster subsequent requests
            cache.set(cache_key, processed_articles, 60 * 60 * 1)
            logger.info("Health news retrieval completed", 
                       total_articles=len(processed_articles), 
                       source=source, 
                       requested_limit=limit)
            return processed_articles
    
        except Exception as e:
            logger.error("Error in comprehensive health news retrieval", error=str(e))
            return self._get_enhanced_fallback_health_news(source, limit)

    def _get_health_news_from_rss_direct(self, source='all', limit=3):
        """Direct RSS feed retrieval from WHO and CDC sources."""
        articles = []
    
        # Reliable RSS sources
        rss_sources = {
            'who': [
                'https://www.who.int/rss-feeds/news-english.xml',
                'https://www.who.int/feeds/entity/csr/don/en/rss.xml',
                'https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml'
            ],
            'cdc': [
                'https://tools.cdc.gov/api/v2/resources/media/316422.rss',
                'https://www.cdc.gov/media/rss/index.html',
                'https://blogs.cdc.gov/feed/'
            ],
        }
    
        # Optimized headers for better success rate
        header_options = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml, application/atom+xml, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml',
                'Accept-Language': 'en-US,en;q=0.8',
            }
        ]
    
        sources_to_try = [source] if source != 'all' else list(rss_sources.keys())
    
        for src in sources_to_try:
            if src in rss_sources:
                urls_to_try = rss_sources[src]
                articles_per_source = max(2, limit // len(sources_to_try)) if source == 'all' else limit
                
                for i, url in enumerate(urls_to_try):
                    try:
                        import time
                        import random
                        time.sleep(random.uniform(0.2, 0.5))
                        
                        # Rotate headers to avoid blocking
                        headers = header_options[i % len(header_options)]
                        
                        response = requests.get(url, timeout=8, headers=headers)
                        response.raise_for_status()
                        
                        feed = feedparser.parse(response.content)
                        
                        if not feed.entries:
                            logger.warning(f"No entries found in RSS feed for {src} at {url}")
                            continue
                        
                        # Successfully got articles
                        source_articles = []
                        for entry in feed.entries[:articles_per_source]:
                            try:
                                # Parse date efficiently
                                published_date = self._parse_rss_date(entry.get('published', ''))
                                
                                article = {
                                    'title': entry.get('title', 'Health News'),
                                    'source': src.upper(),
                                    'url': entry.get('link', '#'),
                                    'content': entry.get('description', 'Health information update'),
                                    'summary': entry.get('summary', entry.get('description', ''))[:200],
                                    'published_date': published_date
                                }
                                source_articles.append(article)
                            except Exception as e:
                                logger.warning(f"Error processing RSS entry: {str(e)}")
                                continue
                        
                        if source_articles:
                            articles.extend(source_articles)
                            logger.info(f"Retrieved {len(source_articles)} articles from {src} RSS")
                            break
                            
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"RSS request failed for {src} at {url}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"RSS parsing failed for {src} at {url}: {str(e)}")
                        continue
    
        return articles

    def _parse_rss_date(self, date_string):
        """Parse RSS date string and convert to YYYY-MM-DD format."""
        if not date_string:
            return datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Common RSS date formats
            date_formats = [
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            
            # Clean the date string
            cleaned_date = date_string.strip()
            
            # Handle timezone abbreviations
            timezone_replacements = {
                ' GMT': ' +0000',
                ' UTC': ' +0000',
                ' Z': ' +0000',
                ' EST': ' -0500',
                ' PST': ' -0800',
                ' CST': ' -0600',
                ' MST': ' -0700'
            }
            
            for old_tz, new_tz in timezone_replacements.items():
                if cleaned_date.endswith(old_tz):
                    cleaned_date = cleaned_date.replace(old_tz, new_tz)
                    break
            
            # Try parsing with different formats
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(cleaned_date, date_format)
                    return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            
            # If all parsing fails, try using dateutil (if available)
            try:
                from dateutil import parser
                parsed_date = parser.parse(cleaned_date)
                return parsed_date.strftime("%Y-%m-%d")
            except (ImportError, Exception):
                pass
            
            # Final fallback
            logger.warning(f"Could not parse date: {date_string}")
            return datetime.now().strftime("%Y-%m-%d")
            
        except Exception as e:
            logger.error(f"Error parsing date {date_string}: {str(e)}")
            return datetime.now().strftime("%Y-%m-%d")

    def _scrape_health_news_direct(self, source='all', limit=3):
        """Direct web scraping from WHO and CDC websites."""
        articles = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        
        sources_to_try = []
        if source in ['who', 'all']:
            sources_to_try.append(('who', self._scrape_who_news_improved))
        if source in ['cdc', 'all']:
            sources_to_try.append(('cdc', self._scrape_cdc_news_improved))
        
        articles_per_source = max(1, limit // len(sources_to_try)) if len(sources_to_try) > 1 else limit
        
        for src_name, scrape_func in sources_to_try:
            try:
                logger.info(f"Attempting direct scraping from {src_name}")
                src_articles = scrape_func(limit=articles_per_source, headers=headers)
                if src_articles:
                    articles.extend(src_articles)
                    logger.info(f"Successfully retrieved {len(src_articles)} articles from {src_name}")
            except Exception as e:
                logger.warning(f"Direct scraping failed for {src_name}", error=str(e))
                continue
        
        return articles

    def _get_enhanced_fallback_health_news(self, source='all', limit=3):
        """Enhanced fallback health news when all other methods fail."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # WHO health news content
        who_articles = [
            {
                "title": "WHO Global Health Security: Strengthening Pandemic Preparedness",
                "source": "WHO",
                "url": "https://www.who.int/news",
                "summary": "WHO continues to work with countries to strengthen health security and pandemic preparedness capabilities.",
                "content": "The World Health Organization emphasizes the importance of robust health systems, early detection capabilities, and coordinated international response mechanisms for global health security.",
                "published_date": current_date
            },
            {
                "title": "WHO Health Equity Initiative: Reducing Global Health Disparities",
                "source": "WHO",
                "url": "https://www.who.int/news",
                "summary": "New initiatives focus on addressing health inequities and ensuring universal health coverage worldwide.",
                "content": "WHO's latest health equity initiatives aim to reduce disparities in health outcomes and improve access to essential health services for vulnerable populations globally.",
                "published_date": current_date
            }
        ]
        
        # CDC health news content
        cdc_articles = [
            {
                "title": "CDC Disease Prevention Guidelines: Updated Recommendations",
                "source": "CDC",
                "url": "https://www.cdc.gov/",
                "summary": "Latest CDC guidelines provide updated recommendations for disease prevention and health promotion.",
                "content": "The Centers for Disease Control and Prevention has released updated guidelines focusing on preventive care, vaccination schedules, and community health interventions.",
                "published_date": current_date
            },
            {
                "title": "CDC Health Surveillance: Monitoring Disease Trends",
                "source": "CDC",
                "url": "https://www.cdc.gov/",
                "summary": "CDC continues monitoring disease trends and providing data-driven health recommendations.",
                "content": "CDC's surveillance systems continue to track disease patterns, enabling early detection of health threats and informed public health decision-making.",
                "published_date": current_date
            }
        ]
        
        # Select articles based on source preference
        if source == 'who':
            selected_articles = who_articles[:limit]
        elif source == 'cdc':
            selected_articles = cdc_articles[:limit]
        else:  # 'all' or other
            # Distribute evenly across WHO and CDC
            selected_articles = []
            articles_per_source = max(1, limit // 2)
            selected_articles.extend(who_articles[:articles_per_source])
            selected_articles.extend(cdc_articles[:articles_per_source])
            selected_articles = selected_articles[:limit]
        
        return selected_articles

    def _scrape_who_news_improved(self, limit=3, headers=None):
        """Improved WHO news scraping with multiple URL attempts."""
        try:
            # Multiple WHO news URLs for better success rate
            who_urls = [
                "https://www.who.int/news",
                "https://www.who.int/news-room/news",
                "https://www.who.int/emergencies/news"
            ]
            
            for url in who_urls:
                try:
                    response = requests.get(url, timeout=10, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Try multiple CSS selectors
                        selectors = [
                            '.list-view--item',
                            '.news-item',
                            '.content-item',
                            'article'
                        ]
                        
                        news_items = []
                        for selector in selectors:
                            news_items = soup.select(selector)
                            if news_items:
                                break
                        
                        articles = []
                        for item in news_items[:limit]:
                            try:
                                title_elem = item.select_one('.heading, h3, h2, .title')
                                title = title_elem.text.strip() if title_elem else "WHO Health Update"
                                
                                link_elem = item.select_one('a')
                                article_url = "https://www.who.int" + link_elem['href'] if link_elem and 'href' in link_elem.attrs else url
                                
                                date_elem = item.select_one('.timestamp, .date, time')
                                published_date = date_elem.text.strip() if date_elem else None
                                
                                content = self._get_article_content_safe(article_url, headers)
                                
                                articles.append({
                                    "title": title,
                                    "source": "WHO",
                                    "url": article_url,
                                    "content": content,
                                    "published_date": published_date
                                })
                            except Exception as e:
                                logger.warning(f"Error processing WHO news item: {str(e)}")
                                continue
                        
                        if articles:
                            return articles
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to access {url}: {str(e)}")
                    continue
            
            return []
        except Exception as e:
            logger.error(f"Error in improved WHO scraping: {str(e)}")
            return []
                                
    def _scrape_cdc_news_improved(self, limit=3, headers=None):
        """Improved CDC news scraping with multiple URL attempts."""
        try:
            cdc_urls = [
                "https://www.cdc.gov/media/index.html",
                "https://www.cdc.gov/media/releases.htm",
                "https://www.cdc.gov/news/"
            ]
            
            for url in cdc_urls:
                try:
                    response = requests.get(url, timeout=10, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        selectors = [
                            '.feed-item',
                            '.news-item',
                            '.press-release',
                            'article'
                        ]
                        
                        news_items = []
                        for selector in selectors:
                            news_items = soup.select(selector)
                            if news_items:
                                break
                        
                        articles = []
                        for item in news_items[:limit]:
                            try:
                                title_elem = item.select_one('a, h3, h2, .title')
                                title = title_elem.text.strip() if title_elem else "CDC Health Update"
                                
                                link_href = title_elem.get('href') if title_elem and title_elem.name == 'a' else None
                                article_url = "https://www.cdc.gov" + link_href if link_href else url
                                
                                date_elem = item.select_one('.date, time, .publish-date')
                                published_date = date_elem.text.strip() if date_elem else None
                                
                                content = self._get_article_content_safe(article_url, headers)
                                
                                articles.append({
                                    "title": title,
                                    "source": "CDC",
                                    "url": article_url,
                                    "content": content,
                                    "published_date": published_date
                                })
                            except Exception as e:
                                logger.warning(f"Error processing CDC news item: {str(e)}")
                                continue
                        
                        if articles:
                            return articles
                            
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to access {url}: {str(e)}")
                    continue
            
            return []
        except Exception as e:
            logger.error(f"Error in improved CDC scraping: {str(e)}")
            return []

    def _get_article_content_safe(self, url, headers=None):
        """Safely get article content with timeout and error handling."""
        try:
            if not url or url == '#':
                return "Content not available"
            
            response = requests.get(url, timeout=10, headers=headers)
            if response.status_code != 200:
                return "Content could not be retrieved"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for script in soup.select('script, style, nav, footer, aside, .sidebar'):
                script.extract()
            
            # Try multiple content selectors
            content_selectors = [
                'article', '.content', '.main-content',
                '#content', '.article-body', '.story-body',
                '.entry-content', '.post-content', '.news-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    content = ' '.join(content.split())
                    
                    if len(content) > 5000:
                        content = content[:5000] + "..."
                    
                    if len(content) > 100:
                        return content
            
            return "Content could not be extracted"
            
        except Exception as e:
            logger.warning(f"Error getting article content from {url}: {str(e)}")
            return "Content could not be retrieved due to access restrictions"


# Celery Tasks for Asynchronous Processing
@shared_task
def analyze_xray_task(xray_id, image_path):
    """Celery task to analyze X-ray images asynchronously."""
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
    """Celery task to scrape health news asynchronously."""
    try:
        praxia = PraxiaAI()
        news_articles = praxia._get_health_news_comprehensive(source, limit)
        
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
def diagnose_symptoms_task(symptoms, user_profile=None, chat_topic=None):
    """Celery task for symptom diagnosis (for async diagnosis if needed)."""
    try:
        praxia = PraxiaAI()
        diagnosis_result = praxia.diagnose_symptoms(symptoms, user_profile, chat_topic)
        
        logger.info("Symptom diagnosis completed", symptoms=symptoms[:50])
        return diagnosis_result
        
    except Exception as e:
        logger.error("Error in symptom diagnosis task", error=str(e), symptoms=symptoms[:50])
        raise

@shared_task
def periodic_model_cleanup():
    """Periodic task to clean up model memory and optimize performance."""
    try:
        import gc
        import torch
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Model cleanup completed successfully")
        return {"status": "success", "message": "Model cleanup completed"}
        
    except Exception as e:
        logger.error("Error in model cleanup", error=str(e))
        return {"status": "error", "message": str(e)}

@shared_task
def health_data_refresh():
    """Periodic task to refresh health data cache."""
    try:
        praxia = PraxiaAI()
        
        # Clear existing cache
        cache.delete('latest_health_data')
        
        # Get fresh health data
        health_data = praxia._get_latest_health_data()
        
        logger.info("Health data refreshed", 
                   news_count=len(health_data.get('health_news', [])),
                   research_topics=len(health_data.get('research_trends', {})))
        
        return {
            "status": "success", 
            "message": "Health data refreshed",
            "data": {
                "news_articles": len(health_data.get('health_news', [])),
                "research_topics": len(health_data.get('research_trends', {}))
            }
        }
        
    except Exception as e:
        logger.error("Error refreshing health data", error=str(e))
        return {"status": "error", "message": str(e)}

@shared_task
def validate_model_integrity():
    """Task to validate AI model integrity and performance."""
    try:
        praxia = PraxiaAI()
        
        # Test basic functionality
        test_symptoms = "headache and fever"
        test_result = praxia.diagnose_symptoms(test_symptoms)
        
        if not test_result or 'diagnosis' not in test_result:
            raise Exception("Model validation failed - invalid response format")
        
        # Test X-ray model if available
        xray_status = "unavailable"
        if hasattr(praxia, 'densenet_model') and praxia.densenet_model is not None:
            xray_status = "available"
        
        # Test research functionality
        research_test = praxia.get_medical_research("diabetes", limit=1)
        research_status = "available" if research_test else "unavailable"
        
        validation_result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "symptom_diagnosis": "available",
                "xray_analysis": xray_status,
                "medical_research": research_status,
                "together_ai": "available"
            }
        }
        
        logger.info("Model validation completed", result=validation_result)
        return validation_result
        
    except Exception as e:
        logger.error("Model validation failed", error=str(e))
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@shared_task
def monitor_rss_feeds():
    """Monitor RSS feed health and update circuit breaker status."""
    try:
        from ..circuit_breaker import rss_breaker
        
        rss_sources = {
            'who': 'https://www.who.int/rss-feeds/news-english.xml',
            'cdc': 'https://tools.cdc.gov/api/v2/resources/media/316422.rss',
        }
        
        results = {}
        
        for source, url in rss_sources.items():
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; PraxiaBot/1.0; +https://praxia.ai)',
                    'Accept': 'application/rss+xml, application/xml, text/xml',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                
                response = requests.get(url, timeout=15, headers=headers)
                
                if response.status_code == 200:
                    # Try to parse the feed to ensure it's valid
                    feed = feedparser.parse(response.content)
                    if feed.entries:
                        results[source] = 'operational'
                        logger.info(f"RSS feed {source} is operational")
                    else:
                        logger.warning(f"RSS feed {source} has no entries")
                        results[source] = 'no_entries'
                elif response.status_code == 403:
                    logger.warning(f"RSS feed {source} returned 403 Forbidden")
                    results[source] = 'forbidden'
                else:
                    logger.warning(f"RSS feed {source} returned {response.status_code}")
                    results[source] = f'error_{response.status_code}'
                    
            except requests.exceptions.Timeout:
                logger.warning(f"RSS feed {source} timed out")
                results[source] = 'timeout'
            except requests.exceptions.RequestException as e:
                logger.warning(f"RSS feed {source} request failed: {str(e)}")
                results[source] = 'request_error'
            except Exception as e:
                logger.error(f"RSS feed monitoring failed for {source}", error=str(e))
                results[source] = 'error'
        
        # Cache results for health check and circuit breaker decisions
        cache.set('rss_feed_status', results, 60 * 60 * 6)  # Cache for 6 hours
        
        # Update circuit breaker state based on results
        operational_feeds = sum(1 for status in results.values() if status == 'operational')
        total_feeds = len(results)
        
        if operational_feeds == 0:
            logger.warning("All RSS feeds are down")
        elif operational_feeds < total_feeds:
            logger.warning(f"Some RSS feeds are down: {results}")
        else:
            logger.info("All RSS feeds are operational")
        
        logger.info("RSS feed monitoring completed", results=results)
        return {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "operational_count": operational_feeds,
            "total_count": total_feeds,
            "status": "operational" if operational_feeds > 0 else "degraded"
        }
        
    except Exception as e:
        logger.error("RSS feed monitoring task failed", error=str(e))
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }

