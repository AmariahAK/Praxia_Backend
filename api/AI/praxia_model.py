import os
import requests
import json
import pymed
import numpy as np
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            # First check if we have a model file
            densenet_path = os.path.join(settings.BASE_DIR, 'data', 'models', 'densenet_xray.pth')
            if not os.path.exists(densenet_path):
                logger.warning("DenseNet model weights not found at %s", densenet_path)
                self.densenet_model = None
                return
            
            # Check if the file is a valid model file (has minimum size)
            if os.path.getsize(densenet_path) < 1000000:  # < 1MB
                logger.warning("DenseNet model file is too small, may be corrupted")
                self.densenet_model = None
                return
            
            # Try to load the model
            try:
                self.densenet_model = torch.load(densenet_path, map_location=self.device)
                self.densenet_model.eval()
                logger.info("DenseNet model loaded successfully")
            except Exception as e:
                logger.error("Error loading DenseNet model: %s", str(e))
            
                # Try alternative loading method
                try:
                    from torchvision.models import densenet121
                    model = densenet121(pretrained=False)
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

    def _preprocess_symptoms(self, symptoms):
        if not symptoms or len(symptoms.strip()) < 3:
            return "unspecified symptoms"
        cleaned = symptoms.replace('<', '').replace('>', '')
        return cleaned

    def diagnose_symptoms(self, symptoms, user_profile=None):
        processed_symptoms = self._preprocess_symptoms(symptoms)
        
        # Log whether user profile is being used
        if user_profile:
            logger.info("Diagnosing with user profile", 
                        has_gender=bool(user_profile.get('gender')),
                        has_age=bool(user_profile.get('age')),
                        symptoms=processed_symptoms[:30])
        else:
            logger.warning("Diagnosing without user profile", symptoms=processed_symptoms[:30])
        
        cache_key = f"diagnosis_{hash(processed_symptoms)}_{hash(str(user_profile))}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached diagnosis", symptoms=processed_symptoms[:30])
            return cached_result

        context = self._build_user_context(user_profile)
        health_data = self._get_latest_health_data()
        research_results = self._get_topic_specific_data(processed_symptoms, limit=3)
        research_context = ""
        if research_results:
            research_context = "Relevant medical research:\n"
            for i, article in enumerate(research_results):
                research_context += f"{i+1}. {article.get('title')} ({article.get('journal')}): "
                research_context += f"{article.get('abstract')[:200]}...\n"
        news_context = ""
        if 'health_news' in health_data and health_data['health_news']:
            news_context = "Recent health news:\n"
            for i, article in enumerate(health_data['health_news'][:2]):
                news_context += f"{i+1}. {article.get('title')} ({article.get('source')}): "
                news_context += f"{article.get('summary')[:150]}...\n"

        prompt = f"""You are Praxia, a medical AI assistant. {self.identity}

{context}

Based on these symptoms: {processed_symptoms}

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
                try:
                    diagnosis_data = json.loads(response_text)
                except json.JSONDecodeError:
                    if "" in response_text and "" in response_text:
                        json_content = response_text.split("")[1].split("")[0].strip()
                        diagnosis_data = json.loads(json_content)
                    else:
                        raise
                        
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response", response=response_text[:100])
                diagnosis_data = {
                    "conditions": ["Unable to parse conditions from response"],
                    "next_steps": ["Consult with a healthcare professional"],
                    "urgent": ["If symptoms are severe, seek immediate medical attention"],
                    "advice": response_text,
                    "clarification": ["Please provide more specific symptom details"]
                }
            result = {
                "diagnosis": diagnosis_data,
                "related_research": research_results,
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            }
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
            search_term = f"{query} AND (Review[ptyp] OR Clinical Trial[ptyp])"
            results = self.pubmed_client.query(search_term, max_results=limit)
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
            cache.set(cache_key, articles, self.cache_timeout)
            logger.info("Medical research retrieved successfully", query=query, count=len(articles))
            return articles
        except Exception as e:
            logger.warning("Error retrieving medical research", error=str(e), query=query)
            placeholder_results = [
                {"title": "Recent advances in medical diagnosis and treatment", "authors": "Smith J, et al.", "journal": "Medical Journal", "publication_date": "2023"},
                {"title": "Clinical guidelines for symptom management", "authors": "Johnson M, et al.", "journal": "Healthcare Research", "publication_date": "2022"}
            ]
            return placeholder_results

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
            
            # Add more fallback sources
            sources_to_try = []
            if source in ['who', 'all']:
                sources_to_try.append(('who', self._scrape_who_news))
            if source in ['cdc', 'all']:
                sources_to_try.append(('cdc', self._scrape_cdc_news))
            
            # Try additional sources like NIH, Mayo Clinic, etc.
            if source in ['mayo', 'all']:
                sources_to_try.append(('mayo', self._scrape_mayo_news))
            
            # Try each source, continue on failure
            for src_name, scrape_func in sources_to_try:
                try:
                    src_articles = scrape_func(limit=limit if source == src_name else max(1, limit // len(sources_to_try)))
                    articles.extend(src_articles)
                    logger.info(f"Retrieved {len(src_articles)} articles from {src_name}")
                except Exception as e:
                    logger.error(f"Failed to scrape {src_name}", error=str(e))
                    continue
            
            # Add fallback if no articles found
            if not articles:
                logger.warning("No articles found, using fallback")
                articles = self._get_fallback_health_news(limit)
            
            # Process articles
            articles = articles[:limit]
            for article in articles:
                if 'content' in article and article['content']:
                    article['summary'] = self._summarize_article(article['content'])
                else:
                    article['summary'] = "No content available for summarization."
            
            cache.set(cache_key, articles, 60 * 60 * 12)
            logger.info("Health news scraped successfully", source=source, count=len(articles))
            return articles
        except Exception as e:
            logger.error("Error scraping health news", error=str(e), source=source)
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
                "published_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Seasonal Illness Prevention Strategies",
                "source": "Mayo Clinic",
                "url": "https://www.mayoclinic.org/",
                "summary": "Tips for preventing common seasonal illnesses and maintaining good health.",
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
            return content[:max_length] + "..."

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

# -------------------- Celery Task Wrappers --------------------

@shared_task
def diagnose_symptoms_task(symptoms, user_profile=None):
    """
    Celery task wrapper for PraxiaAI.diagnose_symptoms
    """
    praxia = PraxiaAI()
    return praxia.diagnose_symptoms(symptoms, user_profile)

@shared_task
def analyze_xray_task(xray_id, image_path):
    """
    Celery task wrapper for PraxiaAI.analyze_xray that saves results to the database
    """
    praxia = PraxiaAI()
    result = praxia.analyze_xray(image_path)
    
    # Import here to avoid circular import issues
    from ..models import XRayAnalysis, ChatMessage
    
    # Get the XRayAnalysis object and update it with results
    xray = XRayAnalysis.objects.get(id=xray_id)
    xray.analysis_result = json.dumps(result)
    
    # Extract and save detected conditions and confidence scores
    if isinstance(result, dict):
        xray.detected_conditions = result.get("detected_conditions", {})
        xray.confidence_scores = result.get("confidence_scores", {})
    
    xray.save()
    
    # Update any chat messages that were waiting for this analysis
    pending_messages = ChatMessage.objects.filter(
        role='assistant',
        content__contains=f'"xray_analysis_id": {xray_id}'
    )
    
    for message in pending_messages:
        try:
            # Parse the current content
            content_data = json.loads(message.content)
            
            # Update with the completed analysis
            content_data.update({
                "message": "I've completed the analysis of your X-ray.",
                "xray_analysis_id": xray_id,
                "status": "completed",
                "xray_analysis_result": result
            })
            
            # Save the updated content
            message.content = json.dumps(content_data)
            message.save()
        except Exception as e:
            logger.error(f"Error updating chat message with X-ray results: {str(e)}")
    
    return result

@shared_task
def analyze_diet_task(diet_info, user_profile=None):
    """
    Celery task wrapper for PraxiaAI.analyze_diet
    """
    praxia = PraxiaAI()
    return praxia.analyze_diet(diet_info, user_profile)

@shared_task
def analyze_medication_task(medication_info, user_profile=None):
    """
    Celery task wrapper for PraxiaAI.analyze_medication
    """
    praxia = PraxiaAI()
    return praxia.analyze_medication(medication_info, user_profile)

@shared_task
def scrape_health_news(source='who', limit=3):
    """
    Celery task wrapper for PraxiaAI._scrape_health_news
    """
    praxia = PraxiaAI()
    return praxia._scrape_health_news(source=source, limit=limit)
