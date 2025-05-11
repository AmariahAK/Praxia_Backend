import requests
import json
from django.conf import settings
from django.core.cache import cache
import structlog

logger = structlog.get_logger(__name__)

class TranslationService:
    """Service for translating text using LibreTranslate"""
    
    def __init__(self):
        self.base_url = settings.LIBRETRANSLATE_URL
        self.cache_timeout = 60 * 60 * 24  # 24 hours cache
        
        # Load medical terms dictionary
        self.medical_terms = {
            'sw': self._load_medical_terms('sw'),  # Swahili
            'es': self._load_medical_terms('es'),  # Spanish
        }
    
    def _load_medical_terms(self, language_code):
        """Load medical terms dictionary for a specific language"""
        try:
            import os
            from django.conf import settings
            
            file_path = os.path.join(settings.BASE_DIR, 'data', f'medical_terms_{language_code}.json')
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading medical terms for {language_code}", error=str(e))
            return {}
    
    def _preprocess_text(self, text, source_lang):
        """Replace known medical terms before translation"""
        if source_lang not in self.medical_terms:
            return text
            
        processed_text = text
        for term, translation in self.medical_terms[source_lang].items():
            processed_text = processed_text.replace(term, translation)
            
        return processed_text
    
    def translate(self, text, source_lang, target_lang):
        """
        Translate text from source language to target language
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code (e.g., 'sw', 'es', 'en')
            target_lang (str): Target language code (e.g., 'en', 'sw', 'es')
            
        Returns:
            str: Translated text
        """
        if source_lang == target_lang:
            return text
            
        # Generate cache key
        cache_key = f"translation_{source_lang}_{target_lang}_{hash(text)}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Using cached translation", source=source_lang, target=target_lang)
            return cached_result
            
        # Preprocess text with medical terms dictionary
        preprocessed_text = self._preprocess_text(text, source_lang)
        
        try:
            # Call LibreTranslate API
            response = requests.post(
                f"{self.base_url}/translate",
                json={
                    "q": preprocessed_text,
                    "source": source_lang,
                    "target": target_lang,
                    "format": "text"
                },
                timeout=5
            )
            
            response.raise_for_status()
            translated_text = response.json()["translatedText"]
            
            # Cache the result
            cache.set(cache_key, translated_text, self.cache_timeout)
            
            logger.info("Translation successful", source=source_lang, target=target_lang)
            return translated_text
            
        except Exception as e:
            logger.error("Translation failed", error=str(e), source=source_lang, target=target_lang)
            return text  # Return original text on failure
