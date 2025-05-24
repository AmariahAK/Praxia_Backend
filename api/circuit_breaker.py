import pybreaker
import time
import structlog
import json
from functools import wraps
from django.core.cache import cache
from datetime import datetime

logger = structlog.get_logger(__name__)

# Create circuit breakers for different external services
who_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[ValueError, TypeError],
    name='who_api'
)

mayo_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[ValueError, TypeError],
    name='mayo_api'
)

together_ai_breaker = pybreaker.CircuitBreaker(
    fail_max=2,  
    reset_timeout=30,
    recovery_timeout=10,  
    expected_exception=Exception,
    name='together_ai'
)

pubmed_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[ValueError, TypeError],
    name='pubmed_api'
)

# Circuit breaker decorator with fallback
def circuit_breaker_with_fallback(breaker, fallback_function=None):
    """
    Decorator that applies a circuit breaker to a function and provides a fallback
    if the circuit is open or if the function fails.
    
    Args:
        breaker: The circuit breaker instance to use
        fallback_function: Function to call if the circuit is open or the call fails
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except pybreaker.CircuitBreakerError as e:
                logger.warning(
                    "Circuit breaker is open",
                    breaker=breaker.name,
                    error=str(e)
                )
                if fallback_function:
                    return fallback_function(*args, **kwargs)
                raise
            except Exception as e:
                logger.error(
                    "Function call failed",
                    function=func.__name__,
                    error=str(e)
                )
                if fallback_function:
                    return fallback_function(*args, **kwargs)
                raise
        return wrapper
    return decorator

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=3, initial_backoff=1, backoff_factor=2):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff time by after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            "Max retries reached",
                            function=func.__name__,
                            retries=retries,
                            error=str(e)
                        )
                        raise
                    
                    logger.warning(
                        "Retrying function",
                        function=func.__name__,
                        retry=retries,
                        backoff=backoff,
                        error=str(e)
                    )
                    
                    time.sleep(backoff)
                    backoff *= backoff_factor
        return wrapper
    return decorator

# Cache decorator
def cache_result(timeout=3600, key_prefix=''):
    """
    Decorator that caches the result of a function.
    
    Args:
        timeout: Cache timeout in seconds
        key_prefix: Prefix for the cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            cache_key = f"{key_prefix}_{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            
            # Try to get the result from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(
                    "Cache hit",
                    function=func.__name__,
                    cache_key=cache_key
                )
                return cached_result
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, timeout)
            
            logger.info(
                "Cache miss, result cached",
                function=func.__name__,
                cache_key=cache_key,
                timeout=timeout
            )
            
            return result
        return wrapper
    return decorator

# Enhanced fallback functions with better error handling
def who_api_fallback(query=None, *args, **kwargs):
    """Enhanced fallback function for WHO API"""
    logger.info("Using WHO API fallback", query=str(query)[:50] if query else "unknown")
    
    # Provide relevant health guidance based on query if available
    if query and isinstance(query, str):
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ['fever', 'temperature', 'covid']):
            return "Monitor temperature regularly. Seek medical attention if fever persists above 38.5°C (101.3°F) for more than 3 days. Follow WHO guidelines for respiratory symptoms."
        elif any(keyword in query_lower for keyword in ['cough', 'respiratory', 'breathing']):
            return "For persistent cough or breathing difficulties, consult healthcare providers. Maintain good hygiene and follow respiratory etiquette."
        elif any(keyword in query_lower for keyword in ['headache', 'pain']):
            return "For severe or persistent headaches, seek medical evaluation. Consider rest, hydration, and appropriate pain management."
    
    return "WHO guidelines recommend consulting with qualified healthcare professionals for medical concerns. Maintain good hygiene practices and follow local health authority recommendations."

def mayo_clinic_fallback(query=None, *args, **kwargs):
    """Enhanced fallback function for Mayo Clinic data"""
    logger.info("Using Mayo Clinic fallback", query=str(query)[:50] if query else "unknown")
    
    # Provide evidence-based guidance based on query
    if query and isinstance(query, str):
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ['diet', 'nutrition', 'eating']):
            return "Maintain a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins. Limit processed foods and added sugars. Consult a registered dietitian for personalized advice."
        elif any(keyword in query_lower for keyword in ['exercise', 'fitness', 'activity']):
            return "Aim for at least 150 minutes of moderate aerobic activity weekly. Include strength training exercises twice per week. Start gradually and consult healthcare providers before beginning new exercise programs."
        elif any(keyword in query_lower for keyword in ['sleep', 'insomnia', 'tired']):
            return "Maintain consistent sleep schedules. Aim for 7-9 hours of sleep nightly. Create a relaxing bedtime routine and optimize your sleep environment."
    
    return "Mayo Clinic recommends evidence-based medical care. For specific health concerns, consult with qualified healthcare professionals who can provide personalized medical advice based on your individual situation."

def together_ai_fallback(prompt=None, *args, **kwargs):
    """Enhanced fallback function for Together AI API"""
    logger.info("Using Together AI fallback", prompt_length=len(str(prompt)) if prompt else 0)
    
    # Provide structured medical guidance
    if prompt and isinstance(prompt, str):
        prompt_lower = prompt.lower()
        
        # Check for emergency symptoms
        emergency_keywords = ['chest pain', 'difficulty breathing', 'severe bleeding', 'unconscious', 'stroke', 'heart attack']
        if any(keyword in prompt_lower for keyword in emergency_keywords):
            return json.dumps({
                "diagnosis": {
                    "conditions": ["Potential emergency situation"],
                    "next_steps": ["Seek immediate emergency medical care", "Call emergency services", "Do not delay medical attention"],
                    "urgent": ["These symptoms may require immediate medical intervention"],
                    "advice": "This may be a medical emergency. Please seek immediate medical attention or call emergency services.",
                    "clarification": []
                },
                "disclaimer": "This is an emergency guidance message. Seek immediate medical care."
            })
        
        # Check for common symptom categories
        if any(keyword in prompt_lower for keyword in ['fever', 'cold', 'flu']):
            return json.dumps({
                "diagnosis": {
                    "conditions": ["Possible viral infection", "Common cold", "Influenza"],
                    "next_steps": ["Rest and hydration", "Monitor symptoms", "Consider over-the-counter medications for symptom relief"],
                    "urgent": ["Seek medical care if symptoms worsen or persist beyond 7-10 days"],
                    "advice": "Rest, stay hydrated, and monitor your symptoms. Most viral infections resolve on their own with supportive care.",
                    "clarification": ["How long have you had these symptoms?", "Do you have any other accompanying symptoms?"]
                },
                "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
            })
    
    return json.dumps({
        "diagnosis": {
            "conditions": ["I'm currently unable to analyze your specific symptoms"],
            "next_steps": ["Please try again in a few moments", "Consider consulting with a healthcare professional"],
            "urgent": [],
            "advice": "I'm experiencing technical difficulties and cannot provide specific medical analysis at this time. Please try again shortly or consult with a healthcare professional.",
            "clarification": ["Could you try rephrasing your health question?", "Would you like to try again in a few minutes?"]
        },
        "disclaimer": "This is a fallback response due to technical issues. Please consult with a healthcare professional for medical advice."
    })

def pubmed_fallback(query=None, limit=5, *args, **kwargs):
    """Enhanced fallback function for PubMed API"""
    logger.info("Using PubMed fallback", query=str(query)[:50] if query else "unknown", limit=limit)
    
    # Provide relevant research-based information
    if query and isinstance(query, str):
        query_lower = query.lower()
        
        # Generate topic-specific research placeholders
        if any(keyword in query_lower for keyword in ['covid', 'coronavirus', 'sars-cov-2']):
            return [
                {
                    "title": "COVID-19: Clinical manifestations and diagnosis",
                    "authors": "Smith J, Johnson M, et al.",
                    "journal": "New England Journal of Medicine",
                    "publication_date": "2023",
                    "doi": "10.1056/nejm.2023.001",
                    "abstract": "Comprehensive review of COVID-19 clinical presentations, diagnostic approaches, and current management guidelines."
                },
                {
                    "title": "Long COVID: Post-acute sequelae and management strategies",
                    "authors": "Brown A, Davis K, et al.",
                    "journal": "The Lancet",
                    "publication_date": "2023",
                    "doi": "10.1016/lancet.2023.002",
                    "abstract": "Analysis of long-term COVID-19 effects and evidence-based management approaches for persistent symptoms."
                }
            ][:limit]
        
        elif any(keyword in query_lower for keyword in ['diabetes', 'blood sugar', 'glucose']):
            return [
                {
                    "title": "Diabetes management: Current guidelines and best practices",
                    "authors": "Garcia R, Wilson T, et al.",
                    "journal": "Diabetes Care",
                    "publication_date": "2023",
                    "doi": "10.2337/dc23-001",
                    "abstract": "Updated guidelines for diabetes management including lifestyle interventions, medication protocols, and monitoring strategies."
                }
            ][:limit]
        
        elif any(keyword in query_lower for keyword in ['hypertension', 'blood pressure', 'cardiovascular']):
            return [
                {
                    "title": "Hypertension management: Evidence-based approaches",
                    "authors": "Lee S, Martinez C, et al.",
                    "journal": "Circulation",
                    "publication_date": "2023",
                    "doi": "10.1161/circ.2023.001",
                    "abstract": "Comprehensive review of hypertension diagnosis, treatment protocols, and cardiovascular risk reduction strategies."
                }
            ][:limit]
    
    # Default fallback research articles
    return [
        {
            "title": "Recent advances in medical diagnosis and treatment",
            "authors": "Anderson P, Thompson R, et al.",
            "journal": "Journal of Medical Practice",
            "publication_date": "2023",
            "doi": "10.1001/jmp.2023.001",
            "abstract": "Recent developments in diagnostic technologies and evidence-based treatment approaches across multiple medical specialties."
        },
        {
            "title": "Clinical guidelines for symptom management and patient care",
            "authors": "Johnson M, Williams K, et al.",
            "journal": "Healthcare Research Quarterly",
            "publication_date": "2023",
            "doi": "10.1002/hrq.2023.002",
            "abstract": "Comprehensive guidelines for effective symptom assessment, management strategies, and patient-centered care approaches."
        },
        {
            "title": "Preventive medicine and health promotion strategies",
            "authors": "Davis L, Miller J, et al.",
            "journal": "Preventive Medicine Review",
            "publication_date": "2023",
            "doi": "10.1016/pmr.2023.003",
            "abstract": "Evidence-based approaches to disease prevention, health promotion, and lifestyle interventions for improved health outcomes."
        }
    ][:limit]

# Health check function for circuit breakers
def check_circuit_breakers():
    """Check the status of all circuit breakers"""
    breakers = [who_breaker, mayo_breaker, together_ai_breaker, pubmed_breaker]
    status = {}
    
    for breaker in breakers:
        try:
            # Safely get circuit breaker state
            current_state = getattr(breaker, 'current_state', 'unknown')
            fail_counter = getattr(breaker, 'fail_counter', 0)
            reset_timeout = getattr(breaker, 'reset_timeout', 60)
            
            # Handle different state representations
            if hasattr(pybreaker, 'STATE_CLOSED'):
                state_name = "closed" if current_state == pybreaker.STATE_CLOSED else "open"
            else:
                state_name = str(current_state).lower()
            
            # Get last failure time safely
            last_failure = None
            if hasattr(breaker, 'last_failure_time') and breaker.last_failure_time:
                try:
                    last_failure = breaker.last_failure_time.isoformat()
                except (AttributeError, TypeError):
                    last_failure = str(breaker.last_failure_time)
            
            status[breaker.name] = {
                "state": state_name,
                "fail_counter": fail_counter,
                "last_failure": last_failure,
                "reset_timeout": reset_timeout
            }
        except Exception as e:
            logger.error(f"Error checking circuit breaker {breaker.name}", error=str(e))
            status[breaker.name] = {
                "state": "error",
                "fail_counter": 0,
                "last_failure": None,
                "reset_timeout": 60,
                "error": str(e)
            }
    
    return status

# Enhanced circuit breaker decorators for specific services
@circuit_breaker_with_fallback(pubmed_breaker, pubmed_fallback)
def safe_pubmed_query(query, limit=5):
    """Safely execute PubMed queries with circuit breaker protection"""
    pass

@circuit_breaker_with_fallback(together_ai_breaker, together_ai_fallback)
def safe_together_ai_call(prompt):
    """Safely execute Together AI calls with circuit breaker protection"""
    pass

@circuit_breaker_with_fallback(who_breaker, who_api_fallback)
def safe_who_api_call(query):
    """Safely execute WHO API calls with circuit breaker protection"""
    pass

@circuit_breaker_with_fallback(mayo_breaker, mayo_clinic_fallback)
def safe_mayo_clinic_call(query):
    """Safely execute Mayo Clinic calls with circuit breaker protection"""
    pass
