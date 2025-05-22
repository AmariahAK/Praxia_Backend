import pybreaker
import time
import structlog
from functools import wraps
from django.core.cache import cache

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
    fail_max=3,
    reset_timeout=30,
    exclude=[ValueError, TypeError],
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

# Example fallback functions
def who_api_fallback(query):
    """Fallback function for WHO API"""
    logger.info("Using WHO API fallback", query=query)
    return "WHO guidelines unavailable; consult local health authorities."

def mayo_clinic_fallback(query):
    """Fallback function for Mayo Clinic scraping"""
    logger.info("Using Mayo Clinic fallback", query=query)
    return "Mayo Clinic data unavailable; refer to standard medical resources."

def together_ai_fallback(prompt):
    """Fallback function for Together AI API"""
    logger.info("Using Together AI fallback", prompt_length=len(prompt))
    return {
        "error": "API unavailable",
        "message": "Fallback response: please consult a healthcare professional."
    }

def pubmed_fallback(query, limit=5):
    """Fallback function for PubMed API"""
    logger.info("Using PubMed fallback", query=query, limit=limit)
    return [
        {"title": "Recent advances in medical diagnosis", "authors": "Smith J, et al.", "journal": "Medical Journal", "publication_date": "2023"},
        {"title": "Clinical guidelines for symptom management", "authors": "Johnson M, et al.", "journal": "Healthcare Research", "publication_date": "2022"}
    ]

# Health check function for circuit breakers
def check_circuit_breakers():
    """Check the status of all circuit breakers"""
    breakers = [who_breaker, mayo_breaker, together_ai_breaker, pubmed_breaker]
    status = {}
    
    for breaker in breakers:
        status[breaker.name] = {
            "state": "closed" if breaker.current_state == pybreaker.STATE_CLOSED else "open",
            "fail_counter": breaker.fail_counter,  
            "last_failure": getattr(breaker, 'last_failure', None),
            "reset_timeout": breaker.reset_timeout
        }
    
    return status
