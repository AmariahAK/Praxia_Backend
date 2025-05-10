from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

class AIConsultationRateThrottle(UserRateThrottle):
    """
    Throttle for AI consultation endpoints
    Limits authenticated users to 10 requests per minute
    """
    rate = '10/minute'
    scope = 'ai_consultation'

class AIXRayRateThrottle(UserRateThrottle):
    """
    Throttle for AI X-ray analysis endpoints
    Limits authenticated users to 5 requests per hour
    """
    rate = '5/hour'
    scope = 'ai_xray'

class AIResearchRateThrottle(UserRateThrottle):
    """
    Throttle for AI research endpoints
    Limits authenticated users to 20 requests per hour
    """
    rate = '20/hour'
    scope = 'ai_research'

class AIChatRateThrottle(UserRateThrottle):
    """
    Throttle for AI chat endpoints
    Limits authenticated users to 30 requests per minute
    """
    rate = '30/minute'
    scope = 'ai_chat'

class AnonymousRateThrottle(AnonRateThrottle):
    """
    Throttle for anonymous users
    Limits anonymous users to 3 requests per minute
    """
    rate = '3/minute'
