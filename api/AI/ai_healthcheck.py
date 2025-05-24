import pybreaker
import json
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from ..models import HealthCheckResult
from datetime import timedelta
from celery import shared_task
import structlog
from .praxia_model import PraxiaAI, scrape_health_news

logger = structlog.get_logger(__name__)

@shared_task
def scheduled_health_check():
    """Scheduled health check to ensure all services are operational and gather latest data"""
    from ..circuit_breaker import check_circuit_breakers
    # Check if we already have a recent health check (less than 6 hours old)
    six_hours_ago = timezone.now() - timedelta(hours=6)
    recent_check = HealthCheckResult.objects.filter(timestamp__gte=six_hours_ago).first()

    if recent_check:
        logger.info("Using recent health check", check_id=recent_check.id)
        return {
            "timestamp": str(recent_check.timestamp),
            "status": recent_check.status,
            "services": recent_check.services_status,
            "external_data": recent_check.external_data
        }

    results = {
        "timestamp": str(timezone.now()),
        "status": "operational",
        "services": {}
    }

    # Check database connection
    try:
        from django.db import connections
        connections['default'].cursor()
        results["services"]["database"] = "operational"
    except Exception as e:
        results["services"]["database"] = f"error: {str(e)}"
        results["status"] = "degraded"

    # Check Redis connection
    try:
        cache.set('health_check', 'ok', 10)
        assert cache.get('health_check') == 'ok'
        results["services"]["redis"] = "operational"
    except Exception as e:
        results["services"]["redis"] = f"error: {str(e)}"
        results["status"] = "degraded"

    # Check circuit breakers
    from ..circuit_breaker import who_breaker, mayo_breaker, together_ai_breaker, pubmed_breaker
    circuit_breaker_status = check_circuit_breakers()
    results["services"]["circuit_breakers"] = circuit_breaker_status

    # Check external services
    external_services = {
        "who_api": who_breaker,
        "mayo_clinic": mayo_breaker,
        "together_ai": together_ai_breaker,
        "pubmed": pubmed_breaker
    }
    for name, breaker in external_services.items():
        if breaker.current_state == pybreaker.STATE_CLOSED:
            results["services"][name] = "operational"
        else:
            results["services"][name] = "degraded"
            results["status"] = "degraded"

    # Check AI models
    try:
        praxia = PraxiaAI()
        if getattr(settings, "INITIALIZE_XRAY_MODEL", False):
            results["services"]["densenet_model"] = "operational" if praxia.densenet_model else "not_loaded"
        else:
            results["services"]["densenet_model"] = "disabled"
    except Exception as e:
        results["services"]["ai_models"] = f"error: {str(e)}"
        results["status"] = "degraded"

    # Check Celery workers
    try:
        from celery.app.control import Inspect
        from praxia_backend.celery import app

        insp = Inspect(app=app)
        if not insp.ping():
            results["services"]["celery"] = "no_workers_online"
            results["status"] = "degraded"
        else:
            results["services"]["celery"] = "operational"
            # Check active tasks
            active = insp.active()
            if active:
                results["services"]["celery_active_tasks"] = sum(len(tasks) for tasks in active.values())
            # Check queue lengths
            import redis
            redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
            queue_length = redis_client.llen('celery')
            results["services"]["celery_queue_length"] = queue_length
            if queue_length > 100:
                results["services"]["celery_queue_status"] = "backlogged"
                results["status"] = "degraded"
            else:
                results["services"]["celery_queue_status"] = "normal"
    except Exception as e:
        results["services"]["celery"] = f"error: {str(e)}"
        results["status"] = "degraded"

    # Gather external data for AI context
    external_data = {}

    # Get latest health news (use Celery task, but call synchronously for health check)
    try:
        news_articles = scrape_health_news(source='all', limit=5)
        external_data["health_news"] = news_articles
    except Exception as e:
        logger.error("Failed to gather health news", error=str(e))
        external_data["health_news"] = []

    # Get latest medical research trends
    try:
        research_topics = ["COVID-19", "cancer treatment", "heart disease", "diabetes", "mental health"]
        research_data = {}
        praxia = PraxiaAI()
        for topic in research_topics:
            try:
                research_data[topic] = praxia.get_medical_research(query=topic, limit=2)
            except Exception as e:
                logger.error("Function call failed", function="get_medical_research", error=str(e))
                from ..circuit_breaker import pubmed_fallback
                research_data[topic] = pubmed_fallback(topic, 2)
        external_data["research_trends"] = research_data
    except Exception as e:
        logger.error("Failed to gather research trends", error=str(e))
        external_data["research_trends"] = {}

    # Store the results in the database
    health_check = HealthCheckResult.objects.create(
        status=results["status"],
        services_status=results["services"],
        external_data=external_data
    )

    # Log health check results
    if results["status"] == "operational":
        logger.info("Health check passed", services=results["services"])
    else:
        logger.warning("Health check detected issues", services=results["services"])

    # Return combined results
    return {
        "timestamp": str(health_check.timestamp),
        "status": health_check.status,
        "services": health_check.services_status,
        "external_data": health_check.external_data
    }

@shared_task
def startup_health_check():
    """Health check to run at startup"""
    six_hours_ago = timezone.now() - timedelta(hours=6)
    recent_check = HealthCheckResult.objects.filter(timestamp__gte=six_hours_ago).first()
    if recent_check:
        logger.info("Using recent health check for startup", check_id=recent_check.id)
        return {
            "timestamp": str(recent_check.timestamp),
            "status": recent_check.status,
            "services": recent_check.services_status,
            "external_data": recent_check.external_data
        }
    return scheduled_health_check()

@shared_task
def websocket_health_check():
    """Check WebSocket server health"""
    import asyncio
    import websockets

    async def check_websocket():
        try:
            uri = f"ws://localhost:8000/ws/health/"
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                data = json.loads(response)
                return data.get("type") == "pong"
        except Exception as e:
            logger.error("WebSocket health check failed", error=str(e))
            return False

    try:
        result = asyncio.run(check_websocket())
        status = "operational" if result else "degraded"
        results = {
            "timestamp": str(timezone.now()),
            "service": "websocket",
            "status": status
        }
        cache.set('websocket_health_check_results', results, 60 * 15)
        if status == "operational":
            logger.info("WebSocket health check passed")
        else:
            logger.warning("WebSocket health check failed")
        return results
    except Exception as e:
        logger.error("WebSocket health check error", error=str(e))
        return {
            "timestamp": str(timezone.now()),
            "service": "websocket",
            "status": "error",
            "error": str(e)
        }

class AIHealthCheck:
    """Class for checking AI system health"""

    def __init__(self):
        self.praxia_ai = PraxiaAI()

    def run_check(self):
        """Run health check on AI system"""
        try:
            if not getattr(self.praxia_ai, 'together_api_key', None):
                logger.error("Together AI API key missing")
                return False
            if getattr(settings, "INITIALIZE_XRAY_MODEL", False):
                if not hasattr(self.praxia_ai, 'densenet_model') or self.praxia_ai.densenet_model is None:
                    logger.error("X-ray model required but not loaded")
                    return False
            return True
        except Exception as e:
            logger.error("AI health check failed", error=str(e))
            return False
