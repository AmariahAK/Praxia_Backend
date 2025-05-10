import time
import logging
from datetime import datetime, timedelta
from django.conf import settings
from celery import shared_task
from .praxia_model import PraxiaAI

logger = logging.getLogger(__name__)

class AIHealthCheck:
    """
    Health check system for the Praxia AI
    Ensures the AI model is responsive and functioning correctly
    """
    
    def __init__(self):
        self.last_check_time = None
        self.check_interval = 14 * 60  # 14 minutes in seconds
        self.praxia = PraxiaAI()
    
    def should_run_check(self):
        """Determine if a health check should be run based on time since last check"""
        if not self.last_check_time:
            return True
        
        time_since_last_check = (datetime.now() - self.last_check_time).total_seconds()
        return time_since_last_check >= self.check_interval
    
    def run_check(self):
        """Run a health check on the Praxia AI system"""
        logger.info("Running Praxia AI health check")
        
        try:
            # Test symptom diagnosis
            diagnosis_result = self.praxia.diagnose_symptoms("mild headache and fatigue")
            if not diagnosis_result or "error" in diagnosis_result:
                logger.error(f"Symptom diagnosis check failed: {diagnosis_result.get('error', 'Unknown error')}")
                return False
            
            # Test medical research retrieval
            research_result = self.praxia.get_medical_research("common cold treatment", limit=1)
            if not research_result or (isinstance(research_result, dict) and "error" in research_result):
                logger.error(f"Medical research check failed: {research_result.get('error', 'Unknown error') if isinstance(research_result, dict) else 'No results'}")
                return False
            
            # Update last check time
            self.last_check_time = datetime.now()
            logger.info("Praxia AI health check passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Praxia AI health check failed with exception: {str(e)}")
            return False

@shared_task
def scheduled_health_check():
    """Celery task for scheduled health checks"""
    health_checker = AIHealthCheck()
    if health_checker.should_run_check():
        return health_checker.run_check()
    return True

def startup_health_check():
    """Health check to run when the server starts"""
    logger.info("Running startup health check for Praxia AI")
    health_checker = AIHealthCheck()
    return health_checker.run_check()
