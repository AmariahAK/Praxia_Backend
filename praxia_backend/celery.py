import os
from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'praxia_backend.settings')

app = Celery('praxia_backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Configure periodic tasks
app.conf.beat_schedule = {
    'run-health-check-every-14-minutes': {
        'task': 'api.AI.ai_healthcheck.scheduled_health_check',
        'schedule': crontab(minute='*/14'),  # Run every 14 minutes
    },
    'fetch-health-news-daily': {
        'task': 'api.AI.praxia_model.PraxiaAI.scrape_health_news',
        'schedule': crontab(hour=6, minute=0),  # Run daily at 6 AM
        'kwargs': {'source': 'all', 'limit': 5},
    },
}
