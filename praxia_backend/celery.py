import os
from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'praxia_backend.settings')

app = Celery('praxia_backend')

app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Configure periodic tasks
app.conf.beat_schedule = {
    'run-health-check-every-6-hours': {
        'task': 'api.AI.ai_healthcheck.scheduled_health_check',
        'schedule': crontab(minute=0, hour='*/6'),  
    },
    'fetch-health-news-daily': {
        'task': 'api.AI.praxia_model.scrape_health_news',  
        'schedule': crontab(hour=6, minute=0), 
        'kwargs': {'source': 'all', 'limit': 5},
    },
    'monitor-rss-feeds': {
        'task': 'api.AI.praxia_model.monitor_rss_feeds',
        'schedule': crontab(minute=0, hour='*/6'), 
    },
    'periodic-model-cleanup': {
        'task': 'api.AI.praxia_model.periodic_model_cleanup',
        'schedule': crontab(minute=0, hour='*/12'), 
    },
    'health-data-refresh': {
        'task': 'api.AI.praxia_model.health_data_refresh',
        'schedule': crontab(minute=30, hour='*/4'),  
    },
    'validate-model-integrity': {
        'task': 'api.AI.praxia_model.validate_model_integrity',
        'schedule': crontab(minute=0, hour=0),  
    },
}

app.conf.timezone = 'UTC'
