"""
Django settings for praxia_backend project.
"""

from pathlib import Path
import os
from decouple import config, Csv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', cast=bool)

ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=Csv())

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'api',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'django_filters',
    'channels',
    'django_prometheus',
    'django_otp',
    'django_otp.plugins.otp_totp',
]

MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django_otp.middleware.OTPMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]

ROOT_URLCONF = 'praxia_backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'praxia_backend.wsgi.application'

# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

# Database sharding settings
USE_SHARDING = config('USE_SHARDING', cast=bool)

if USE_SHARDING:
    # Define multiple database connections for sharding
    DATABASES = {
        'default': {
            'ENGINE': config('DB_ENGINE'),
            'NAME': config('DB_NAME'),
            'USER': config('DB_USER'),
            'PASSWORD': config('DB_PASSWORD'),
            'HOST': config('DB_HOST'),
            'PORT': config('DB_PORT'),
            'CONN_MAX_AGE': config('DB_CONN_MAX_AGE', cast=int),
            'OPTIONS': {
                'connect_timeout': config('DB_CONNECT_TIMEOUT', cast=int),
                'application_name': 'praxia',
                'keepalives': config('DB_KEEPALIVES', cast=int),
                'keepalives_idle': config('DB_KEEPALIVES_IDLE', cast=int),
                'keepalives_interval': config('DB_KEEPALIVES_INTERVAL', cast=int),
                'keepalives_count': config('DB_KEEPALIVES_COUNT', cast=int),
            },
            'ATOMIC_REQUESTS': config('DB_ATOMIC_REQUESTS', cast=bool),
            'AUTOCOMMIT': config('DB_AUTOCOMMIT', cast=bool),
        },
        'shard_1': {
            'ENGINE': config('DB_ENGINE'),
            'NAME': config('SHARD1_DB_NAME'),
            'USER': config('SHARD1_DB_USER'),
            'PASSWORD': config('SHARD1_DB_PASSWORD'),
            'HOST': config('SHARD1_DB_HOST'),
            'PORT': config('SHARD1_DB_PORT'),
            'CONN_MAX_AGE': config('DB_CONN_MAX_AGE', cast=int),
            'OPTIONS': {
                'connect_timeout': config('DB_CONNECT_TIMEOUT', cast=int),
                'application_name': 'praxia_shard1',
                'keepalives': config('DB_KEEPALIVES', cast=int),
                'keepalives_idle': config('DB_KEEPALIVES_IDLE', cast=int),
                'keepalives_interval': config('DB_KEEPALIVES_INTERVAL', cast=int),
                'keepalives_count': config('DB_KEEPALIVES_COUNT', cast=int),
            },
            'ATOMIC_REQUESTS': config('DB_ATOMIC_REQUESTS', cast=bool),
            'AUTOCOMMIT': config('DB_AUTOCOMMIT', cast=bool),
        },
        'shard_2': {
            'ENGINE': config('DB_ENGINE'),
            'NAME': config('SHARD2_DB_NAME'),
            'USER': config('SHARD2_DB_USER'),
            'PASSWORD': config('SHARD2_DB_PASSWORD'),
            'HOST': config('SHARD2_DB_HOST'),
            'PORT': config('SHARD2_DB_PORT'),
            'CONN_MAX_AGE': config('DB_CONN_MAX_AGE', cast=int),
            'OPTIONS': {
                'connect_timeout': config('DB_CONNECT_TIMEOUT', cast=int),
                'application_name': 'praxia_shard2',
                'keepalives': config('DB_KEEPALIVES', cast=int),
                'keepalives_idle': config('DB_KEEPALIVES_IDLE', cast=int),
                'keepalives_interval': config('DB_KEEPALIVES_INTERVAL', cast=int),
                'keepalives_count': config('DB_KEEPALIVES_COUNT', cast=int),
            },
            'ATOMIC_REQUESTS': config('DB_ATOMIC_REQUESTS', cast=bool),
            'AUTOCOMMIT': config('DB_AUTOCOMMIT', cast=bool),
        },
    }
    
    # Database routers
    DATABASE_ROUTERS = ['api.db_routers.ShardingRouter']
else:
    DATABASES = {
        'default': {
            'ENGINE': config('DB_ENGINE'),
            'NAME': config('DB_NAME'),
            'USER': config('DB_USER'),
            'PASSWORD': config('DB_PASSWORD'),
            'HOST': config('DB_HOST'),
            'PORT': config('DB_PORT'),
            'CONN_MAX_AGE': config('DB_CONN_MAX_AGE', cast=int),
            'OPTIONS': {
                'connect_timeout': config('DB_CONNECT_TIMEOUT', cast=int),
                'application_name': 'praxia',
                'keepalives': config('DB_KEEPALIVES', cast=int),
                'keepalives_idle': config('DB_KEEPALIVES_IDLE', cast=int),
                'keepalives_interval': config('DB_KEEPALIVES_INTERVAL', cast=int),
                'keepalives_count': config('DB_KEEPALIVES_COUNT', cast=int),
            },
            'ATOMIC_REQUESTS': config('DB_ATOMIC_REQUESTS', cast=bool),
            'AUTOCOMMIT': config('DB_AUTOCOMMIT', cast=bool),
        }
    }

# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

ASGI_APPLICATION = 'praxia_backend.asgi.application'
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [(config('REDIS_HOST'), config('REDIS_PORT', cast=int))],
        },
    },
}

# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = config('LANGUAGE_CODE')
TIME_ZONE = config('TIME_ZONE')
USE_I18N = config('USE_I18N', cast=bool)
USE_TZ = config('USE_TZ', cast=bool)

# Email Configurations
EMAIL_BACKEND = config('EMAIL_BACKEND')
EMAIL_HOST = config('EMAIL_HOST')
EMAIL_PORT = config('EMAIL_PORT', cast=int)
EMAIL_USE_TLS = config('EMAIL_USE_TLS', cast=bool)
EMAIL_HOST_USER = config('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = config('DEFAULT_FROM_EMAIL')

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

# CDN settings
USE_CDN = config('USE_CDN', cast=bool)
CDN_URL = config('CDN_URL')

if USE_CDN and CDN_URL:
    # Prepend CDN URL to static and media URLs in production
    STATIC_URL = f'{CDN_URL}/static/'
    MEDIA_URL = f'{CDN_URL}/media/'
else:
    STATIC_URL = 'static/'
    MEDIA_URL = '/media/'

# AWS S3 settings for CDN (if using AWS CloudFront with S3)
if config('USE_S3', cast=bool):
    # AWS settings
    AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_CUSTOM_DOMAIN = config('AWS_S3_CUSTOM_DOMAIN')
    AWS_S3_OBJECT_PARAMETERS = {
        'CacheControl': config('AWS_S3_OBJECT_CACHE_CONTROL'),
    }
    
    # S3 static settings
    STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    STATIC_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/static/'
    
    # S3 media settings
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/media/'
    
    # Add django-storages to INSTALLED_APPS
    INSTALLED_APPS.append('storages')

# Media files
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': config('REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_ANON'),
        'user': config('REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_USER'),
        'ai_consultation': config('REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_CONSULTATION'),
        'ai_xray': config('REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_XRAY'),
        'ai_research': config('REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_RESEARCH'),
        'ai_chat': config('REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_CHAT'),
    }
}

# CORS settings
CORS_ALLOW_ALL_ORIGINS = config('CORS_ALLOW_ALL_ORIGINS', cast=bool)
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', cast=Csv())

# Celery settings
CELERY_BROKER_URL = config('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = config('CELERY_RESULT_BACKEND')
CELERY_ACCEPT_CONTENT = config('CELERY_ACCEPT_CONTENT', cast=Csv())
CELERY_TASK_SERIALIZER = config('CELERY_TASK_SERIALIZER')
CELERY_RESULT_SERIALIZER = config('CELERY_RESULT_SERIALIZER')
CELERY_TIMEZONE = TIME_ZONE

# AI settings
TOGETHER_AI_API_KEY = config('TOGETHER_AI_API_KEY')
TOGETHER_AI_MODEL = config('TOGETHER_AI_MODEL')
INITIALIZE_XRAY_MODEL = config('INITIALIZE_XRAY_MODEL', cast=bool)

# Health check settings
HEALTH_CHECK_INTERVAL = config('HEALTH_CHECK_INTERVAL', cast=int)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}

# LibreTranslate settings
LIBRETRANSLATE_URL = config('LIBRETRANSLATE_URL')

# Health news settings
HEALTH_NEWS_SOURCES = config('HEALTH_NEWS_SOURCES', cast=Csv())
HEALTH_NEWS_CACHE_TIMEOUT = config('HEALTH_NEWS_CACHE_TIMEOUT', cast=int)

# 2FA settings
OTP_TOTP_ISSUER = config('OTP_TOTP_ISSUER')

# File upload limits
DATA_UPLOAD_MAX_MEMORY_SIZE = config('DATA_UPLOAD_MAX_MEMORY_SIZE', cast=int)
FILE_UPLOAD_MAX_MEMORY_SIZE = config('FILE_UPLOAD_MAX_MEMORY_SIZE', cast=int)
