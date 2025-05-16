"""
ASGI config for praxia_backend project.
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'praxia_backend.settings')
django.setup()  # Set up Django before importing models

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from api.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})
