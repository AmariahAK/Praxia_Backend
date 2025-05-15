# Praxia Backend

Praxia is an AI-powered healthcare assistant backend system developed by Amariah Kamau. This system provides medical symptom analysis, X-ray image interpretation, medical research retrieval, and personalized health recommendations through a robust REST API. It is now fully open-source to encourage collaboration and innovation in the medtech space.

## Features

- **Symptom Analysis**: AI-powered diagnosis of medical symptoms with personalized recommendations
- **Multilingual Support**: Translation of symptoms and responses in English, French, and Spanish
- **X-ray Analysis**: Deep learning-based interpretation of X-ray images for pneumonia, fractures, and tumors
- **Medical Research**: Integration with PubMed for retrieving relevant medical research
- **Health News**: Automated scraping and summarization of health news from WHO and CDC
- **Diet Analysis**: Nutritional assessment and personalized dietary recommendations
- **User Profiles**: Personalized health profiles with medical history and preferences
- **Chat System**: Persistent chat sessions with the AI assistant
- **Real-time Communication**: WebSocket support for instant AI responses
- **Circuit Breakers**: Resilience against external API failures with fallback responses
- **Comprehensive Monitoring**: Prometheus and Grafana integration for system metrics

## Tech Stack

- **Framework**: Django & Django REST Framework
- **Database**: PostgreSQL with connection pooling
- **Cache & Message Broker**: Redis
- **Task Queue**: Celery
- **AI Integration**: Together AI API
- **Medical Imaging**: MONAI with DenseNet121
- **Translation**: LibreTranslate (self-hosted)
- **Web Scraping**: BeautifulSoup4
- **Text Summarization**: Transformers with distilbart
- **Server**: Daphne (ASGI) with Nginx
- **Containerization**: Docker & Docker Compose
- **Monitoring**: Prometheus & Grafana
- **WebSockets**: Django Channels

## Getting Started

For detailed setup instructions, refer to the [Setup Guide](guide/Setup.md).

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/AmariahAK/Praxia_Backend.git
   cd Praxia_Backend
   ```

2. Create a `.env` file based on the `.env.prod` example in the project root.

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8000/api/`

## API Endpoints

### Authentication
- `POST /api/auth/register/`: Register a new user
- `POST /api/auth/login/`: Login and get authentication token
- `POST /api/auth/logout/`: Logout and invalidate token

### User Profile
- `GET /api/profile/`: Get user profile
- `PATCH /api/profile/`: Update user profile
- `POST /api/profile/confirm-gender/`: Confirm and lock gender information

### Medical Consultations
- `GET /api/consultations/`: List all consultations
- `POST /api/consultations/`: Create a new consultation (supports multilingual input)

### X-ray Analysis
- `GET /api/xray-analyses/`: List all X-ray analyses
- `POST /api/xray-analyses/`: Upload and analyze an X-ray image (detects pneumonia, fractures, tumors)

### Research Queries
- `GET /api/research/`: List all research queries
- `POST /api/research/`: Create a new research query

### Health News
- `GET /api/health-news/`: Get latest health news articles
  - Query parameters: `source` (who, cdc, all), `limit` (default: 3, max: 10)

### Chat
- `GET /api/chat-sessions/`: List all chat sessions
- `POST /api/chat-sessions/`: Create a new chat session
- `GET /api/chat-sessions/{id}/`: Get a specific chat session
- `GET /api/chat-sessions/{id}/messages/`: Get messages for a chat session
- `POST /api/chat-sessions/{id}/messages/`: Send a message and get AI response

### WebSocket Endpoints
- `ws://localhost:8000/ws/chat/{session_id}/`: Real-time chat with the AI
- `ws://localhost:8000/ws/health/`: Health check WebSocket endpoint

### System
- `GET /api/health/`: Check system health status

## Project Structure

```
Praxia_Backend/
├── api/                    # Main application
│   ├── AI/                 # AI-related logic
│   │   ├── ai_healthcheck.py  # Health check views for AI services
│   │   └── praxia_model.py    # AI model integration
│   ├── email_templates/    # Email templates for user notifications
│   │   ├── base.html
│   │   ├── password_changed.html
│   │   ├── password_reset.html
│   │   └── verification_email.html
│   ├── middleware/         # Custom middleware
│   │   └── throttling.py   # Rate limiting logic
│   ├── models/             # Database models
│   │   ├── __init__.py
│   │   ├── ai.py           # AI-related models
│   │   ├── auth.py         # Authentication models
│   │   ├── translation.py  # Translation models
│   │   └── user.py         # User models
│   ├── serializers/        # API serializers
│   │   ├── __init__.py
│   │   ├── ai_serializer.py   # AI data serialization
│   │   ├── auth_serializer.py # Authentication serialization
│   │   └── user_serializer.py # User data serialization
│   ├── urls/               # URL routing
│   │   └── urls.py         # API URL configurations
│   ├── views/              # API views
│   │   ├── ai_views.py     # AI-related views
│   │   ├── auth_views.py   # Authentication views
│   │   ├── health_views.py # Health check views
│   │   └── user_views.py   # User-related views
│   ├── __init__.py
│   ├── admin.py            # Django admin configurations
│   ├── apps.py             # App configurations
│   ├── circuit_breaker.py  # Circuit breaker for external APIs
│   ├── consumers.py        # WebSocket consumers
│   ├── db_routers.py       # Database routing logic
│   ├── locustfile.py       # Load testing configurations
│   ├── routing.py          # WebSocket routing
│   ├── signals.py          # Signal handlers
│   └── tests.py            # Unit tests
├── data/                   # Data files
│   ├── grafana/provisioning/  # Grafana provisioning configs
│   ├── models/             # AI model weights
│   └── ai_identity.txt     # AI identity information
├── nginx/                  # Nginx configuration
│   └── nginx.conf          # Nginx server config
├── praxia_backend/         # Project settings
│   ├── __init__.py
│   ├── asgi.py             # ASGI configuration
│   ├── celery.py           # Celery configuration
│   ├── settings.py         # Django settings
│   ├── urls.py             # Root URL configurations
│   └── wsgi.py             # WSGI configuration
├── prometheus/             # Prometheus configuration
│   └── prometheus.yml      # Prometheus config file
├── utils/                  # Utility functions
│   ├── download_model.py   # Model download utilities
│   └── email.py            # Email utilities
├── docker-compose.prod.yml # Docker Compose for production
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker configuration
├── Dockerfile.sam          # Dockerfile for SAM
├── entrypoint-prod.sh      # Production entrypoint script
├── init-db.sh              # Database initialization script
├── manage.py               # Django management script
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Deployment

For detailed deployment instructions, refer to the [Setup Guide](guide/Setup.md).

## Rate Limiting
The API implements rate limiting to prevent abuse:

| Endpoint Type      | Authenticated Rate | Anonymous Rate |
|--------------------|--------------------|----------------|
| Chat              | 30/minute          | 3/minute       |
| Consultations     | 10/minute          | 3/minute       |
| X-ray Analysis    | 5/hour             | 3/minute       |
| Research          | 20/hour            | 3/minute       |
| Health News       | 20/hour            | 3/minute       |

## Resilience Features

### Circuit Breakers
Circuit breakers for external API calls prevent cascading failures:

| Service         | Failure Threshold | Reset Timeout |
|-----------------|-------------------|---------------|
| WHO API         | 5 failures        | 60 seconds    |
| Mayo Clinic     | 5 failures        | 60 seconds    |
| Together AI     | 3 failures        | 30 seconds    |
| PubMed          | 5 failures        | 60 seconds    |
| LibreTranslate  | 5 failures        | 60 seconds    |

### Caching
Responses are cached to improve performance:

| Data Type         | Cache Duration |
|-------------------|----------------|
| Diagnosis         | 24 hours       |
| Research          | 24 hours       |
| WHO Guidelines    | 24 hours       |
| Mayo Clinic Data  | 24 hours       |
| Translations      | 24 hours       |
| X-ray Analysis    | 24 hours       |
| Health News       | 12 hours       |

## Monitoring

### Health Checks
Automatic health checks run every 14 minutes at `/api/health/`.

### Prometheus & Grafana
- Prometheus metrics: port 9090
- Grafana dashboards: port 3000
- Default Grafana login: admin / admin_password

## Multilingual Support
Supports symptom analysis in:
- **English**: Default
- **French**: Medical terms and symptoms
- **Spanish**: Medical terms and symptoms

The system translates user input to English for processing and responses back to the original language.

## X-ray Analysis Capabilities
Detects:
- **Pneumonia**: Lung inflammation patterns
- **Fractures**: Bone fractures
- **Tumors**: Tumorous growths
- **Normal**: No significant abnormalities

Includes confidence scores and recommendations.

## Contributing to Praxia

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Contributions must comply with the Praxia License, including not using the name "Praxia" for derivative works and crediting Amariah Kamau, as specified in [LICENSE.md](guide/LICENSE.md).

## License

Licensed under the Praxia License. See [LICENSE.md](guide/LICENSE.md) for details.

### Third-Party Licenses
- [Django](https://github.com/django/django) - BSD License
- [Django REST Framework](https://github.com/encode/django-rest-framework) - BSD License
- [PostgreSQL](https://www.postgresql.org/) - PostgreSQL License
- [Redis](https://github.com/redis/redis) - BSD License
- [Celery](https://github.com/celery/celery) - BSD License
- [MONAI](https://github.com/Project-MONAI/MONAI) - Apache 2.0 License
- [Docker](https://github.com/docker/docker-ce) - Apache 2.0 License

## Contact

- **Developer**: Amariah Kamau
- **LinkedIn**: [https://www.linkedin.com/in/amariah-kamau-3156412a6/](https://www.linkedin.com/in/amariah-kamau-3156412a6/)
- **GitHub**: [https://github.com/AmariahAK](https://github.com/AmariahAK)