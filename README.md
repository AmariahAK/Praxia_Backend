# Praxia Backend

Praxia is an AI-powered healthcare assistant backend system developed by Amariah Kamau. This system provides medical symptom analysis, X-ray image interpretation, medical research retrieval, and personalized health recommendations through a robust REST API.

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

2. Create a `.env` file based on the `.env.example` in the project root.

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8000/api/`

## Project Structure

```
Praxia_Backend/
├── api/                          # Main Django application
│   ├── AI/                       # AI-related logic
│   ├── email_templates/          # Email templates for notifications
│   ├── middleware/               # Custom middleware
│   ├── models/                   # Database models
│   ├── serializers/              # API serializers
│   ├── urls/                     # URL routing
│   │   └── urls.py               # API endpoint definitions
│   ├── utils/                    # Utility functions
│   │   └── download_model.py     # DenseNet model download utility
│   ├── views/                    # API views
│   ├── admin.py                  # Django admin configurations
│   ├── apps.py                   # App configurations
│   ├── circuit_breaker.py        # Circuit breaker for external APIs
│   ├── consumers.py              # WebSocket consumers
│   ├── db_routers.py             # Database routing logic
│   ├── routing.py                # WebSocket routing
│   └── signals.py                # Signal handlers
├── data/                         # Data files and configurations
│   ├── ai_identity.txt           # AI identity configuration (customizable)
│   └── models/                   # AI model weights storage
├── grafana/                      # Grafana configuration and dashboards
├── guide/                        # Documentation
│   ├── Setup.md                  # Detailed setup instructions
│   └── LICENSE.md                # License information
├── logs/                         # Application logs
├── nginx/                        # Nginx configuration
│   ├── nginx.conf                # Nginx server configuration
│   └── Dockerfile                # Nginx Docker configuration
├── praxia_backend/               # Django project settings
│   ├── __init__.py
│   ├── asgi.py                   # ASGI configuration
│   ├── celery.py                 # Celery configuration
│   ├── settings.py               # Django settings
│   ├── urls.py                   # Root URL configurations
│   └── wsgi.py                   # WSGI configuration
├── prometheus/                   # Prometheus monitoring configuration
├── docker-compose.yml            # Development Docker Compose
├── docker-compose.prod.yml       # Production Docker Compose
├── Dockerfile                    # Main application Docker configuration
├── docker-entrypoint-wrapper.sh  # Docker entrypoint wrapper
├── entrypoint.sh                 # Development entrypoint script
├── entrypoint.prod.sh            # Production entrypoint script
├── manage.py                     # Django management script
├── requirements.txt              # Python dependencies
├── .env                          # Development environment variables
├── .env.prod                     # Production environment variables
└── README.md                     # This file
```

## API Endpoints

All API endpoints are defined in `api/urls/urls.py`. Key endpoints include:

### Authentication
- `POST /api/auth/register/` - Register a new user
- `POST /api/auth/login/` - Login and get authentication token
- `POST /api/auth/logout/` - Logout and invalidate token
- `POST /api/auth/verify_email/` - Verify email address
- `POST /api/auth/password-reset-request/` - Request password reset
- `POST /api/auth/password-reset-confirm/` - Confirm password reset

### User Profile
- `GET /api/profile/` - Get user profile
- `PATCH /api/profile/` - Update user profile
- `POST /api/profile/confirm-gender/` - Confirm and lock gender information

### Medical Services
- `GET /api/consultations/` - List medical consultations
- `POST /api/consultations/` - Create new consultation (supports multilingual input)
- `GET /api/xray-analyses/` - List X-ray analyses
- `POST /api/xray-analyses/` - Upload and analyze X-ray image
- `GET /api/research/` - List research queries
- `POST /api/research/` - Create new research query

### Chat System
- `GET /api/chat-sessions/` - List chat sessions
- `POST /api/chat-sessions/` - Create new chat session
- `GET /api/chat-sessions/{id}/messages/` - Get chat messages
- `POST /api/chat-sessions/{id}/messages/` - Send message and get AI response

### Health & News
- `GET /api/health/` - System health check
- `GET /api/health-news/` - Get latest health news

### WebSocket Endpoints
- `ws://localhost:8000/ws/chat/{session_id}/` - Real-time chat
- `ws://localhost:8000/ws/health/` - Health monitoring

## Configuration

### Environment Files
- `.env` - Development configuration
- `.env.prod` - Production configuration

### AI Identity Customization
Developers can customize Praxia's identity by editing `data/ai_identity.txt`. This file contains:
- AI assistant name and description
- Developer information
- Primary healthcare functions
- Personality traits and response guidelines

### DenseNet Model Setup
If you encounter issues with the X-ray analysis model:

1. Manually run the model download script:
   ```bash
   python api/utils/download_model.py
   ```

2. Restart the Docker containers:
   ```bash
   docker-compose restart
   ```

## Rate Limiting

The API implements rate limiting to prevent abuse:

| Endpoint Type      | Authenticated Rate | Anonymous Rate |
|--------------------|--------------------|----------------|
| Chat              | 30/minute          | 3/minute       |
| Consultations     | 10/minute          | 3/minute       |
| X-ray Analysis    | 5/hour             | 3/minute       |
| Research          | 20/hour            | 3/minute       |
| Health News       | 20/hour            | 3/minute       |

## Monitoring

### Health Checks
- Automatic health checks run every 6 hours at `/api/health/`
- Authenticated health checks available at `/api/health/authenticated/`

### Prometheus & Grafana
- Prometheus metrics: `http://localhost:9090`
- Grafana dashboards: `http://localhost:3000`
- Default Grafana credentials: admin / admin_password

## Multilingual Support

Supports medical consultations in:
- **English** (default)
- **French** - Medical terms and symptoms
- **Spanish** - Medical terms and symptoms

## X-ray Analysis Capabilities

The DenseNet121 model can detect:
- **Pneumonia** - Lung inflammation patterns
- **Fractures** - Bone fractures
- **Tumors** - Tumorous growths
- **Normal** - No significant abnormalities

Results include confidence scores and medical recommendations.

## Contributing to Praxia

Thank you for your interest in Praxia! While I appreciate the community's enthusiasm, this project is currently maintained as a personal portfolio project. I encourage developers to fork the repository and use it as inspiration for their own healthcare applications.

If you'd like to build upon Praxia's foundation, please feel free to:
- Fork the repository for your own projects
- Use the codebase as a learning resource
- Adapt the concepts for your own healthcare solutions

Please ensure any derivative works comply with the Praxia License, including not using the name "Praxia" for derivative works and crediting Amariah Kamau, as specified in [LICENSE.md](guide/LICENSE.md).

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m "Add YourFeature"`)
6. Push to the branch (`git push origin feature/YourFeature`)
7. Open a Pull Request

## License

This project is licensed under the Praxia License. See [LICENSE.md](guide/LICENSE.md) for details.

### Third-Party Licenses

Praxia uses the following open-source software:
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