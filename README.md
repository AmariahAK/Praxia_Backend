# Praxia Backend

Praxia is an AI-powered healthcare assistant backend system developed by Amariah Kamau. This system provides medical symptom analysis, X-ray image interpretation, medical research retrieval, and personalized health recommendations through a robust REST API.

## Features

- **Symptom Analysis**: AI-powered diagnosis of medical symptoms with personalized recommendations
- **X-ray Analysis**: Deep learning-based interpretation of X-ray images using MONAI models
- **Medical Research**: Integration with PubMed for retrieving relevant medical research
- **Diet Analysis**: Nutritional assessment and personalized dietary recommendations
- **User Profiles**: Personalized health profiles with medical history and preferences
- **Chat System**: Persistent chat sessions with the AI assistant

## Tech Stack

- **Framework**: Django & Django REST Framework
- **Database**: PostgreSQL
- **Cache & Message Broker**: Redis
- **Task Queue**: Celery
- **AI Integration**: Together AI API
- **Medical Imaging**: MONAI
- **Server**: Daphne (ASGI) with Nginx
- **Containerization**: Docker & Docker Compose

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Together AI API key (for AI functionality)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AmariahAK/Praxia_Backend.git
   cd Praxia_Backend
   ```

2. Create a `.env` file based on the provided example:
   ```bash
   cp .env.example .env
   ```

3. Update the `.env` file with your Together AI API key and other settings.

4. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

5. The API will be available at `http://localhost:8000/api/`

### Development Setup

For local development without Docker:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database:
   ```bash
   python manage.py migrate
   ```

4. Create a superuser:
   ```bash
   python manage.py createsuperuser
   ```

5. Run the development server:
   ```bash
   python manage.py runserver
   ```

## API Endpoints

### Authentication
- `POST /api/auth/register/`: Register a new user
- `POST /api/auth/login/`: Login and get authentication token
- `POST /api/auth/logout/`: Logout and invalidate token

### User Profile
- `GET /api/profile/`: Get user profile
- `PATCH /api/profile/`: Update user profile

### Medical Consultations
- `GET /api/consultations/`: List all consultations
- `POST /api/consultations/`: Create a new consultation

### X-ray Analysis
- `GET /api/xray-analyses/`: List all X-ray analyses
- `POST /api/xray-analyses/`: Upload and analyze an X-ray image

### Research Queries
- `GET /api/research/`: List all research queries
- `POST /api/research/`: Create a new research query

### Chat
- `GET /api/chat-sessions/`: List all chat sessions
- `POST /api/chat-sessions/`: Create a new chat session
- `GET /api/chat-sessions/{id}/`: Get a specific chat session
- `GET /api/chat-sessions/{id}/messages/`: Get messages for a chat session
- `POST /api/chat-sessions/{id}/messages/`: Send a message and get AI response

### System
- `GET /api/health/`: Check system health status

## Deployment

### Production Deployment

For production deployment:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

This will start the application with production settings, including:
- Nginx with SSL support
- Optimized server configurations
- Reduced debug information

### Environment Variables

Key environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | Generated |
| `DEBUG` | Debug mode | `True` in dev, `False` in prod |
| `ALLOWED_HOSTS` | Allowed hostnames | `localhost,127.0.0.1` |
| `DB_NAME` | Database name | `praxia_db` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `postgres` |
| `TOGETHER_AI_API_KEY` | Together AI API key | None |
| `TOGETHER_AI_MODEL` | AI model to use | `Qwen/Qwen2.5-7B-Instruct` |

## Rate Limiting

The API implements rate limiting to prevent abuse:

| Endpoint Type | Authenticated Rate | Anonymous Rate |
|---------------|-------------------|----------------|
| Chat | 30/minute | 3/minute |
| Consultations | 10/minute | 3/minute |
| X-ray Analysis | 5/hour | 3/minute |
| Research | 20/hour | 3/minute |

## Health Checks

The system performs automatic health checks every 14 minutes to ensure all components are functioning correctly. The health status can be checked at `/api/health/`.

## Project Structure

```
praxia_backend/
├── api/                    # Main application
│   ├── AI/                 # AI models and logic
│   ├── models/             # Database models
│   ├── serializers/        # API serializers
│   ├── views/              # API views
│   └── urls/               # URL routing
├── data/                   # Data files
│   ├── ai_identity.txt     # AI identity information
│   └── models/             # AI model weights
├── media/                  # User-uploaded files
├── nginx/                  # Nginx configuration
├── praxia_backend/         # Project settings
└── docker-compose.yml      # Docker configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is proprietary and owned by Amariah Kamau.

## Contact

- **Developer**: Amariah Kamau
- **LinkedIn**: [https://www.linkedin.com/in/amariah-kamau-3156412a6/](https://www.linkedin.com/in/amariah-kamau-3156412a6/)
- **GitHub**: [https://github.com/AmariahAK](https://github.com/AmariahAK)
