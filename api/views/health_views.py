from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from ..AI.ai_healthcheck import AIHealthCheck

class HealthCheckView(APIView):
    """View for API health check"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Check if the API is healthy"""
        # Check database connection
        try:
            from django.db import connection
            connection.ensure_connection()
            db_status = "ok"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check AI system
        ai_checker = AIHealthCheck()
        ai_status = "ok" if ai_checker.run_check() else "error"
        
        # Prepare response
        health_status = {
            "status": "healthy" if db_status == "ok" and ai_status == "ok" else "unhealthy",
            "database": db_status,
            "ai_system": ai_status,
            "version": "1.0.0"
        }
        
        status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return Response(health_status, status=status_code)
