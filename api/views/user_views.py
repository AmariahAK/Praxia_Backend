from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.shortcuts import get_object_or_404
from ..models import UserProfile
from ..serializers.user_serializer import UserProfileSerializer, UserProfileUpdateSerializer
from ..authentication import SessionJWTAuthentication

class UserProfileView(APIView):
    """View for retrieving and updating user profile"""
    authentication_classes = [SessionJWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    
    def get(self, request):
        """Get the authenticated user's profile"""
        profile = get_object_or_404(UserProfile, user=request.user)
        serializer = UserProfileSerializer(profile, context={'request': request})
        return Response(serializer.data)
    
    def patch(self, request):
        """Update the authenticated user's profile"""
        profile = get_object_or_404(UserProfile, user=request.user)
        serializer = UserProfileUpdateSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            # Return the full profile after update
            full_serializer = UserProfileSerializer(profile, context={'request': request})
            return Response(full_serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ConfirmGenderView(APIView):
    """View for confirming and locking gender"""
    authentication_classes = [SessionJWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        """Confirm and lock gender"""
        profile = get_object_or_404(UserProfile, user=request.user)
        
        # Check if gender is already locked
        if profile.gender_locked:
            return Response(
                {"detail": "Gender is already confirmed and cannot be changed."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if gender is set
        if not profile.gender:
            return Response(
                {"detail": "Please set your gender before confirming."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Lock the gender
        profile.gender_locked = True
        profile.save()
        
        serializer = UserProfileSerializer(profile, context={'request': request})
        return Response({
            "detail": "Gender has been confirmed and cannot be changed in the future.",
            "profile": serializer.data
        })

from rest_framework.authtoken.models import Token

class LogoutView(APIView):
    """View for user logout"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        """Log out the user by deleting their auth token"""
        try:
            # Delete the user's auth token
            Token.objects.filter(user=request.user).delete()
            return Response({"detail": "Successfully logged out."})
        except Exception as e:
            return Response(
                {"detail": "An error occurred during logout."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )