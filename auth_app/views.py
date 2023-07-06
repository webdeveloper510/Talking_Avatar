from django.shortcuts import render
from .models import *
from .serializers import *
from rest_framework.views import APIView
from auth_app.renderer import UserRenderer
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import update_last_login
from rest_framework.permissions import IsAuthenticated
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.shortcuts import get_object_or_404, redirect, render

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        # 'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

class RegistrationView(APIView):
    renderer_classes=[UserRenderer]

    def post(self,request,format=None):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            data = serializer.data
            for k in data:
                data[k] = str(data[k])
                if k is None:
                    k == ""
            del data['password']
            return Response({"status":"200", "message":"success", "data":data})
        else:
            for v in serializer.errors.values():
                return Response({"status":"400", "message":v[0]})

class LoginView(APIView):
    renderer_classes=[UserRenderer]
    def post(self,request,format=None):
        serializer=UserLoginSerializer(data=request.data)
        if serializer.is_valid(raise_exception=False):
            email=serializer.data.get('email')
            password=serializer.data.get('password')
            user=authenticate(email=email,password=password)
            if user is not None:
                token= get_tokens_for_user(user)
                update_last_login(None, user)
                return Response({'status':"200",'message':'Login successful','token':token['access']})
            else:
                return Response({'status':"400", "message":'email or password is not valid'})
        else:
            for k in serializer.errors.keys():
                return Response({"status":"400", "message":"Please enter "+k})

#Apple login api
class AppleLoginView(APIView):
    renderer_classes=[UserRenderer]
    
    def post(self,request,format=None):
        email = request.data.get('email')
        if not email:
            return Response({'status':"400", "message":'Please enter email'})
        try:
            validate_email(email)
        except ValidationError  as e:
            return Response({'status':"400", "message": str(e.message)})
        if User.objects.filter(email=email).exists():
            obj = get_object_or_404(User, email=email)
            user=authenticate(email=email,password=obj.password)
            if user is not None:
                token= get_tokens_for_user(user)
                update_last_login(None, user)
                return Response({'status':"200",'message':'Login successful','token':token['access']})
        else:
            User.objects.create(email=email)
            user = User.objects.get(email=email)
            if user is not None:
                token= get_tokens_for_user(user)
                update_last_login(None, user)
                return Response({'status':"200",'message':'Login successful','token':token['access']})
            return Response({'status':"200",'message':'Login successful'})
         
class ProfileView(APIView):
    renderer_classes=[UserRenderer]
    permission_classes=[IsAuthenticated]
    
    def post(self,request,format=None):
        serializer = UserProfileSerializer(request.user)
        dict = serializer.data
        for key in dict:
            dict[key] = str(dict[key])
            if key is None:
                key == ""
        return Response({"status":"200", "message":"success", "data":dict})

