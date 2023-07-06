from rest_framework import serializers
from auth_app.models import User
from django.contrib.auth.hashers import make_password

class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model=User
        fields=['id','email','name','dob','gender','password']

        extra_kwargs={
            'email': {'error_messages': {'required': "email is required",'blank':'Please enter email'}},
            'name': {'error_messages': {'required': "name is required",'blank':'Please enter name'}},     
            'dob': {'error_messages': {'required': "dob is required",'blank':'Please enter dob'}},     
            'gender': {'error_messages': {'required': "gender is required",'blank':'Please enter gender'}},     
            'password': {'error_messages': {'required': "password is required",'blank':'Please enter password'}}     
          }

    def create(self, validated_data,):
        return User.objects.create(
            email=validated_data['email'],
            name=validated_data['name'],
            dob=validated_data['dob'],
            gender=validated_data['gender'],
            password = make_password(validated_data['password'])
)


         
class UserLoginSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(max_length=250)
    class Meta:
     model=User
     fields=['email','password']
     extra_kwargs={
        'email': {'error_messages': {'required': "email is required",'blank':'please provide a email'}},
        'password': {'error_messages': {'required': "password is required",'blank':'please Enter a email'}}
        
    }
class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model=User
        fields=['id', 'name','email','dob','gender']        

# class User_Update_Profile_Serializer(serializers.ModelSerializer):
#     class Meta:
#         model=User
#         fields=['id','First_name','Middle_name','Last_name','email','mobile','Date_of_birth','Gender','Country_of_birth','location', 'occupation','payment_per_annum','value_per_annum']        

# class UserChangePasswordSerializer(serializers.Serializer):
#     old_password = serializers.CharField(required=True)
#     new_password = serializers.CharField(required=True)


# class User_list_Serializer(serializers.ModelSerializer):
#     class Meta:
#         model= User
#         fields=['id','customer_id', 'country_code','email','First_name','Middle_name','Last_name','Date_of_birth','Gender','Country_of_birth','mobile','location','is_verified','occupation','payment_per_annum','value_per_annum','aml_pep_status', 'read']        

# class User_address_Serializer(serializers.ModelSerializer):
#     class Meta:
#         model= User_address
#         fields=['id','flat','building','street', 'postcode','city','state','country','country_code']        

