from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager 

#  Custom User Manager
class UserManager(BaseUserManager):
    def create_user(self, email, password=None ):
        if not email:
            raise ValueError('User must have an email address')

        user = self.model(
            email = self.normalize_email(email),
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None):
        user = self.create_user(
            email,
            password=password,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user
 
   
#  Custom User Model
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(verbose_name='Email', max_length=200, unique=True)
    name = models.CharField(max_length=30)
    dob = models.CharField(max_length=30)
    gender = models.CharField(max_length=30)
    is_admin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
   
    objects = UserManager()

    USERNAME_FIELD = 'email'
  
    def __str__(self):
        return self.email

