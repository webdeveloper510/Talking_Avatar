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
            email=email,
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
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
   
    objects = UserManager()

    USERNAME_FIELD = 'email'
    def __str__(self):
        return self.email
    
    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        return self.is_admin


    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        return self.is_admin



class Therapy(models.Model):
    name=models.CharField(max_length=100)
    
    
class TherapyTrainingData(models.Model):
    name = models.CharField(max_length=100, null=True, blank=True)
    questions = models.TextField()
    answers = models.TextField()

class Avatar(models.Model):
    image = models.ImageField(upload_to="images/")
    image_url=models.CharField(max_length=200,blank=True,null=True)
    sample_video=models.FileField(upload_to="sample_videos")

class Conversation(models.Model):
    avatar= models.ForeignKey(Avatar,on_delete=models.CASCADE)
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    avatar_name = models.CharField(max_length=100, null=True, blank=True)
    therapy= models.BigIntegerField()
    
class Message(models.Model):
    conversation=models.ForeignKey(Conversation,on_delete=models.CASCADE)
    question_text=models.TextField()
    question_audio = models.FileField(upload_to='question_audio/')
    answer_text=models.TextField()
    answer_audio = models.TextField(max_length=100,null=True,blank=True)
    answer_video=models.TextField(max_length=100,null=True,blank=True)
   
