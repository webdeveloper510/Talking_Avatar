from django.db import models
from django.contrib.auth.models import *


class theropy(models.Model):
    topic=models.CharField(max_length=200, null=True,blank=True)
    questions=models.TextField()
    answers=models.TextField()

class AudioVideo(models.Model):
    face = models.FileField(upload_to='videos/')
    audio = models.FileField(upload_to='audio/')