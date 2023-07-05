from django.urls import path
from . import views

urlpatterns = [
    path('data/', views.theropy_data_get.as_view()),
    # path('getvideo/', views.wav2lip_sync.as_view()),
]