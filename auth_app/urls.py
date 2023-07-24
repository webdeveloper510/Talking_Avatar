from django.urls import path
from . views import RegistrationView, LoginView, AppleLoginView, ProfileView, TherapyDATA,CreateAvatar,CreateConversion
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('register/', RegistrationView.as_view() ,name='register'),
    path('login/',LoginView.as_view(),name='login'),
    path('apple-login/', AppleLoginView.as_view(),name='apple-login'),
    path('user-profile/', ProfileView.as_view(),name='userprofile'),
    path('data/', TherapyDATA.as_view()),
    path('imageupload/',CreateAvatar.as_view()),
    path('conversation/',CreateConversion.as_view()),

    
    
    # path('send-password-reset-email/', SendResetPasswordEmailView.as_view(), name='send-reset-password-email'),
    # path('reset-password/', ResetPasswordView.as_view(), name='reset-password'),
    # path('update-profile/', Update_profile_view.as_view(), name='auth_update_profile'),   
    # path('verify-email/', VerifyEmailMobileView.as_view(), name='verify-email'),
    # path('exchange-rate/', Exchange_Rate_Converter.as_view(), name='exchange-rate'),
    # path('referral-link/', referral_link_View.as_view(), name='referral-link'),
    # path('digital-verification/', Digital_Id.as_view(), name='digital-verification'),
    # path('is-digitalid-verified/', Is_digital_id_verified.as_view(), name='is-digitalid-verified'),
    # path('create-sender/', create_sender_details_view.as_view(), name='create-sender'),
    # path('resend-otp/', Resend_OTP_View.as_view(), name='resend-otp'),
    # path('activate-email/',  EmailActivationView.as_view(), name='activate-email'),
    # path('delete-user/<int:pk>', Delete_User_View.as_view(), name="delete-user"),
    # path('email/', email.as_view(), name='email'),
    # path('test/', views.test, name="test"),
]

if settings.DEBUG == True or settings.DEBUG == False:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

