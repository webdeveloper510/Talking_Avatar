from django.shortcuts import render
from .models import *
from .serializers import *
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
from pydub import AudioSegment
from os import path
from django.core.files import File
from django.shortcuts import render
from . models import *
from django.http import HttpResponse ,JsonResponse
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from social_django.utils import psa
from requests.exceptions import HTTPError
import re
from rest_framework import status
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential ,model_from_json , load_model
from keras.layers import Embedding , Dense , GlobalAveragePooling1D
from keras.utils import pad_sequences
from keras.preprocessing.text  import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from django.core.files.base import ContentFile
from gtts import gTTS
import pyttsx3
from IPython.display import Audio
import pandas as pd
import numpy as np
from .serializers import *
from pydub.playback import play
import os
import subprocess
from urllib.parse import urljoin
import random 
import string
from googletrans import Translator
translator=Translator()


# url="http://127.0.0.1:8000/static/media/"
url="http://13.53.234.84:8080/static/media/"



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
        
        
        
@api_view(['POST'])
@permission_classes([AllowAny])
@psa()
def register_by_access_token(request, backend):
    backend = request.backend
    print(backend)
    token = request.data.get('access_token')
    user = backend.do_auth(token)
    print(request)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key}, status=status.HTTP_200_OK)
    else:
        return Response({"error": "Invalid_token"}, status=status.HTTP_400_BAD_REQUEST)
        
        
@api_view(["GET","POST"])
def authentication_test(request):
    print(request.user)
    return Response({"message":"User Successfully Created"}, status=status.HTTP_200_OK)
    
    
class CreateAvatar(APIView):
    def post(self,request):
        image_file=request.FILES.get("image")
        video=request.FILES.get("video")

        if not image_file:
            return Response({"message": "No image provided."}, status=status.HTTP_400_BAD_REQUEST)
        if not video:
            return Response({"message": "No video provided."}, status=status.HTTP_400_BAD_REQUEST)
        avatar_image=Avatar.objects.create(image=image_file,sample_video=video)
        image_name = avatar_image.image.name
        full_url=urljoin(url,image_name)
        avatar_image.image_url=full_url
        avatar_image.save()
        return Response({"message":"Image get Successfully"})
        
        
class CreateConversion(APIView):
    def post(self,request):
        avatar_id=request.POST.get("avatar_id")
        user_id=request.POST.get("user_id")
        avatar_name=request.POST.get("avatar_name")
        therapy_id=request.POST.get("therapy_id")
        if not avatar_id:
            return Response({"message":"Please Enter Avatar ID"})
        if not user_id:
            return Response({"message":"User ID Is Must"})
        if not avatar_name:
            return Response({"message":"Please Enter Avata Name"})
        if not therapy_id:
            return Response({"message":"Please Therapy ID"})
        
        userID = User.objects.get(id=user_id)
        conversation=Conversation.objects.create(avatar_id=avatar_id ,user_id=userID.id ,avatar_name=avatar_name ,therapy=therapy_id)
        conversation.save()
        return Response({"message":"Conversation Create Successfully"})
 

class get_avatar(APIView):
    def get(self,request):
        avatar=Avatar.objects.all().values('id','image_url')
        ArraData =[]
        for dict in avatar:
            ArraData.append(dict)
        if ArraData:
            return Response({'status':status.HTTP_200_OK,'message':'Success', 'data':ArraData})
        else:
            return Response({'status':status.HTTP_400_BAD_REQUEST,'message':'Bad Request'})
                
        
    
class TherapyDATA(APIView):
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')     
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS=set(stopwords.words("english"))
        if not text:
            return ""
        text=text.lower()
        text=REPLACE_BY_SPACE_RE.sub(' ',text)
        text=BAD_SYMBOLS_RE.sub(' ',text)
        text=text.replace('x','')
        text=' '.join(word for word in text.split() if word not in STOPWORDS)
        return text
    def language_translate(self,text):
        result = translator.translate(text=text, dest='us')
        return result
    def generate_random_string(self,length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
    def post(self,request):
        conversation_id=request.POST.get("conversation_id")                 #get conversation_audio
        topic=request.POST.get("therapy_name")                              # get therapy_name from the user
        user_input=request.POST.get("question_text")                        # get question from the user
        questionaudio=request.FILES.get("question_audio")                    # get question_audio
        
        if not Therapy.objects.filter(name=topic).exists():
            return JsonResponse({"error": "Therapy Name is Not Found"}, status=404)
        if not topic:
            return Response({"message":"Please Select Therapy"})
        if not user_input:
            return Response({"message":"Please Ask a question"})
        if not questionaudio:
            return Response({"message":"Please send question Audio File"})
        if not conversation_id:
            return Response({"message":"Please Send Conversation ID"})
        
        
        conversation = Conversation.objects.get(id=conversation_id)
        print("conversation------------->>>>",conversation)
        
        # filter out  ['depression', 'anxiety', 'trauma'] from the multiple labels
        therapy_unique_name=Therapy.objects.all().order_by("id") 
        therapy_serializer=TherapySerializer(data=therapy_unique_name,many=True)
        therapy_serializer.is_valid()
        label = [x["name"] for x in therapy_serializer.data]   # Get three label
        
        
        data=TherapyTrainingData.objects.all().order_by('id')
        serializer=TherapyTrainingDataSerializer(data=data , many=True)
        serializer.is_valid()
        array=[]
        for x in serializer.data:
            therapy_name=x['name']
            question=self.clean_text(x['questions'])
            answer=x['answers']
            if therapy_name in label:
                data_dict={"Topic":therapy_name,"Questions":question,"Answers":answer}
                array.append(data_dict)
                
        questions=[dict['Questions']for dict in array]
        MAX_NB_WORDS = 2000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(questions)
        word_index = tokenizer.word_index
        sequence= tokenizer.texts_to_sequences(questions)
        Sentences=pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)              # input

        Y_data=[dict['Topic'] for dict in array]
        
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(Y_data)
        Label = lbl_encoder.transform(Y_data)                            #output_data

        cluster_label=lbl_encoder.classes_.tolist()
        num_class=len(cluster_label)
        
        # model = Sequential()
        # model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
        # model.add(GlobalAveragePooling1D())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(num_class, activation='softmax'))

        # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        # epochs = 500
        # batch_size=128
        # model.fit(Sentences, np.array(Label), epochs=epochs, batch_size=batch_size)
        # # # Save Model
        # model_json=model.to_json()
        # with open("/home/codenomad/Desktop/Talking_Avatar/auth_app/saved_model/classification_model.json", "w") as json_file:
        #     json_file.write(model_json)
        # model.save_weights("/home/codenomad/Desktop/Talking_Avatar/auth_app/saved_model/classification_model_weights.h5")
        
        # return Response({"message":"Model Trained and Saved with successfully."})
        
        # Load Model
        json_file=open(os.getcwd()+"/auth_app/saved_model/classification_model.json","r")
        loaded_model_json=json_file.read()
        json_file.close()

        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights(os.getcwd()+"/auth_app/saved_model/classification_model_weights.h5")

        cleaned_text =self.clean_text(user_input)
        new_input = tokenizer.texts_to_sequences([cleaned_text])
        new_input = pad_sequences(new_input, maxlen=MAX_SEQUENCE_LENGTH)                    # input

        # Make a prediction on the new input
        pred = loaded_model.predict(new_input)
        databasew_match=pred, cluster_label[np.argmax(pred)]
        result=databasew_match[1]
        # Get the Anser
        filter_data=[dict for dict in array if dict["Topic"].strip()== result.strip()]
        get_all_questions=[dict['Questions'] for dict in filter_data]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(get_all_questions)
        question_vectors = vectorizer.transform(get_all_questions)                                  # 2. all questions
        input_vector = vectorizer.transform([cleaned_text])
        similarity_scores = question_vectors.dot(input_vector.T).toarray().squeeze()
        max_sim_index = np.argmax(similarity_scores)
        similarity_percentage = similarity_scores[max_sim_index] * 100
        print("Similarity Score",similarity_percentage)
        if (similarity_percentage)>=75:
            answer = filter_data[max_sim_index]['Answers']   
            text = f'This is {result} Related Therapy. {answer}'
            output=gTTS(text=text, lang="en",tld='us')
            audio_file_name=self.generate_random_string(8)
            output.save(os.getcwd()+f"/static/media/answer_audio/{audio_file_name}.wav")
            audio_file_path = f"answer_audio/{audio_file_name}.wav"
            
            message=Message.objects.create(conversation_id=conversation.id,question_text=user_input,question_audio=questionaudio,
                                            answer_text=text,answer_audio=audio_file_path)
            message.save()
            # # GetAudio(self, request)
        else:
            answer="Sorry I have not gotten your Question"
            output=gTTS(text=answer, lang="en-us", slow=False)
            # output.save("/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/media/audio/input_audio.wav")
            # GetAudio(output)
        return Response({"Label Name ->":result,"Question":question,"Answer":answer})
    
# def GetAudio(self, request):
#     video_path = "/home/codenomad/Desktop/Talking_Avatar/Wav2Lip/media/videos/real_avatar.mp4"
#     audio_path = "/home/codenomad/Desktop/Talking_Avatar/Wav2Lip/media/audio/input_audio.wav"
#     model_path = "/home/codenomad/Desktop/Talking_Avatar/Wav2Lip/checkpoints/wav2lip_gan.pth"
#     file_save_path = "/home/codenomad/Desktop/Talking_Avatar/Wav2Lip/results"

    
#     command = [
#         'python3', '/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/inference.py',
#         '--checkpoint_path', str(model_path),
#         '--face', str(video_path),
#         '--audio', str(audio_path),
#         '--outfile', str(file_save_path)
#     ]
#     subprocess.run(command)

#     return 'done!'

