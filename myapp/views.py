from django.shortcuts import render
from . models import *
from django.http import HttpResponse 
from rest_framework.response import Response
from rest_framework.views import APIView
import re
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential ,model_from_json , load_model
from keras.layers import Embedding , Dense , GlobalAveragePooling1D
from keras.utils import pad_sequences
from keras.preprocessing.text  import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gtts import gTTS
import pyttsx3
from playsound import playsound
from IPython.display import Audio
import pandas as pd
import numpy as np
from .serializers import theropySerializer
from pydub import AudioSegment
from os import path
from pydub.playback import play
import os
import subprocess
from googletrans import Translator
translator=Translator()


class theropy_data_get(APIView):
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
    
    def post(self,request):
        data=theropy.objects.all().order_by('id')
        serializer=theropySerializer(data=data , many=True)
        serializer.is_valid()
        array=[]
        for x in serializer.data:
            topic=x['topic']
            question=self.clean_text(x['questions'])
            answer=x['answers']
            data_dict={"Topic":topic,"Questions":question,"Answers":answer}
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
        # # Load Model

        json_file=open("/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/myapp/saved_model/classification_model.json","r")
        loaded_model_json=json_file.read()
        json_file.close()

        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/myapp/saved_model/classification_model_weights.h5")

        user_input=request.POST.get("input")
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
            print("satpal")
            output.save("/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/media/audio/input_audio.wav")
            GetAudio(self, request)
        else:
            answer="Sorry I have not gotten your Question"
            output=gTTS(text=answer, lang="en-us", slow=False)
            output.save("/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/media/audio/input_audio.wav")
            # GetAudio(output)
            
        return Response({"Label Name ->":result,"Question":user_input,"Answer":answer})


def GetAudio(self, request):
    video_path = "/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/media/video/real_avatar.mp4"
    audio_path = "/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/media/audio/input_audio.wav"
    model_path = "/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/checkpoints/wav2lip_gan.pth"
    file_save_path = "/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/results/result.mp4"

    
    command = [
        'python3', '/home/codenomad/Desktop/avatar_project/threeDtalkingAvatar/Wav2Lip/inference.py',
        '--checkpoint_path', str(model_path),
        '--face', str(video_path),
        '--audio', str(audio_path),
        '--outfile', str(file_save_path)
    ]
    subprocess.run(command)

    return 'done!'

        
    