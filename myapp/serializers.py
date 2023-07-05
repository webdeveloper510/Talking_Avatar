from rest_framework import serializers
from .models import theropy

class theropySerializer(serializers.ModelSerializer):
    class Meta:
        model=theropy
        fields = ['id','topic','questions','answers']
        # fields= '__all__'