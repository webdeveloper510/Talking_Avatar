# Generated by Django 4.2.3 on 2023-07-24 07:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auth_app', '0007_alter_message_table'),
    ]

    operations = [
        migrations.AlterField(
            model_name='message',
            name='answer_audio',
            field=models.TextField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='message',
            name='answer_video',
            field=models.TextField(blank=True, max_length=100, null=True),
        ),
    ]
