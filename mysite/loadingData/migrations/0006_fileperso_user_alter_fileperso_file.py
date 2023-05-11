# Generated by Django 4.2.1 on 2023-05-11 12:33

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import loadingData.models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('loadingData', '0005_fileperso_workingdirectory_workingfiles'),
    ]

    operations = [
        migrations.AddField(
            model_name='fileperso',
            name='user',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='fileperso',
            name='file',
            field=models.FileField(upload_to=loadingData.models.upload_to),
        ),
    ]
