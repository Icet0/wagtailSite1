# Generated by Django 4.2.1 on 2023-06-22 09:08

from django.db import migrations, models
import prediction.models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0002_prediction_architecture_alter_prediction_file'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='file',
            field=models.FileField(max_length=650, null=True, upload_to=prediction.models.upload_to),
        ),
    ]
