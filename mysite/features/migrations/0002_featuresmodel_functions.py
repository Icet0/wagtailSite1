# Generated by Django 4.2.1 on 2023-06-01 13:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('features', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='featuresmodel',
            name='functions',
            field=models.CharField(blank=True, max_length=250),
        ),
    ]