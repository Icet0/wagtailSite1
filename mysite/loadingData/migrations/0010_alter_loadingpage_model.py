# Generated by Django 4.2.1 on 2023-05-31 12:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('loadingData', '0009_loadingpage_model'),
    ]

    operations = [
        migrations.AlterField(
            model_name='loadingpage',
            name='model',
            field=models.CharField(blank=True, max_length=250, null=True),
        ),
    ]