# Generated by Django 4.2.1 on 2023-05-22 12:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('context', '0005_contextmodel_patients'),
    ]

    operations = [
        migrations.AddField(
            model_name='contextmodel',
            name='positions',
            field=models.BinaryField(blank=True, null=True),
        ),
    ]