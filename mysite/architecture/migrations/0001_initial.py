# Generated by Django 4.2.1 on 2023-05-15 12:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Architecture',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_type', models.CharField(choices=[('CNN', 'Convolutional Neural Network'), ('RNN', 'Recurrent Neural Network')], max_length=50)),
                ('architecture', models.TextField()),
                ('training_split', models.FloatField()),
                ('batch_size', models.IntegerField()),
                ('model_epochs', models.IntegerField()),
                ('repetition', models.IntegerField()),
                ('evaluation_metrics', models.TextField()),
            ],
        ),
    ]
