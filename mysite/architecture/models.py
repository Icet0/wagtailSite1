from django.db import models

# Create your models here.
from django.db import models

class Architecture(models.Model):
    # MODEL_TYPES = [
    #     ('CNN', 'Convolutional Neural Network'),
    #     ('RNN', 'Recurrent Neural Network'),
    #     # Ajoutez ici d'autres types de modèles possibles
    # ]
    
    EVAL_MERTICS = [
        ('accuracy', 'Accuracy'),
        ('f1', 'F1'),
        ('precision', 'Precision'),
    ]

    model_type = models.CharField(max_length=100, blank=False, null=False, default='CNN')
    architecture = models.TextField()
    training_split = models.FloatField()
    batch_size = models.IntegerField()
    model_epochs = models.IntegerField()
    repetition = models.IntegerField()
    evaluation_metrics = models.CharField(max_length=50, choices=EVAL_MERTICS, blank=True, null=True)

    def __str__(self):
        return self.model_type
