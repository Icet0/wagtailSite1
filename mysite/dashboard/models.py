from django.db import models

# Create your models here.
class Fichier(models.Model):
    nom = models.CharField(max_length=255)
    est_repertoire = models.BooleanField(default=False)
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)

    def __str__(self):
        return self.nom