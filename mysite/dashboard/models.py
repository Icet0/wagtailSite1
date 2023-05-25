from django.db import models

# Create your models here.
class Fichier(models.Model):
    nom = models.CharField(max_length=255)
    est_repertoire = models.BooleanField(default=False)
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='fichier_enfants')
    enfants = models.ManyToManyField('self', blank=True, symmetrical=False,related_name='parents')

    def __str__(self):
        return self.nom