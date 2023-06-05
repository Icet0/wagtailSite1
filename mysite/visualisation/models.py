from django.db import models

# Create your models here.
class VisualisationModel(models.Model):
    
    files = models.CharField(max_length=500)
    visualisation = models.CharField(max_length=500, blank=True)

    def __str__(self):
        return self.files
    class Meta:
        verbose_name_plural = "VisualisationModel"