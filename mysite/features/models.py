from django.db import models

# Create your models here.
class FeaturesModel(models.Model):
    files = models.CharField(max_length=500)
    functions = models.CharField(max_length=500, blank=True)

    def __str__(self):
        return self.files
    class Meta:
        verbose_name_plural = "FeaturesModel"