import re
from django.conf import settings
from django.db import models
from django.contrib.auth.models import User


def upload_to(instance, filename):
    return settings.MEDIA_ROOT+ '/uploads/' + instance.user.username + '/data' + '/' + re.sub(r"[^\w.-]", "-", instance.addFiles.name)

# Create your models here.
class FeaturesModel(models.Model):
    files = models.CharField(max_length=500)
    functions = models.CharField(max_length=500, blank=True)

    
    def __str__(self):
        return self.files
    class Meta:
        verbose_name_plural = "FeaturesModel"
        
class AddFormModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    addFiles = models.FileField(max_length=650,upload_to=upload_to) 

    
    def __str__(self):
        return self.addFiles
    class Meta:
        verbose_name_plural = "AddFormModel"