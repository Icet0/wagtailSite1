from django.conf import settings
from django.db import models



def upload_to(instance, filename):
    return settings.MEDIA_ROOT+ '/uploads/' + instance.architecture.contextModel.workingDirectory.user.username + '/exp' + str(instance.architecture.contextModel.workingDirectory.numExp) + '/mat/prediction/' + instance.file.name

# Create your models here.
class Prediction(models.Model):
    architecture = models.ForeignKey('architecture.Architecture', on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(null=True,max_length=650,upload_to=upload_to) #" exp"
    model = models.FileField(null=True) #" model"