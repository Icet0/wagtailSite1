from django.conf import settings
from django.db import models
from wagtail.models import Page
from wagtail.models import Page
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel

from myUtils.utils.Utils import convertCSVsTOmat

# Create your models here.
class Context(Page):
    
    body = RichTextField(blank=True)
    intro = models.CharField(max_length=250)

    content_panels = Page.content_panels + [
        FieldPanel('body'),
        FieldPanel('intro'),
    ]


class ContextModel(models.Model):
    montage = models.CharField(max_length=100)
    electrodes = models.CharField(max_length=100)
    frequences = models.CharField(max_length=100)
    frequence_max = models.PositiveIntegerField()
    nombre_epochs = models.PositiveIntegerField()
    workingDirectory = models.ForeignKey('loadingData.workingDirectory', on_delete=models.CASCADE, null=True, blank=True)
    features = models.BinaryField(null=True, blank=True)

    def __str__(self):
        return f"{self.montage} - {self.electrodes} - {self.frequences} - {self.frequence_max} - {self.nombre_epochs}"

    class Meta:
        verbose_name_plural = "ContextModel"

    def save(self, *args, **kwargs):
        # Logique personnalisée avant la sauvegarde
        # enregistrer tout dans des matrices : 
        myFiles = []
        for f in self.workingDirectory.workingFiles.all():
            myFiles.append(f.file)  # Obtenez les fichiers CSV à convertir
        labels = self.workingDirectory.labels  # Obtenez les étiquettes
        print("labels", labels)
        print("myFiles", myFiles)
        epochs = self.nombre_epochs  # Récupérez le nombre d'epochs
        frequencies = self.frequences  # Récupérez les fréquences
        path = settings.MEDIA_ROOT+"/uploads/"+self.workingDirectory.user.username+'/'+f'exp{self.workingDirectory.numExp}'+'/mat'  # Récupérez le chemin du répertoire de travail
        features = convertCSVsTOmat(myFiles, labels,path,self.electrodes, epochs, frequencies)  # Convertir les fichiers CSV en matrice
        print("features", features.shape)
        self.features = features
        # Appel de la méthode save() de la classe parent pour effectuer la sauvegarde
        super().save(*args, **kwargs)
        
        # Logique personnalisée après la sauvegarde
        