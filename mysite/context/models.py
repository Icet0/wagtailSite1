from django.db import models
from wagtail.models import Page
from wagtail.models import Page
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel

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

    # Ajoutez d'autres champs et méthodes selon vos besoins

    def __str__(self):
        return f"{self.montage} - {self.electrode} - {self.frequences} - {self.frequence_max} - {self.nombre_epochs}"

    class Meta:
        verbose_name_plural = "ContextModel"
