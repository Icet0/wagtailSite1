from django.db import models

# Create your models here.
class Workflow(models.Model):
    workflow = models.TextField()
    def __str__(self):
        return self.workflow
    
    
    image = models.ImageField(null=True, blank=True, )
