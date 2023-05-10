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
