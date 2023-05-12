import io
import json
import re
from typing import Any, Iterable, Optional
from django import forms
from django.conf import settings
from django.db import models
from django.shortcuts import render
from wagtail.models import Page,Orderable,ParentalKey
from wagtail.admin.panels import FieldPanel, MultipleChooserPanel,MultiFieldPanel
from wagtail.fields import RichTextField
# importation du module os
import os
from django.core.files import File
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.files.base import ContentFile




# Create your models here.
def upload_to(instance, filename):
    # Generate a unique filename using the user's ID and the current timestamp
    user = instance.user.username
    current_time = timezone.now().strftime('%Y%m%d')
    
    exp = instance.numExp
    
    filename = f"{user}/exp{exp}/{current_time}_{filename}"
    return f"uploads/{filename}"

class FilePerso(models.Model):
    
    numExp = models.IntegerField(default=0)
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    file = models.FileField(upload_to=upload_to)

        
class workingDirectory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    uploaded_at = models.DateTimeField(default=timezone.now)  # Provide a default value
    # directory = models.FilePathField(blank=True, path=upload_to,allow_files=False, allow_folders=True,  null=True)
    csv_file = models.FileField(null=True, upload_to=upload_to) #" patients"
    # Créer une liste pour stocker les informations des fichiers CSV
    csv_files = models.JSONField(null=True, blank=True, default=list)
    workingFiles = models.ManyToManyField(FilePerso, blank=True)
    labels = models.FileField(null=True, upload_to=upload_to)
    location = models.FileField(null=True, upload_to=upload_to)
    numExp = models.IntegerField(default=1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # super().__init__(*args, **kwargs)
        created = kwargs.pop('created', False) 
        super(workingDirectory, self).__init__(*args, **kwargs)
        base_dir = os.path.join(settings.MEDIA_ROOT, "uploads/"+self.user.username)

        print("INIT WORKING DIRECTORY")

        # Find the last existing 'exp' directory
        if created:
            last_exp = 0
            while True:
                exp_path = os.path.join(base_dir, f"exp{last_exp+1}")
                if os.path.exists(exp_path):
                    last_exp += 1
                else:
                    break

            # Create the next 'exp' directory
            new_exp = last_exp + 1
            self.numExp = new_exp
        
        
    
    def __str__(self):
        return self.user.username
    
    def handle_uploaded_file(self,file):
        # Récupérer le nom du fichier
        filename = file['filename']
        file_content = file['content'].encode('utf-8')  # Convertir la chaîne de caractères en bytes
        file_obj = ContentFile(file_content, name=filename)

        file_instance = FilePerso.objects.create(file=file_obj, numExp=self.numExp)
        self.workingFiles.add(file_instance)
        
    def getCsv_files(self):
        return json.loads(self.csv_files)
    
    
class LoadingPage(Page):
    intro = models.CharField(max_length=250)
    body = RichTextField(blank=True)
    # csv_file = models.FileField(blank=True)
    # # csv_files = [models.FileField(blank=True)]
    # csv_files = models.ManyToManyField('wagtaildocs.Document', blank=True)
    # directory = models.FilePathField(blank=True, path=settings.MEDIA_ROOT,   allow_files=False, allow_folders=True,  null=True)
    
    
    content_panels = Page.content_panels + [
        FieldPanel('intro'),
        FieldPanel('body'),
        # MultiFieldPanel([
        #     FieldPanel('csv_files'),
        # ], heading='CSV Files'),
    ]

    

        # fonction pour récupérer les fichiers CSV
    # def get_csv_file(file,user):
    #     print("GET CSV FILES")
    #     # Get the directory where you want to save the file
    #     user_directory = os.path.join(settings.MEDIA_ROOT, str(user.id))
    #     if not os.path.exists(user_directory):
    #         os.makedirs(user_directory)
    #     # Build the full path to the uploaded file
    #     file_path = os.path.join(user_directory, file.name)
    #     # Write the file to disk
    #     with open(file_path, 'wb+') as destination:
    #         for chunk in file.chunks():
    #             destination.write(chunk)
    #     # Now you can use the file path to call os.listdir()
    #     contents = os.listdir(user_directory)
    #     return contents[0]


    def get_csv_files(files, user):
        print("GET CSV FILES")
        user_directory = os.path.join(settings.MEDIA_ROOT, str(user.id))
        if not os.path.exists(user_directory):
            os.makedirs(user_directory)

        csv_files = []  # Liste pour stocker les fichiers CSV

        for file in files:
            file_path = os.path.join(user_directory, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            csv_file = File(open(file_path, 'rb'))  # Crée un objet File à partir du fichier sur le disque
            csv_files.append(csv_file)

        contents = os.listdir(user_directory)
        return csv_files

            
            

