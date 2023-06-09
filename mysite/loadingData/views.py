import json
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render

# Create your views here.
from django import forms

# from loadingData.models import get_csv_file

from django.shortcuts import render
from django.views import View
from django.contrib import messages
import os
import csv
from django.db import models

from dashboard.models import Fichier

from .models import LoadingPage, workingDirectory
from multiupload.fields import MultiFileField

        
# Création du formulaire pour choisir le répertoire de travail
class DirectoryForm(forms.Form):
    csv_file = forms.FileField(label='Choisissez un fichier CSV vos patients ', required=True)
    label = forms.FileField(label='Choisissez un fichier CSV pour vos labels ', required=True)
    location = forms.FileField(label='Choisissez un fichier CSV pour la localisation des capteurs *OPTIONEL*', required=False)
    files = MultiFileField(label='Choisir un répertoire de travail ', min_num=1)



# Définition de la vue pour afficher le formulaire et récupérer les fichiers CSV
def load_csv(request):
    try:
        page = LoadingPage.objects.first()
    except LoadingPage.DoesNotExist:
        page = None
    print("PAGE",page)
    model = request.GET.get('model', None)
    if model is not None:
        model_id = model
        try:
            myModel = Fichier.objects.get(id=model_id)
            print("MODEL",myModel)
            page.title = myModel.nom
            page.intro = "Load the same shape of data as the model " + myModel.nom
            page.model = settings.MEDIA_URL + "uploads/" + str(myModel.user.username) + "/" + myModel.parent.parent.nom + "/" + myModel.parent.nom + "/" + myModel.nom
            page.save()
        except Fichier.DoesNotExist:
            # Handle the case when the Fichier object does not exist
            # Return an error message or redirect the user to an appropriate page
            print("The requested Fichier does not exist.")
    else:
        page.title = "Loading data"
        page.intro = "Load your data to train a new model"
        page.model = None
        page.save()
        
    print("LOAD CSV")
    if request.method == 'POST':
        print("POST")
        form = DirectoryForm(request.POST, request.FILES)

        print(form.is_valid())

        if form.is_valid():
            print("VALID")
            for f in request.FILES.getlist('files'):
                print(str(f))
            file = form.cleaned_data['csv_file'] # Récupérer le fichier CSV pour les patients
            files = request.FILES.getlist('files')
            file_label = form.cleaned_data['label']
            file_location = form.cleaned_data['location']
            
            print("FILES",files)
            print("file",file)

            csv_files = []

            # Parcourir vos fichiers CSV et ajouter les informations à la liste
            for f in files:
                file_info = {
                    'filename': f.name,
                    'content': f.read().decode('utf-8')  # Convertir le contenu en chaîne de caractères
                }
                csv_files.append(file_info)
            # Convertir la liste en JSON
            json_data = json.dumps(csv_files)            
            
            working_directory = workingDirectory.objects.create(csv_file = file , csv_files = json_data, labels = file_label , location = file_location,created=True)
            for f in working_directory.getCsv_files():
                working_directory.handle_uploaded_file(f)
            working_directory.save()

            # Utilisez les fichiers CSV comme vous le souhaitez
            # return render(request, 'loading_page.html', {'form': form, 'csv_files': csv_files})
            # return HttpResponse('Formulaire soumis ' + str(working_directory.pk) + " file uploaded " + str(working_directory.csv_file))
            
            if(model is not None):
                print("my model during post")
                return redirect("workflow_view")
                
            else:
                request.session['working_directory_pk'] = working_directory.pk
                return redirect("context_view")

    else:
        form = DirectoryForm()
    return render(request, 'loadingData/loading_page.html', {'form': form, 'page': page})
