import ast
import os
import pickle
from django import forms
from django.conf import settings
from django.shortcuts import render
import numpy as np
import pandas as pd
import torch
from myUtils.utils.Utils import convertCSVsTOmat
from architecture.models import Architecture
from myUtils.utils.Train import generate_images

from prediction.models import Prediction
from django.core.files import File



class PredictionForm(forms.Form):
    predict = forms.FileField(label='Choisissez un fichier ', required=True)

def predict(file,model, architecture):
    print("PREDICT")
    print(model)
    print(file)
    
    labels = None
    path        = settings.MEDIA_ROOT + '/uploads/' + architecture.contextModel.workingDirectory.user.username + '/exp' + str(architecture.contextModel.workingDirectory.numExp) + '/mat/prediction/'
    electrodes  = architecture.contextModel.electrodes
    epoch       = architecture.contextModel.nombre_epochs
    frequencies = architecture.contextModel.frequences
    locations   = (pickle.loads)(architecture.contextModel.positions)
    electrodes  = ast.literal_eval(electrodes)
    frequencies = ast.literal_eval(frequencies)
    class myFile:
        path = None
        def __init__(self,path):
            self.path = os.path.join(settings.MEDIA_ROOT,path.name)
        def __str__(self) -> str:
            return str(str(self.path))
            
    file = myFile(file)
    print("file",file)
    #? Charger le csv en mat + images
    
    
    print('locations',locations)
    mat = convertCSVsTOmat([file],labels,path,electrodes,epoch,frequencies)
    path = os.path.join(path, 'images_time.mat')
    img = generate_images(len(frequencies),epoch,path,mat,locations)
    Mean_Images = np.mean(img, axis=0)
    Images = np.transpose(img, (1, 0, 2, 3, 4)) # (nbExp, n_epoch, nbFreq, xIMG,yIMG)
    #? ------------------------------    
    
    # Mettre le modèle en mode évaluation
    model.eval()

    # Prétraitement des données de l'instance à prédire
    # Assurez-vous de prétraiter les données de la même manière que lors de l'entraînement

    # Créer un tenseur PyTorch à partir de l'instance de données
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if architecture.model_type == 'BasicCNN':
        data = torch.tensor(Mean_Images, dtype=torch.float32).to(torch.device(device))
    else:
        data = torch.tensor(Images, dtype=torch.float32).to(torch.device(device))

    # Effectuer la prédiction
    with torch.no_grad():
        output = model(data)

    # Convertir les résultats en probabilités ou en étiquettes de classe
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    # Afficher les résultats de la prédiction
    print("Probabilités :", probabilities)
    print("Classe prédite :", predicted_class+1) # +1 car les classes commencent à 1 et non à 0
    
    
    
    
# Create your views here.
def prediction_view(request):
    
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            print("VALID")
            architecture_pk = request.session.get('architecture_pk')
            myArchitecture = Architecture.objects.get(pk=architecture_pk)
            prediction_file = request.FILES.get('predict')
            model = request.session.get('model')
            prediction = Prediction(architecture = myArchitecture , file=prediction_file, model=model)
            prediction.save()
            print("model",model)
            with open(model, 'rb') as file:
                model = pickle.load(file)
            print("model",model)
            print("prediction_file",prediction.file.name)
            
            
            predict(prediction.file,model,myArchitecture)
            
    else:
        form = PredictionForm()
    
    context = {"form": form}
    return render(request, 'prediction/prediction.html', context=context)