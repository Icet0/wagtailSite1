import os
from django import forms
from django.conf import settings
from django.shortcuts import render
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from architecture.models import Architecture

from .models import VisualisationModel,Visualisation

def raw_signal(df,path,n_epochs):
    print(' IN raw_signal')
    # Diviser le DataFrame en n epochs
    epochs = np.array_split(df, n_epochs)
    n_cols = 2  # Nombre d'images par colonne
    n_rows = (n_epochs + n_cols - 1) // n_cols  # Nombre de lignes nécessaires

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*5))  # Crée une grille de sous-graphiques

    # Parcourez les epochs et tracez les courbes correspondantes dans les sous-graphiques
    for i, epoch in enumerate(epochs):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].plot(epoch['Time'], epoch['AF7'], label='AF7')
        axs[row, col].plot(epoch['Time'], epoch['AF8'], label='AF8')
        axs[row, col].plot(epoch['Time'], epoch['TP9'], label='TP9')
        axs[row, col].plot(epoch['Time'], epoch['TP10'], label='TP10')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Amplitude')
        axs[row, col].set_title(f'Epoch {i+1}')
        axs[row, col].legend()
    print('axs : ',axs)
    plt.tight_layout()  # Ajuste automatiquement les espacements entre les sous-graphiques
    plt.show()
    plt.savefig(path+"raw_signal.png")
    plt.close()

    return path+"raw_signal.png"
    
def myVisualisation(df,n_epochs,path,visu,names=None):
    print(' IN myVisualisation')

    keyName = visu.items()
    myKey = []
    for i in keyName:
        if i[0] in names:
            myKey.append(i)
    # Initialize a dictionary of pandas dataframes with the features as keys
    if(names != None):
        feat = {key[0]: pd.DataFrame() for key in myKey}  
    else:
        feat = {key[0]: pd.DataFrame() for key in visu.items()}  
    if(names != None):
        for key in myKey:
            print('key : ',eval(key[1]))
            feat = eval(key[1])
    else:
        for key in visu.items():
            feat = eval(key[1])
    print('feat : ',feat)

    return feat

class ListForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        visualisation = kwargs.pop('visualisation', [])
        files = kwargs.pop('files', [])

        super(ListForm, self).__init__(*args, **kwargs)

        self.fields['files'] = forms.ChoiceField(
            choices=[(file, file) for file in files],
            widget=forms.Select(attrs={'class': 'form-control', 'size': '5'})
        )
        self.fields['files'].initial = files
        
        self.fields['visualisation'] = forms.MultipleChoiceField(
            choices=[(function, function) for function in visualisation],
            widget=forms.SelectMultiple(attrs={'class': 'form-control', 'size': '10'})
        )
        self.fields['visualisation'].initial = visualisation
        
        
    class Meta:
        model = VisualisationModel
        fields = ['files','visualisation']
        widgets = {

        }

# Create your views here.
def visualisation_view(request):
    
    myArchitecture_pk = request.session.get('architecture_pk', None)
    myArchitecture = Architecture.objects.get(pk=myArchitecture_pk)
    working_directory = myArchitecture.contextModel.workingDirectory
    file_names = [file.file.name for file in working_directory.workingFiles.all()]
    n_epochs = myArchitecture.contextModel.nombre_epochs
    files = []
    img = None
    for file_name in file_names:
        files.append( file_name.split('/')[-1])
    print("num experiment ",working_directory.numExp)
    
    visualisations = { 'VISU 1': 'visu1(data)'
                , 'VISU 1': 'visu1(data)'
                , 'VISU 2': 'visu2(data)'
                , "Raw signal" : "raw_signal(df,path,n_epochs)"
    }
    if request.method == 'POST':
        
        form = ListForm(request.POST, files=files, visualisation=visualisations)
        if form.is_valid():
            file = form.cleaned_data['files']
            visualisations_choice = form.cleaned_data['visualisation']
            print('files', files)
            for f in file_names:
                if file in f:
                    real_file = f
                    break
            real_file = os.path.join(settings.MEDIA_ROOT, real_file)
            print('real_file', real_file)
            for function in visualisations_choice:
                print('function', function)

            df = pd.read_csv(real_file, sep=",")
            if not "Time" in df.columns or not "time" in df.columns:
                rajoutTime = np.arange(0, 60 , 60/df.shape[0])
                df["Time"] = rajoutTime
                # Nom de la colonne à placer en premier
                column_name = 'Time'

                # Obtenir la liste des colonnes dans l'ordre actuel
                columns = df.columns.tolist()

                # Placer la colonne à l'index 0
                columns.insert(0, columns.pop(columns.index(column_name)))

                # Réindexer le DataFrame avec les nouvelles colonnes
                df = df.reindex(columns=columns)
            elif not "time" in df.columns:
                # Nom de la colonne à placer en premier
                column_name = 'time'

                # Obtenir la liste des colonnes dans l'ordre actuel
                columns = df.columns.tolist()

                # Placer la colonne à l'index 0
                columns.insert(0, columns.pop(columns.index(column_name)))

                # Réindexer le DataFrame avec les nouvelles colonnes
                df = df.reindex(columns=columns)
            elif not "Time" in df.columns:
                # Nom de la colonne à placer en premier
                column_name = 'Time'

                # Obtenir la liste des colonnes dans l'ordre actuel
                columns = df.columns.tolist()

                # Placer la colonne à l'index 0
                columns.insert(0, columns.pop(columns.index(column_name)))

                # Réindexer le DataFrame avec les nouvelles colonnes
                df = df.reindex(columns=columns)
                
            ch_names = df.columns[1:]
            ch_names.tolist()
            path = settings.MEDIA_ROOT + '/uploads/' + request.user.username + '/exp' + str(working_directory.numExp) + '/Visualisation/'
            if not os.path.exists(path):
                os.makedirs(path)
            result = myVisualisation(df,n_epochs,path,visualisations,visualisations_choice)
            if result != None:
                print('ok')
                Visualisation.objects.all().delete()
                img = Visualisation(image=result)
                                
                

    form = ListForm(files=files,visualisation=visualisations.keys())

    
    context = {
        'title':'Visualisation',
        'figures':img,
        'form': form,
    }
    
    return render(request, 'visualisation/visualisation.html', context)