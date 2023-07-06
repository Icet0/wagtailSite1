import ast
import os
from django import forms
from django.conf import settings
from django.shortcuts import render
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mne import *
import datetime

import pytz
from architecture.models import Architecture

from .models import VisualisationModel,Visualisation

def raw_signal(df,path,n_epochs):
    print(' IN raw_signal')
    # Diviser le DataFrame en n epochs
    epochs = np.array_split(df, n_epochs)
    n_cols = 2  # Nombre d'images par colonne
    n_rows = (n_epochs + n_cols - 1) // n_cols  # Nombre de lignes nécessaires

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*5))  # Crée une grille de sous-graphiques

    print("epochs : ",epochs)
    # Créer un DataFrame à partir des données
    
    df = pd.DataFrame(epochs[0])

    # Récupérer les noms des colonnes
    column_names = df.columns.tolist()
    column_names.pop(0)  # Supprimer le premier élément

    print('column_names : ',column_names)
    # Parcourez les epochs et tracez les courbes correspondantes dans les sous-graphiques
    for i, epoch in enumerate(epochs):
        row = i // n_cols
        col = i % n_cols
        print('row : ',row)
        print('col : ',col)
        if(len(epochs) == 1):

            for column_name in column_names:
                axs[0].plot(epoch['Time'], epoch[column_name], label=column_name)
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title(f'Epoch {i+1}')
            axs[0].legend()
        else:
            for column_name in column_names:
                axs[row, col].plot(epoch['Time'], epoch[column_name], label=column_name)
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Amplitude')
            axs[row, col].set_title(f'Epoch {i+1}')
            axs[row, col].legend()
    print('axs : ',axs)
    plt.tight_layout()  # Ajuste automatiquement les espacements entre les sous-graphiques
    plt.savefig(path+"raw_signal.png")
    plt.close()

    return path+"raw_signal.png"

def ica_properties(df,path,myArchitecture_pk):
    architecture = Architecture.objects.get(pk=myArchitecture_pk)
    contextModel = architecture.contextModel
    sfreq = contextModel.frequence_max
    montage = contextModel.montage
    electrodes = contextModel.electrodes
    df = pd.read_csv(contextModel.workingDirectory.workingFiles.all()[0].file)
    if "Time" in df.columns:
        df_tmp = df.drop(columns=["Time"])
    elif "time" in df.columns:
        df_tmp = df.drop(columns=["time"])
    else:
        df_tmp = df
    ch_names = ast.literal_eval(electrodes)
    
    data = {
        ch_name: df_tmp[ch_name].values
        for ch_name in ch_names
    }
    
    info = create_info(ch_names, sfreq, ch_types='eeg')

    info.set_montage(montage)
    
    current_datetime = datetime.datetime.now()

    # Convert the current datetime to UTC
    utc_timezone = pytz.timezone('UTC')
    current_utc_datetime = current_datetime.astimezone(utc_timezone)

    # Convert the UTC datetime to a UNIX timestamp
    unix_timestamp = current_utc_datetime.timestamp()
    data = np.array(list(data.values()))

    raw = io.RawArray(data, info=info)
    raw.set_meas_date(unix_timestamp)
    
    print(' IN ica_properties')
    
    ica = preprocessing.ICA(n_components=4, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.exclude = [0,1,2, 3]  # details on how we picked these are omitted here
    fig = ica.plot_properties(raw, picks=ica.exclude)
    plt.savefig(path+"ica_properties.png")
    plt.close()
    
    return path+"ica_properties.png"
    
def compute_psd(df,path,myArchitecture_pk):
    architecture = Architecture.objects.get(pk=myArchitecture_pk)
    contextModel = architecture.contextModel
    sfreq = contextModel.frequence_max
    montage = contextModel.montage
    electrodes = contextModel.electrodes
    df = pd.read_csv(contextModel.workingDirectory.workingFiles.all()[0].file)
    if "Time" in df.columns:
        df_tmp = df.drop(columns=["Time"])
    elif "time" in df.columns:
        df_tmp = df.drop(columns=["time"])
    else:
        df_tmp = df
    ch_names = ast.literal_eval(electrodes)
    
    data = {
        ch_name: df_tmp[ch_name].values
        for ch_name in ch_names
    }
    
    info = create_info(ch_names, sfreq, ch_types='eeg')

    info.set_montage(montage)
    
    current_datetime = datetime.datetime.now()

    # Convert the current datetime to UTC
    utc_timezone = pytz.timezone('UTC')
    current_utc_datetime = current_datetime.astimezone(utc_timezone)

    # Convert the UTC datetime to a UNIX timestamp
    unix_timestamp = current_utc_datetime.timestamp()
    data = np.array(list(data.values()))

    raw = io.RawArray(data, info=info)
    raw.set_meas_date(unix_timestamp)
    
    print(' IN compute_psd')
    fig = raw.compute_psd(fmax=50).plot(picks="data", exclude="bads")
    print('fig : ',fig)
    plt.savefig(path+"compute_psd.png")
    plt.close()
    return path+"compute_psd.png"
    
    
def myVisualisation(df,n_epochs,path,visu,names=None,myArchitecture_pk=None):
    print(' IN myVisualisation')

    keyName = visu.items()
    myKey = []
    for i in keyName:
        if i[0] in names:
            myKey.append(i)
    # Initialize a dictionary of pandas dataframes with the features as keys
    feat = []
 
    if(names != None):
        for key in myKey:
            print('key : ',eval(key[1]))
            feat.append( eval(key[1]) )
    else:
        for key in visu.items():
            feat.append( eval(key[1]) )
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
    images = []
    for file_name in file_names:
        files.append( file_name.split('/')[-1])
    print("num experiment ",working_directory.numExp)
    
    visualisations = { 'VISU 1': 'visu1(data)'
                , 'Compute psd':'compute_psd(df,path,myArchitecture_pk)'
                , "Raw signal" : "raw_signal(df,path,n_epochs)"
                , 'ICA properties' : 'ica_properties(df,path,myArchitecture_pk)'
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
            print('path : ',path)
            result = myVisualisation(df,n_epochs,path,visualisations,visualisations_choice,myArchitecture_pk)
            if result != None:
                print('ok')
                Visualisation.objects.all().delete()
                for elt in result:
                    #? tryyyyy
                    elt = elt.split('/')[1:]
                    elt = os.path.join(*elt)
                    print('elt : ',elt)
                    #? ----------------
                    img = Visualisation(image=elt)
                    images.append(img)
                                
                

    form = ListForm(files=files,visualisation=visualisations.keys())

    
    context = {
        'title':'Visualisation',
        'figures':images,
        'form': form,
    }
    
    return render(request, 'visualisation/visualisation.html', context)