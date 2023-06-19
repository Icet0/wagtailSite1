import io
import os
import pickle
from django import forms
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from matplotlib import pyplot as plt
import numpy as np

from loadingData.models import workingDirectory
from .models import ContextModel
from django.core.files.base import ContentFile
from django.core.files import File
import pandas as pd
from myUtils.utils.Utils import getMontageList
from myUtils.utils.Utils import getMontage
import datetime
import pytz
from mne import *
import ast
from visualisation.models import Visualisation

class VotreFormulaire(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        montages = kwargs.pop('montages', [])
        electrodes = kwargs.pop('electrodes', [])
        frequencies = kwargs.pop('frequencies', [])

        super(VotreFormulaire, self).__init__(*args, **kwargs)

        names = []
        infos = []

        for item in montages:
            for key, value in item.items():
                names.append(key)
                infos.append(value)
        self.fields['montage'] = forms.ChoiceField(
            choices=[(mt, mt) for mt in names],
            widget=forms.Select(attrs={'class': ' desciption form-control', 'size': '5'})
        )
        self.fields['montage'].initial = montages
        
        self.fields['electrodes'] = forms.MultipleChoiceField(
            choices=[(elec, elec) for elec in electrodes],
            widget=forms.SelectMultiple(attrs={'class': 'form-control', 'size': '5'})
        )
        self.fields['electrodes'].initial = electrodes
        
        self.fields['frequences'] = forms.MultipleChoiceField(
            choices=[(freq, freq) for freq in frequencies],
            widget=forms.SelectMultiple(attrs={'class': 'form-control', 'size': '5'})
        )
        self.fields['frequences'].initial = frequencies

        self.fields['frequence_max'].initial = 256
        self.fields['nombre_epochs'].initial = 7
    class Meta:
        model = ContextModel
        fields = ['montage', 'electrodes', 'frequences', 'frequence_max', 'nombre_epochs']
        widgets = {
        
            'frequence_max': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'nombre_epochs': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
        }

        
# Create your views here.
def context_view(request):
    # Votre logique de traitement du formulaire ici
    print("avant le get")
    working_directory = workingDirectory.objects.get(pk=request.session['working_directory_pk'])
    print("apres le get")
    file_names = [file.file.name for file in working_directory.workingFiles.all()]
    print("num experiment ",working_directory.numExp)
    print("working_directory ",working_directory.csv_file)
    # Appel de la fonction pour récupérer les informations des électrodes
    # Accédez au champ FileField dans l'instance du modèle
    file_object = File(open(settings.MEDIA_ROOT+'/'+file_names[0], 'rb'))
    electrodes = get_columns(file_object)
    print("electrodes",electrodes)
    # Appel de la fonction pour récupérer les informations des montages
    montages = get_montages()
    frequencies = get_frequencies()
    # ... Autres traitements ...

    # Passer les informations des électrodes et des montages au contexte
    context = {
        'electrodes': electrodes,
        'montages': montages,
        'frequencies': frequencies,
    }
    
    
    
    if request.method == 'POST':
        # Traitements à effectuer
        form = VotreFormulaire(request.POST, montages=montages, electrodes=electrodes, frequencies=frequencies)
        if form.is_valid():
            electrodes = form.cleaned_data['electrodes']
            montage = form.cleaned_data['montage']
            frequences = form.cleaned_data['frequences']
            frequence_max = form.cleaned_data['frequence_max']
            nombre_epochs = form.cleaned_data['nombre_epochs']
            
            user = request.user
            print(working_directory)
            positions =  pickle.dumps(getMontage(working_directory,user,montage,electrodes))
            # print("montage",montage)
            contextModel = ContextModel.objects.create(montage=montage,electrodes=electrodes,
                                                       frequences=frequences ,frequence_max=frequence_max,
                                                       nombre_epochs=nombre_epochs, workingDirectory=working_directory, positions=positions)
            contextModel.save()
            
            
            request.session['contextModel_pk'] = contextModel.pk
            # return redirect("architecture_view")
            return redirect('modal_content_view')
        
        # Redirection vers une autre page et modèle
        return redirect("context_view", {'msg': 'Votre formulaire est invalide.', 'form': form})
    else:
        form = VotreFormulaire(montages=montages, electrodes=electrodes, frequencies=frequencies)

        context['form'] = form
        # print("context",context)
        return render(request, 'context/context_page.html', context)
    
def get_columns(file: File):

    # if file.filename == '':
    #     return None
    print("file",file)
    try:
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        file.close()

        return columns
    except Exception as e:
        file.close()
        return None

def get_montages():
    return getMontageList()

def get_frequencies():
    return ["alpha", "beta", "gamma", "delta", "theta"]




def modal_content_view(request):
    
    #gestion des boutons
    if request.method == 'POST':
        if request.POST.get('action') == 'Valider':
            return redirect('architecture_view')
        elif request.POST.get('action') == 'Annuler':
            return redirect('context_view')
            
        
        
        
    #? CREATION OF MNE RAW OBJECT ------------------
    contextModel_pk = request.session['contextModel_pk']
    contextModel = ContextModel.objects.get(pk=contextModel_pk)
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
    
    
    fig = raw.plot_sensors(show_names=True)
    print("MEDIA_ROOT",settings.MEDIA_ROOT)
    # path = settings.MEDIA_ROOT + '/uploads/' + str(request.user.username) + '/exp' + str(contextModel.workingDirectory.numExp) + '/Results/'
    # print("path",path)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # path = path + 'sensors.png'
    
    path = os.path.join('uploads', str(request.user.username), 'exp' + str(contextModel.workingDirectory.numExp), 'Results')

    # Vérification et création du répertoire parent si nécessaire
    parent_dir = os.path.join(settings.MEDIA_ROOT, path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    path_tmp = os.path.join(parent_dir, 'sensors.png')

    plt.savefig(path_tmp)
    plt.close()
    Visualisation.objects.all().delete()
    img = Visualisation(image=path)
    print("img",img.image)
    context = {
        "file": img,
    }

    #? CREATION OF MNE RAW OBJECT -------------------    
    return render(request, 'context/modal.html',context=context)