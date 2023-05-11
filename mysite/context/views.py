import io
import os
from django import forms
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render

from loadingData.models import workingDirectory
from .models import ContextModel
from django.core.files.base import ContentFile
from django.core.files import File
import pandas as pd
from myUtils.utils.Utils import getMontageList
from myUtils.utils.Utils import getMontage


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
    working_directory = workingDirectory.objects.get(pk=request.session['working_directory_pk'])
    file_names = [file.file.name for file in working_directory.workingFiles.all()]

    print("working_directory",working_directory.csv_file)
    # Appel de la fonction pour récupérer les informations des électrodes
    print(file_names)
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
            montage = getMontage(user,montage,electrodes)
            print("montage",montage)
        
        
        # Redirection vers une autre page et modèle
        return HttpResponse('POST '+str(working_directory))
    else:
        form = VotreFormulaire(montages=montages, electrodes=electrodes, frequencies=frequencies)

        context['form'] = form
        print("context",context)
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