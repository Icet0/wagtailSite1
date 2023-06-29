import os
import shutil
from django import forms
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from architecture.models import Architecture
from context.views import get_columns
from django.core.files import File
from visualisation.views import myVisualisation
from visualisation.models import Visualisation

from .myFeatures.featuresAPI import addFeatures

from .models import *
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required

class AddForm(forms.ModelForm):
    
    def __init__(self, *args, **kwargs):
        addFiles = kwargs.pop('addFiles', [])
        super(AddForm, self).__init__(*args, **kwargs)

        self.fields['addFiles'] = forms.FileField(label='Ajouter un fichier ', required=False)
    class Meta:
            model = AddFormModel
            fields = ['addFiles']
  


class ListForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        files = kwargs.pop('files', [])
        functions = kwargs.pop('functions', [])
        super(ListForm, self).__init__(*args, **kwargs)

        
        
        self.fields['files'] = forms.ChoiceField(
            choices=[(file, file) for file in files],
            widget=forms.Select(attrs={'class': 'form-control', 'size': '5'})
        )
        self.fields['files'].initial = files
        
        self.fields['functions'] = forms.MultipleChoiceField(
            choices=[(function, function) for function in functions],
            widget=forms.SelectMultiple(attrs={'class': 'form-control', 'size': '10'})
        )
        self.fields['functions'].initial = functions
        

        
        
    class Meta:
        model = FeaturesModel
        fields = ['files', 'functions']
        widgets = {

        }


# Create your views here.
@login_required
def features_view(request):
    print('features_view')
    myArchitecture_pk = request.session.get('architecture_pk', None)
    files = []
    file_names = []
    epoch = 1 #! a changer On part du principe qu'on à qu'une epoch, mais à gérer ici et au niveau des calculs de features

    if(myArchitecture_pk is None):
        files = ["demo.csv"]
        images = []
        images_path = []
        #copie de notre fichier demo dans le dossier uploads
        demo_path = settings.MEDIA_ROOT+'/filesDemo/demo.csv'
        good_path = settings.MEDIA_ROOT+'/uploads/'+request.user.username+"/exp1/demo.csv"
        shutil.copyfile(demo_path, good_path)

        file_names = [good_path]
        my_path = settings.MEDIA_ROOT+'/uploads/'+request.user.username+"/exp1/"
        
    else:
        myArchitecture = Architecture.objects.get(pk=myArchitecture_pk)
        working_directory = myArchitecture.contextModel.workingDirectory
        file_names = [file.file.name for file in working_directory.workingFiles.all()]
        images = []
        images_path = []
        my_path = settings.MEDIA_ROOT+'/uploads/'+request.user.username+"/exp"+str(working_directory.numExp)+"/"
        for file_name in file_names:
            files.append( file_name.split('/')[-1])
            print("num experiment ",working_directory.numExp)
        file_names.append(AddFormModel.objects.filter(user=request.user).last().addFiles.name)
        files.append(AddFormModel.objects.filter(user=request.user).last().addFiles.name.split('/')[-1])
            
            
            
    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'
                , 'spectral edge frequency': 'calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs)'
                , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'
                , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'
                # , 'shannon entropy (dyad)' : 'calcShannonEntropyDyad(epoch, lvl, nt, nc, fs)'
                , 'crosscorrelation (dyad)' : 'calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs)'
                , 'hjorth activity' : 'calcActivity(epoch)'
                , 'hjorth mobility' : 'calcMobility(epoch)'
                , 'hjorth complexity' : 'calcComplexity(epoch)'
                , 'skewness' : 'calcSkewness(epoch)'
                , 'kurtosis' : 'calcKurtosis(epoch)'
                , 'Petrosian FD' : 'calcPetrosianFD(epoch)'
                , 'Hjorth FD' : 'calcHjorthFD(epoch)'
                , 'Katz FD' : 'calcKatzFD(epoch)'
                , 'Higuchi FD' : 'calcHiguchiFD(epoch)'
                # , 'calcSampleEntropy' : 'calcSampleEntropy(epoch)'   #LONGUEEEEE
            , 'calcWE' : 'calcWE(epoch)'
            , 'calsSE' : 'calcSE(epoch)'
            , 'calcSPEn' : 'calcSPEn(epoch,fs)'
            , 'calc_PP_SampEn' : 'calc_PP_SampEn(epoch)'
            , 'calcApEn' : 'calcApEn(epoch)'
            , 'calcTWE' : 'calcTWE(epoch,fs)'
            # , 'calcWaveletTransform' : 'calcWaveletTransform(epoch)'
            # , 'Detrended Fluctuation Analysis' : 'calcDFA(epoch)'  # DFA takes a long time!
                }
    
    if request.method == 'POST':
        
        form = ListForm(request.POST, files=files, functions=functions)
        if form.is_valid():
            
            file = form.cleaned_data['files']
            functions_list = form.cleaned_data['functions']
            
            print('files', files)
            # file_names.append(AddFormModel.objects.last().addFiles.name)
            print('file_names after AddFormModel', file_names)
            for f in file_names:
                if file in f:
                    real_file = f
                    break
            real_file = os.path.join(settings.MEDIA_ROOT, real_file)
            print('real_file', real_file)
            print('functions_list', functions_list)
            path = addFeatures(real_file, functions_list)
            print('path', path)
            request.session['path'] = path
            
            #?affichage features
            df = pd.read_csv(path)

            print ('len(df.nunique(1)) : ', len(df.nunique(0)))
            print("df\n",df)
            cap = []
            for i in range (len(df.nunique(0))):
                oneFeature = pd.DataFrame(np.zeros([len(df.index),epoch]))
                print("oneFeature shape : ",oneFeature.shape)
                if i == 0 :
                    cap = df.iloc[:, i]
                else:
                    for j in range(epoch):
                        if i + j < df.shape[1]:  # Vérifier si l'index est valide
                            if j == 0:
                                column = df.iloc[:, i + j]
                                oneFeature = pd.concat([cap, column], axis=1, ignore_index=True)
                            else:
                                column = df.iloc[:, i + j]
                                oneFeature = pd.concat([oneFeature, column], axis=1, ignore_index=True)
                    print("oneFeature : ",oneFeature)
                    
                    if (myArchitecture_pk is None):
                        path = os.path.join(settings.MEDIA_ROOT, 'uploads/'+request.user.username+'/exp'+str(1)+'/Visualisation/')
                        path_tmp = ('uploads/'+request.user.username+'/exp'+str(1)+'/Visualisation/')
                    else:
                        path = os.path.join(settings.MEDIA_ROOT, 'uploads/'+request.user.username+'/exp'+str(working_directory.numExp)+'/Visualisation/')
                        path_tmp = ('uploads/'+request.user.username+'/exp'+str(working_directory.numExp)+'/Visualisation/')
                    if not os.path.exists(path):
                        os.makedirs(path)
                        
                    # Obtenir les capteurs
                    oneFeature = oneFeature.reset_index(drop=True)

                    # Obtenir les capteurs à partir de la colonne 0
                    sensors = oneFeature.iloc[:, 0]               
                    # Vérifier le nombre d'epochs
                    num_epochs = oneFeature.shape[1] - 1  # Exclure la colonne 'Capteurs'

                    # Vérifier s'il y a plusieurs epochs
                    if num_epochs > 1:
                        # Créer une figure et des sous-graphiques pour chaque capteur
                        fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 6), sharex=True)

                        # Parcourir les capteurs
                        for i, sensor in enumerate(sensors):
                            # Obtenir les valeurs de la feature pour le capteur donné
                            feature_values = oneFeature.loc[sensor].values[1:]


                            # Tracer une courbe pour chaque capteur
                            axes[i].plot(range(1, num_epochs + 1), feature_values)
                            axes[i].set_ylabel('Valeur de la feature')
                            axes[i].set_title(f'Capteur: {sensor}')

                        # Ajouter un titre commun pour les sous-graphiques
                        fig.suptitle('Variation de la feature en fonction des epochs')

                        # Ajuster les espacements entre les sous-graphiques
                        plt.tight_layout()

                    else:
                        # Obtenir les valeurs de la feature pour une seule epoch
                        feature_values = oneFeature.iloc[:, 1]

                        # Tracer un histogramme pour chaque capteur
                        plt.bar(range(len(sensors)), feature_values)
                        plt.xlabel('Capteurs EEG')
                        plt.ylabel('Valeur de la feature')
                        plt.title('Histogramme de '+str(df.columns[i+j])+' pour une seule epoch')

                        # Ajouter les noms des capteurs sous les barres d'histogramme
                        plt.xticks(range(len(sensors)), sensors)

                    # Afficher le graphe
                    print("path : ",path)
                    plt.savefig(path+"feature"+str(i)+".png")
                    
                    images_path.append(path_tmp+"feature"+str(i)+".png")
                    plt.close()
                    #? affichage graphique
            
            Visualisation.objects.all().delete()
            for elt in images_path:
                img = Visualisation(image=elt)
                images.append(img)
                
            #! A RAJOUTER POUR TELECHARGER, METTRE UNE NOIVELLE VUE + BOUTON
            # with open(path, 'rb') as fh:
            #     response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            #     response['Content-Disposition'] = 'inline; filename=' + os.path.basename(path)
            #     return response
            # ! -------------------------------------------------------------
        addForm = AddForm(request.POST,request.FILES)
        if addForm.is_valid():
            addFile = addForm.cleaned_data.get('addFiles')
            print('addFile', addFile)
            if(addFile):
                myfile = AddFormModel(addFiles=addFile,user=request.user)
                myfile.save()
                print('myfile', myfile.addFiles)
                files.append(myfile.addFiles.name.split('/')[-1])
                file_names.append(myfile.addFiles.name)
        
        
        action = request.POST.get('action')
        if action == 'compute':
            print('in compute', files)
            real_file = os.path.join(my_path,files[0])
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
            try:
                path = settings.MEDIA_ROOT + '/uploads/' + request.user.username + '/exp' + str(working_directory.numExp) + '/Visualisation/'
            except:
                path = settings.MEDIA_ROOT + '/uploads/' + request.user.username + '/exp' + str(1) + '/Visualisation/'
            if not os.path.exists(path):
                os.makedirs(path)
            print('path : ',path)
            visualisations = {
                'Compute psd':'compute_psd(df,path,myArchitecture_pk)',
                "Raw signal" : "raw_signal(df,path,n_epochs)"
                }
            visualisations_choice = ['Compute psd','Raw signal']
            result = myVisualisation(df,epoch,path,visualisations,visualisations_choice,myArchitecture_pk)
            if result != None:
                print('ok result : ', result)
                Visualisation.objects.all().delete()
                for elt in result:
                    #? tryyyyy
                    elt = elt.split('/')[1:]
                    elt = os.path.join(*elt)
                    print('elt : ',elt)
                    #? ----------------
                    img = Visualisation(image=elt)
                    images.append(img)
                

    print('files', files)
    form = ListForm(files=files,functions=functions.keys())
    addForm = AddForm()
    
    context = {
        'title':'Features',
        'figures':images,
        'form': form,
        'addForm': addForm, 
    }
    return render(request, 'features/features.html',context)

@login_required
def download_features(request):
    path = request.session.get('path',None)
    if path is None:
        redirect('features_view')
        
    with open(path, 'rb') as fh:
        response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
        response['Content-Disposition'] = 'inline; filename=' + os.path.basename(path)
        return response
    
    
# @login_required
# def add_features(request):
    