from asyncio import get_running_loop
import json
from typing import AsyncIterable
from django.shortcuts import redirect, render
from matplotlib import pyplot as plt

from myUtils.utils.Train import loadData
# from prefect import flow, task

from architecture.models import Architecture
from myUtils.utils.Train import *
from asgiref.sync import sync_to_async, async_to_sync
from django.template.response import TemplateResponse
from django.http import StreamingHttpResponse

import asyncio
import json
import random
from django.http import JsonResponse
from PIL import Image
from mne.viz import plot_topomap
from mne.channels import find_layout

from .models import Workflow
from matplotlib.colors import ListedColormap
import matplotlib

matplotlib.use('Agg')  # Utiliser le backend non interactif Agg

# Create your views here.

def workflow_view(request):
    is_processing = False  # Variable pour indiquer si le traitement est en cours

    try:
        architecture_pk = request.session['architecture_pk']
        myArchitecture = Architecture.objects.get(pk=architecture_pk)
    except KeyError:
        myArchitecture = None
        print("no architecture_pk in session")
    
    
    model = request.GET.get('model', None)
    if model is not None:
        try:
            print('model', model)
            path = settings.MEDIA_ROOT + '/'
            for elt in model.split('/')[2:]:
              path += str(elt) + '/'
            path = path.rstrip('/')
            print('path', path)
            # Ouverture du fichier en mode lecture binaire ('rb')
            with open(path, 'rb') as file:
                # Chargement des données du fichier pkl
                obj = pickle.load(file)

            # Utilisation de l'objet chargé depuis le fichier pkl
            # Par exemple, affichage du contenu de l'objet
            print(obj)
            
            architecture_pk = obj.architecture_pk
            myArchitecture = Architecture.objects.get(pk=architecture_pk)
            print('myArchitecture in pickle', myArchitecture)
        except FileNotFoundError:
            print('Le fichier n\'existe pas')

    
    if request.method == 'POST':
        is_processing = True
        result,models = myFlow(request, myArchitecture)
        if(result is None and models is None):
            print("result is None and models is None")
            return redirect('features_view')
        
        print('result append (after)')
        print ('result.shape', np.mean(result, axis=1).shape)
        cmap_labels = ['hot', 'viridis', 'plasma']  # Liste des noms de colormap pour chaque fréquence
        images_path = []
        mean_result = np.mean(result, axis=1)
        
        fig, axes = plt.subplots(nrows=mean_result.shape[0], ncols=mean_result.shape[1], figsize=(12, 20))
        path = settings.MEDIA_ROOT +'/uploads/'+ request.user.username + '/exp'+ str(myArchitecture.contextModel.workingDirectory.numExp) +'/img/'
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        for epoch in range(mean_result.shape[0]):
            for frequence in range(mean_result.shape[1]):
                image = mean_result[epoch][frequence]
                # imagePIL = Image.fromarray(image, mode='RGB')  # 'L' for grayscale image -> RGB

                
                # imagePIL.save(path)
                ax = axes[epoch][frequence]
                im = ax.imshow(image, cmap=cmap_labels[frequence])
                fig.colorbar(im, ax=ax)
                ax.set_title('Epoch {}, Fréquence {}'.format(epoch+1, frequence+1))

        fig.tight_layout()  # Ajuster les espaces entre les sous-graphiques
        
        fig.savefig(path+"myImages.png")
            
        Workflow.objects.all().delete()
        image = Workflow(image = path+"myImages.png")
        image.save()
        images = Workflow.objects.all()

        context = {
            'is_processing': is_processing,
            'results': images,
            'models': models,
        }
        return render(request, 'workflow/workflow_page.html', context)
    
    context = {
        'is_processing': is_processing,
        'results': None,
        'models': None,
    }
    return render(request, 'workflow/workflow_page.html', context)






# @flow(log_prints=True, name="myWorkflow")
def myFlow(info,myArchitecture,modelPretrained = None):
    print("In myFlow")
    #contextModel
    Features = (pickle.loads)(myArchitecture.contextModel.features)
    Labels = Features[:,-1].astype(int)
    print('Labels  : \n', Labels)
    Patients = (pickle.loads)(myArchitecture.contextModel.patients)
    Locations = (pickle.loads)(myArchitecture.contextModel.positions)
    Frequences = myArchitecture.contextModel.frequences
    Frequence_max = myArchitecture.contextModel.frequence_max
    Nombre_epochs = myArchitecture.contextModel.nombre_epochs
    
    #model
    model = myArchitecture.model_type
    training_split = myArchitecture.training_split
    batch_size = myArchitecture.batch_size
    model_epochs = myArchitecture.model_epochs
    repetition = myArchitecture.repetition
    evaluation_metrics = myArchitecture.evaluation_metrics
    
    
    button_id = info.POST.get('button_id')
    print('button_id', button_id)
    if button_id == "1":
        #workflow 1
        print('workflow 1')
        user = info.user
        numExp = myArchitecture.contextModel.workingDirectory.numExp
        directory = settings.MEDIA_ROOT +'/uploads/'+ user.username + '/exp'+ str(numExp) +'/mat/images_time.mat'
        print("before generate_images")
        Images = generate_images(len(Frequences.split(",")),Nombre_epochs,directory,Features,Locations)
        print("after generate_images")
        save = settings.MEDIA_ROOT +'/uploads/'+ user.username + '/exp'+ str(numExp)
        print("before trainning")
        if(modelPretrained is not None):
            Models = trainning(Images, Labels, Patients,modelPretrained,save,training_split,batch_size,model_epochs,repetition,True,myArchitecture.pk,)
        else:
            Models = trainning(Images, Labels, Patients,model,save,training_split,batch_size,model_epochs,repetition,False,myArchitecture.pk)
        print("after trainning")
        print(Models)
        
        
    elif button_id == "2":
        #workflow 2
        print('workflow 2')
    elif button_id == "3":
        #workflow 3
        print('workflow 3')
        return None, None
    else:
        print('button_id not found')
        
    
    return Images, Models
        
        
        
# async def workflow_view(request):
#     is_processing = False
#     # sync_render = sync_to_async(render)

#     get_session = sync_to_async(request.session.get)
#     architecture_pk = await get_session('architecture_pk')
#     print('architecture_pk', architecture_pk)
#     myArchitecture = await sync_to_async(Architecture.objects.get)(pk=architecture_pk)
#     myArchitecture_str = await sync_to_async(str)(myArchitecture)
#     print('myArchitecture:', myArchitecture_str)

#     if request.method == 'POST':
#         is_processing = True

#         async def generate_results():
#             results = []
#             print("In generate_results")
#             async for result in myFlow(request, myArchitecture):
#                 print('result append (after)')
#                 is_processing = False
#                 context = {
#                     'is_processing': is_processing,
#                     'results': result,
#                 }
#                 yield context

        
#         return JsonResponse({'results': generate_resultats()})

#     context = {
#         'is_processing': is_processing,
#         'results': None,
#     }
#     return await sync_render(request, 'workflow/workflow_page.html', context)