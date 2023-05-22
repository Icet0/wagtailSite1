import json
from django.shortcuts import render

from myUtils.utils.Train import loadData
from prefect import flow, task

from architecture.models import Architecture
from myUtils.utils.Train import *
from asgiref.sync import sync_to_async

# Create your views here.

def workflow_view(request):
    is_processing = False  # Variable pour indiquer si le traitement est en cours

    architecture_pk = request.session['architecture_pk']
    myArchitecture = Architecture.objects.get(pk=architecture_pk)
    
    if request.method == 'POST':
        is_processing = True
        
        myFlow(request, myArchitecture)
        
        context = {
            'is_processing': is_processing,
        }
        return render(request, 'workflow/workflow_page.html', context)
    
    context = {
        'is_processing': is_processing,
        'results': None,
    }
    return render(request, 'workflow/workflow_page.html', context)

# async def workflow_view(request):
#     is_processing = False  # Variable pour indiquer si le traitement est en cours

#     architecture_pk = request.session['architecture_pk']
#     myArchitecture = await sync_to_async(Architecture.objects.get)(pk=architecture_pk)
    
#     if request.method == 'POST':
#         is_processing = True
        
#         async def generate_results():
#             async for result in myFlow(request, myArchitecture):
#                 yield result
        
#         context = {
#             'is_processing': is_processing,
#             'results': generate_results(),
#         }
#         return render(request, 'workflow/workflow_page.html', context)
    
#     context = {
#         'is_processing': is_processing,
#         'results': None,
#     }
#     return render(request, 'workflow/workflow_page.html', context)



@flow(log_prints=True, name="myWorkflow")
def myFlow(info,myArchitecture):

    #contextModel
    Features = pickle.loads(myArchitecture.contextModel.features)
    Labels = Features[:,-1].astype(int)
    print('Labels  : \n', Labels)
    Patients = pickle.loads(myArchitecture.contextModel.patients)
    Locations = pickle.loads(myArchitecture.contextModel.positions)
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
        Images = generate_images(len(Frequences.split(",")),Nombre_epochs,directory,Features,Locations)
        # yield Images
        save = settings.MEDIA_ROOT +'/uploads/'+ user.username + '/exp'+ str(numExp)
        Models = trainning(Images, Labels, Patients,model,save,training_split,batch_size,model_epochs,repetition)
        # yield Models
        print(Models)
        
        
    elif button_id == "2":
        #workflow 2
        print('workflow 2')
        # yield 'workflow 2'
    elif button_id == "3":
        #workflow 3
        print('workflow 3')
    else:
        print('button_id not found')