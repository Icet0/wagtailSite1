from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render

from loadingData.models import workingDirectory

# Create your views here.
def context_view(request):
    # Votre logique de traitement du formulaire ici
    working_directory_pk = request.session.get('working_directory_pk')
    working_directory = get_object_or_404(workingDirectory, pk=working_directory_pk)
    print("working_directory",working_directory.csv_file)
    if request.method == 'POST':
        # Traitements à effectuer

        # Redirection vers une autre page et modèle
        return HttpResponse('POST '+str(working_directory))
    else:
        
        return HttpResponse('GET'+str(working_directory))