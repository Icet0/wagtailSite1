import os
from django.conf import settings
from django.shortcuts import render
from .models import Fichier
from django.http import JsonResponse

# Create your views here.

def get_enfants_view(request):
    fichier_id = request.GET.get('fichier_id', None)
    try:
        fichier = Fichier.objects.get(id=fichier_id)
        enfants = fichier.enfants.all().values('id', 'nom', 'est_repertoire')
        return JsonResponse({'enfants': list(enfants)})
    except Fichier.DoesNotExist:
        return JsonResponse({'enfants': []})


def creer_fichiers_recursif(dossier, parent, user):
    files = []
    for nom in os.listdir(dossier):
        chemin_complet = os.path.join(dossier, nom)
        est_repertoire = os.path.isdir(chemin_complet)
        fichier, _ = Fichier.objects.get_or_create(nom=nom, est_repertoire=est_repertoire, user=user)
        fichier.parent = parent
        files.append(fichier)
        if est_repertoire:
            enfants = creer_fichiers_recursif(chemin_complet, fichier, user)
            fichier.enfants.set(enfants)
        fichier.save()

    return files  
            
def dashboard_view(request):
    user = request.user
    path = settings.MEDIA_ROOT+"/uploads/" + str(user.username)
    
    Fichier.objects.all().delete()

    files = creer_fichiers_recursif(path, None, user)
    print("files : \n",files)
    fichiers = Fichier.objects.filter(user=request.user, parent=None)
    print("fichier : \n",fichiers)
    
    return render(request, 'dashboard/dashboard.html', {"fichiers": fichiers})