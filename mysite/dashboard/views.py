import os
from django.conf import settings
from django.shortcuts import render
from .models import Fichier
from django.http import HttpResponse, JsonResponse

# Create your views here.
def download_file(request):
    fichier_id = request.GET.get('file_id', None)
    try:
        print("fichier_id : ",fichier_id)
        fichier = Fichier.objects.get(id=fichier_id)
        print("fichier : ",fichier)
        if fichier.est_repertoire:
            return JsonResponse({'error': 'Ce n\'est pas un fichier'})
        else:
            hierarchie = []
            f = fichier
            while fichier.parent is not None:
                hierarchie.append(fichier.parent)
                fichier = fichier.parent
            fichier_path = settings.MEDIA_ROOT + '/uploads/' + str(fichier.user.username) + '/'
            hierarchie.reverse()  # Inverse l'ordre de la liste pour commencer par le dossier racine
            for dossier in hierarchie:
                fichier_path += dossier.nom + '/'
            fichier_path += f.nom
            with open(fichier_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(fichier_path)
                return response
    except Fichier.DoesNotExist:
        return JsonResponse({'error': 'Ce fichier n\'existe pas'})

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
    
    Fichier.objects.filter(user=request.user).delete()

    files = creer_fichiers_recursif(path, None, user)
    print("files : \n",files)
    fichiers = Fichier.objects.filter(user=request.user, parent=None)
    print("fichier : \n",fichiers)
    models = Fichier.objects.filter(user=request.user, nom = "Models")
    print("models : \n",models)
    
    return render(request, 'dashboard/dashboard.html', {"fichiers": fichiers, "models": models})