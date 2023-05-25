import os
from django.conf import settings
from django.shortcuts import render
from .models import Fichier

# Create your views here.

def dashboard_view(request):
    user = request.user
    path = settings.MEDIA_ROOT+"/uploads/" + str(user.username)
    
    fichiers = os.listdir(path)
    files = []
    for fichier in fichiers:
        chemin_complet = os.path.join(path, fichier)
        est_repertoire = os.path.isdir(chemin_complet)
        files.append(Fichier.objects.get_or_create(nom=fichier, est_repertoire=est_repertoire,user=user))
    fichiers = Fichier.objects.filter(user=request.user)
    
    return render(request, 'dashboard/dashboard.html', {"fichiers": fichiers})