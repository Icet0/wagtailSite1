import os
from django import forms
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render

from architecture.models import Architecture
from context.views import get_columns
from django.core.files import File

from .myFeatures.featuresAPI import addFeatures

from .models import *
from django.template.loader import render_to_string




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
def features_view(request):
    print('features_view')
    myArchitecture_pk = request.session.get('architecture_pk', None)
    myArchitecture = Architecture.objects.get(pk=myArchitecture_pk)
    working_directory = myArchitecture.contextModel.workingDirectory
    file_names = [file.file.name for file in working_directory.workingFiles.all()]
    
    files = []
    for file_name in file_names:
        files.append( file_name.split('/')[-1])
    print("num experiment ",working_directory.numExp)
    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'
                , 'spectral edge frequency': 'calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs)'
                , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'
                , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'
                , 'shannon entropy (dyad)' : 'calcShannonEntropyDyad(epoch, lvl, nt, nc, fs)'
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
                , 'calcERP' : 'calcERP(epoch)'
                , 'calcSampleEntropy' : 'calcSampleEntropy(epoch)'   #LONGUEEEEE
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
            for f in file_names:
                if file in f:
                    real_file = f
                    break
            real_file = os.path.join(settings.MEDIA_ROOT, real_file)
            print('real_file', real_file)
            print('functions_list', functions_list)
            path = addFeatures(real_file, functions_list)
            print('path', path)
            with open(path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(path)
                return response
            # Rendre le template pour afficher la réponse et le code JavaScript pour la redirection
            # redirect_delay = 2000  # Délai en millisecondes avant la redirection
            # redirect_url = 'dashboard_view'  # Remplacer par le nom de la vue vers laquelle vous souhaitez rediriger
            
            # context = {
            #     'response_content': response.content,
            #     'redirect_delay': redirect_delay,
            #     'redirect_url': redirect_url,
            # }
            
            # rendered_template = render_to_string('features/template.html', context)
            
            # return HttpResponse(rendered_template)



    form = ListForm(files=files,functions=functions.keys())

    
    context = {
        'title':'Features',
        'form': form,
    }
    return render(request, 'features/features.html',context)

