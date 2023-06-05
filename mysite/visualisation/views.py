import os
from django import forms
from django.conf import settings
from django.shortcuts import render

from architecture.models import Architecture

from .models import VisualisationModel



class ListForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        visualisation = kwargs.pop('visualisation', [])
        files = kwargs.pop('files', [])

        super(ListForm, self).__init__(*args, **kwargs)

        self.fields['files'] = forms.ChoiceField(
            choices=[(file, file) for file in files],
            widget=forms.Select(attrs={'class': 'form-control', 'size': '5'})
        )
        self.fields['files'].initial = files
        
        self.fields['visualisation'] = forms.MultipleChoiceField(
            choices=[(function, function) for function in visualisation],
            widget=forms.SelectMultiple(attrs={'class': 'form-control', 'size': '10'})
        )
        self.fields['visualisation'].initial = visualisation
        
        
    class Meta:
        model = VisualisationModel
        fields = ['files','visualisation']
        widgets = {

        }

# Create your views here.
def visualisation_view(request):
    
    myArchitecture_pk = request.session.get('architecture_pk', None)
    myArchitecture = Architecture.objects.get(pk=myArchitecture_pk)
    working_directory = myArchitecture.contextModel.workingDirectory
    file_names = [file.file.name for file in working_directory.workingFiles.all()]
    
    files = []
    for file_name in file_names:
        files.append( file_name.split('/')[-1])
    print("num experiment ",working_directory.numExp)
    
    visualisations = { 'VISU 1': 'visu1(data)'
                , 'VISU 1': 'visu1(data)'
                , 'VISU 2': 'visu2(data)'
                , 'VISU 1': 'visu1(data)'
                , 'VISU 1': 'visu1(data)'
    }
    if request.method == 'POST':
        
        form = ListForm(request.POST, files=files, visualisation=visualisations)
        if form.is_valid():
            file = form.cleaned_data['files']
            visualisations = form.cleaned_data['visualisation']
            print('files', files)
            for f in file_names:
                if file in f:
                    real_file = f
                    break
            real_file = os.path.join(settings.MEDIA_ROOT, real_file)
            print('real_file', real_file)
            for function in visualisations:
                print('function', function)



    form = ListForm(files=files,visualisation=visualisations.keys())

    
    context = {
        'title':'Features',
        'form': form,
    }
    
    return render(request, 'visualisation/visualisation.html', context)