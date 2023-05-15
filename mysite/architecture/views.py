import importlib
import inspect
import sys
from django import forms
from django.http import HttpResponse
from django.shortcuts import redirect, render

from context.models import ContextModel
from .models import Architecture



class ArchitectureForm(forms.ModelForm):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['training_split'].initial = 0.8
        self.fields['batch_size'].initial = 32
        self.fields['model_epochs'].initial = 10
        self.fields['repetition'].initial = 3
        self.fields['evaluation_metrics'].initial = 'accuracy'
        
        self.fields['model_type'] = forms.ChoiceField(
            choices=[(md, lb) for md,lb in get_models()],
            widget=forms.Select(attrs={'class': 'form-control'})
        )
        self.fields['model_type'].initial = get_models()[0]
        print(self.fields['model_type'].initial)
        
    class Meta:
        model = Architecture
        fields = ('model_type', 'training_split', 'batch_size', 'model_epochs', 'repetition', 'evaluation_metrics')
        widgets = {
            # 'model_type': forms.Select(attrs={'class': 'form-control'}),
            'training_split': forms.NumberInput(attrs={'class': 'form-control','min': '0', 'max': '1'}),
            'batch_size': forms.NumberInput(attrs={'class': 'form-control', 'min': '1'}),
            'model_epochs': forms.NumberInput(attrs={'class': 'form-control','min': '1'}),
            'repetition': forms.NumberInput(attrs={'class': 'form-control', 'min': '1'}),
            'evaluation_metrics': forms.Select(attrs={'class': 'form-control'}),
        }
        
        
        
# Create your views here.
def architecture_view(request):
    
    contextModel = ContextModel.objects.get(pk=request.session['contextModel_pk'])
    print('contextModel', contextModel)
    if request.method == 'POST':
        form = ArchitectureForm(request.POST)
        if form.is_valid():
            model_type = form.cleaned_data['model_type']
            training_split = form.cleaned_data['training_split']
            batch_size = form.cleaned_data['batch_size']
            model_epochs = form.cleaned_data['model_epochs']
            repetition = form.cleaned_data['repetition']
            evaluation_metrics = form.cleaned_data['evaluation_metrics']
            
            architecture = Architecture.objects.create(model_type=model_type, training_split=training_split, batch_size=batch_size,
                                                       model_epochs=model_epochs, repetition=repetition, evaluation_metrics=evaluation_metrics,
                                                       contextModel=contextModel)
            architecture.save()
            
            request.session['architecture_pk'] = architecture.pk
            return redirect('workflow_view')
    else:
        form = ArchitectureForm()
    return render(request, 'architecture/architecture_page.html', {'form': form})




def get_models():
    """
    Get the columns of the csv file
    """
    #for each class in the Models.py file, return the name of the class
    models = []
    module = importlib.import_module("myUtils.utils.Models")
    print("module", module)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            models.append((name.lower(), name))
    return models