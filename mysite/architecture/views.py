from django.shortcuts import render

from context.models import ContextModel

# Create your views here.
def architecture_view(request):
    
    contextModel = ContextModel.objects.get(pk=request.session['contextModel_pk'])
    print('contextModel', contextModel)
    return render(request, 'architecture/architecture_page.html', {"context": contextModel})