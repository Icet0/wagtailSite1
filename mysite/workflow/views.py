from django.shortcuts import render

# Create your views here.
def workflow_view(request):
    return render(request, 'workflow/workflow_page.html')