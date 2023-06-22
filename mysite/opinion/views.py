from django.shortcuts import render

# Create your views here.
def opinion_view(request):
    return render(request, 'opinion/opinion.html')
