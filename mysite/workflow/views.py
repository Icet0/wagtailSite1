from django.shortcuts import render

# Create your views here.
def workflow_view(request):
    
    if request.method == 'POST':
        button_id = request.POST.get('button_id')
        print('button_id', button_id)
    return render(request, 'workflow/workflow_page.html')