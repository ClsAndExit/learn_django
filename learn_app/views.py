from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
def index(request):
    return HttpResponse(u'hello django')

def home(request):
    return render(request,'home.html')