from django.urls import path
from .views import PipelineRunListCreateView, PipelineRunDetailView

urlpatterns = [
    path('pipeline/runs/', PipelineRunListCreateView.as_view(), name='pipeline-run-list-create'),
    path('pipeline/runs/<uuid:pk>/', PipelineRunDetailView.as_view(), name='pipeline-run-detail'),
]
