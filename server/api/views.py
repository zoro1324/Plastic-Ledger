import sys
import threading
from pathlib import Path
from django.utils import timezone
from rest_framework import generics
from .models import PipelineRun
from .serializers import PipelineRunSerializer

# Add src/ to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from pipeline.run_pipeline import run_pipeline

def execute_pipeline(run_id):
    """Background task to run the ML pipeline."""
    try:
        run_instance = PipelineRun.objects.get(id=run_id)
        run_instance.status = 'RUNNING'
        run_instance.save()
        
        output_dir = PROJECT_ROOT / 'data' / 'runs' / str(run_instance.id)
        model_path = PROJECT_ROOT / 'models' / 'runs' / 'marida_v1' / 'best_model.pth'
        config_path = SRC_DIR / 'config' / 'config.yaml'
        
        run_instance.output_dir = str(output_dir)
        run_instance.save()
        
        bbox = tuple(float(x) for x in run_instance.bbox.split(','))
        
        summary = run_pipeline(
            bbox=bbox,
            target_date=str(run_instance.target_date),
            output_dir=output_dir,
            model_path=model_path,
            cloud_cover=run_instance.cloud_cover,
            backtrack_days=run_instance.backtrack_days,
            config_path=str(config_path)
        )
        
        has_failures = len(summary.get('stages_failed', [])) > 0
        run_instance.status = 'FAILED' if has_failures else 'COMPLETED'
        run_instance.summary = summary
        run_instance.completed_at = timezone.now()
        run_instance.save()
        
    except Exception as e:
        run_instance.status = 'FAILED'
        run_instance.error_message = str(e)
        run_instance.completed_at = timezone.now()
        run_instance.save()

class PipelineRunListCreateView(generics.ListCreateAPIView):
    queryset = PipelineRun.objects.all()
    serializer_class = PipelineRunSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        thread = threading.Thread(target=execute_pipeline, args=(instance.id,))
        thread.daemon = True
        thread.start()

class PipelineRunDetailView(generics.RetrieveAPIView):
    queryset = PipelineRun.objects.all()
    serializer_class = PipelineRunSerializer
