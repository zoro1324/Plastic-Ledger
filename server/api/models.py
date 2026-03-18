from django.db import models
import uuid

class PipelineRun(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bbox = models.CharField(max_length=255, help_text="Comma separated: lon_min,lat_min,lon_max,lat_max")
    target_date = models.DateField(help_text="Target date for the run in YYYY-MM-DD")
    cloud_cover = models.IntegerField(default=20)
    backtrack_days = models.IntegerField(default=30)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    output_dir = models.CharField(max_length=500, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    # Store summary stats as JSON and error messages if any
    summary = models.JSONField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Run {self.id} - {self.status}"
    
    class Meta:
        ordering = ['-created_at']
