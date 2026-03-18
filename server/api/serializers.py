from rest_framework import serializers
from .models import PipelineRun

class PipelineRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = PipelineRun
        fields = '__all__'
        read_only_fields = ['id', 'status', 'output_dir', 'created_at', 'completed_at', 'summary', 'error_message']

    def validate_bbox(self, value):
        parts = value.split(',')
        if len(parts) != 4:
            raise serializers.ValidationError("bbox must have exactly 4 comma-separated values (lon_min,lat_min,lon_max,lat_max)")
        try:
            [float(p) for p in parts]
        except ValueError:
            raise serializers.ValidationError("All 4 bbox values must be numbers")
        return value
