"""Tests for Stage 1 — Satellite Data Ingestion."""

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ingest = importlib.import_module("pipeline.01_ingest")


class TestSearchScenes:
    """Tests for the STAC search functionality."""

    @patch.object(ingest, "Client")
    def test_search_returns_scenes(self, mock_client_cls):
        """Search should return sorted scene dicts on success."""
        mock_item = MagicMock()
        mock_item.id = "S2A_TEST_001"
        mock_item.datetime = "2024-01-15T10:00:00Z"
        mock_item.properties = {"eo:cloud_cover": 5}
        mock_item.bbox = [80.0, 7.0, 81.0, 8.0]
        mock_item.assets = {"B02": MagicMock(href="https://example.com/B02.tif")}
        mock_item.geometry = {"type": "Polygon", "coordinates": []}

        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client_cls.open.return_value.search.return_value = mock_search

        result = ingest.search_scenes((80.0, 7.0, 81.0, 8.0), "2024-01-01", "2024-01-31")

        assert len(result) == 1
        assert result[0]["id"] == "S2A_TEST_001"
        assert result[0]["cloud_cover"] == 5

    @patch.object(ingest, "Client")
    def test_search_no_scenes_raises(self, mock_client_cls):
        """Search with no results should raise RuntimeError."""
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client_cls.open.return_value.search.return_value = mock_search

        with pytest.raises(RuntimeError, match="No Sentinel-2"):
            ingest.search_scenes((0, 0, 1, 1), "2024-01-01", "2024-01-31")


class TestDownloadScene:
    """Tests for scene download functionality."""

    def test_download_creates_directory(self, tmp_path):
        """Download should create the scene directory."""
        scene = {
            "id": "test_scene",
            "assets": {},
        }
        scene_dir, band_paths = ingest.download_scene(scene, tmp_path)
        assert scene_dir.exists()
        assert (scene_dir / "metadata.json").exists()
