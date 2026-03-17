"""Tests for Stage 6 — Source Attribution."""

import importlib

import pytest

attribute = importlib.import_module("pipeline.06_attribute")


class TestScoringFunctions:
    """Tests for individual scoring dimensions."""

    def test_fishing_heuristic_no_token(self):
        """Without GFW token, should return a heuristic score."""
        result = attribute.score_fishing(
            source_bbox=(80.0, 7.0, 81.0, 8.0),
            date_start="2024-01-01",
            date_end="2024-01-31",
            gfw_token=None,
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_shipping_heuristic(self):
        """Shipping score should work without reference data."""
        result = attribute.score_shipping(
            source_bbox=(80.0, 7.0, 81.0, 8.0),
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_river_scoring(self):
        """River scoring should find nearby rivers for Indian Ocean."""
        result = attribute.score_river(
            source_bbox=(80.0, 7.0, 81.0, 8.0),
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "nearest_river" in result


class TestCompositeAttribution:
    """Tests for the composite attribution scoring."""

    def test_weighted_score(self):
        """Composite score should be a weighted sum of individual scores."""
        scores = {
            "fishing": {"score": 0.8},
            "industrial": {"score": 0.5},
            "shipping": {"score": 0.3},
            "river": {"score": 0.1},
        }
        weights = {"fishing": 0.4, "industrial": 0.3, "shipping": 0.2, "river": 0.1}

        result = attribute.compute_attribution(scores, weights)

        expected = 0.4 * 0.8 + 0.3 * 0.5 + 0.2 * 0.3 + 0.1 * 0.1
        assert result["attribution_score"] == pytest.approx(expected, abs=1e-6)
        assert result["source_type"] == "fishing"

    def test_empty_scores(self):
        """Missing scores should default to zero."""
        result = attribute.compute_attribution(
            scores={},
            weights={"fishing": 0.5, "industrial": 0.5},
        )
        assert result["attribution_score"] == 0.0
