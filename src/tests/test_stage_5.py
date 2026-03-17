"""Tests for Stage 5 — Hydrodynamic Back-Tracking."""

import importlib
from datetime import datetime

import numpy as np
import pytest

backtrack = importlib.import_module("pipeline.05_backtrack")


class TestBacktrackParticle:
    """Tests for the RK4 particle back-tracking."""

    def test_particle_moves_backward(self):
        """A back-tracked particle should end at a different position."""
        traj = backtrack.backtrack_particle(
            start_lon=80.5, start_lat=7.5,
            start_time=datetime(2024, 1, 15),
            ocean_ds=None, wind_ds=None,
            hours=24, dt_hours=1.0,
        )

        assert len(traj) > 1
        start = traj[-1]
        end = traj[0]
        assert start[0] != end[0] or start[1] != end[1]

    def test_trajectory_length(self):
        """Trajectory should have expected number of points."""
        hours = 48
        dt = 1.0
        traj = backtrack.backtrack_particle(
            start_lon=80.5, start_lat=7.5,
            start_time=datetime(2024, 1, 15),
            ocean_ds=None, wind_ds=None,
            hours=hours, dt_hours=dt,
        )

        expected_points = int(hours / dt) + 1
        assert len(traj) == expected_points


class TestClusterEndpoints:
    """Tests for DBSCAN endpoint clustering."""

    def test_single_cluster(self):
        """Tightly grouped points should form a single cluster."""
        rng = np.random.default_rng(42)
        endpoints = [(80.5 + rng.normal(0, 0.1),
                      7.5 + rng.normal(0, 0.1))
                     for _ in range(20)]

        sources = backtrack.cluster_endpoints(endpoints, eps_degrees=0.5, min_samples=5)
        assert len(sources) >= 1
        assert sources[0]["source_probability"] > 0

    def test_empty_endpoints(self):
        """Empty endpoints should return empty list."""
        sources = backtrack.cluster_endpoints([], eps_degrees=0.5, min_samples=5)
        assert sources == []
