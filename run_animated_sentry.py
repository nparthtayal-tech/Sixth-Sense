#!/usr/bin/env python3
"""Launcher for the interactive SensorSentry Matplotlib simulation with phone hardware-in-the-loop."""

import sys
from pathlib import Path

# Add the Sixth-Sense-main directory to sys.path
sim_dir = Path(__file__).parent / "sixth sense parth" / "Sixth-Sense-main"
if sim_dir.exists():
    sys.path.insert(0, str(sim_dir))

from demo_animated_sentry import main

if __name__ == "__main__":
    main()
