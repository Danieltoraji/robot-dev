import sys, numpy as np
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
from sim.line_tracking_sim import run_full_test
run_full_test()
