import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import munch
import yaml
import numpy as np
import torch

def mkdirs(*paths):
    """Makes a list of directories.

    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def set_dir_from_config(config):
    """Creates a output folder for experiment (and save config files).

    Naming format: {root (e.g. results)}/{tag (exp id)}/{seed}_{timestamp}_{git commit id}

    """
    # Make run folder (of a seed run for an experiment)
    seed = str(config.seed) if config.seed is not None else "-"
    timestamp = str(datetime.datetime.now().strftime("%b-%d-%H-%M-%S"))
    try:
        commit_id = subprocess.check_output(
            ["git", "describe", "--tags", "--always"]
        ).decode("utf-8").strip()
        commit_id = str(commit_id)
    except:
        commit_id = "-"
    run_dir = "seed{}_{}_{}".format(seed, timestamp, commit_id)
    # Make output folder.
    config.output_dir = os.path.join(config.output_dir, config.tag, run_dir)
    mkdirs(config.output_dir)
    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)
    # Save command.
    with open(os.path.join(config.output_dir, 'cmd.txt'), 'a') as file:
        file.write(" ".join(sys.argv) + "\n")
