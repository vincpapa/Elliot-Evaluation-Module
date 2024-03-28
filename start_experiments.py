from elliot.run import run_experiment
import glob
import os

# parser = argparse.ArgumentParser(description="Run sample main.")
# parser.add_argument('--config', type=str, default='svdgcn')
# args = parser.parse_args()
per_user = False
conf_list = glob.glob(os.sep.join(["config_files/DEF_compute_metrics_ml-100k_LightGCN.yml"]))
for conf_file in conf_list:
    run_experiment(f"{conf_file}", per_user)
