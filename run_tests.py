import os
import sys
import subprocess

"""
execute run_config with all configs in configs/tests
and observe stdout and stderr

these are basically integration tests
"""

def run_config(config):
    print("Running config: ", config)
    p = subprocess.Popen([sys.executable, "run_tests.py", config], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = p.communicate(timeout=10)  # Set a timeout, for example, 60 seconds
        print("stdout: ", stdout)
        print("stderr: ", stderr)
    except subprocess.TimeoutExpired:
        print(f"Config {config} timed out.")
        p.kill()

#get all configs in configs/tests
configs = os.listdir("configs/tests")
configs = [os.path.join("configs/tests", config) for config in configs]
#run all configs
for config in configs:
    run_config(config)
