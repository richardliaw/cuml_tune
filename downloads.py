from datasets import prepare_dataset
import os
os.makedirs(os.path.expanduser("~/data"), exist_ok=True)
prepare_dataset("~/data", "airline", None)
