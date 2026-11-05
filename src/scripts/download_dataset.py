import os
import argparse
import sys
from dotenv import load_dotenv
from roboflow import Roboflow

def download_dataset(api_key, path):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-100").project("construction-safety-gsnvb").version(2)
    dataset = project.download("yolov8", location=path)
    print(f"Successed downloaded: {dataset.location}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="datasets/construction-safety")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print(f"ROBOFLOW_API_KEY not defined")
        sys.exit(1)

    try:
        download_dataset(api_key=api_key, path=args.path)
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)