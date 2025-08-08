from pathlib import Path
import modal

app = modal.App("pathfinding-training")

tag = "11.3.1-cudnn8-devel-ubuntu20.04"

# Modal image configuration
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
    .pip_install_from_requirements(
        "requirements.txt",
        extra_index_url="https://download.pytorch.org/whl/cu113",
        find_links="https://data.pyg.org/whl/torch-1.11.0+cu113.html",
    )
    .add_local_file(
        Path(__file__).parent / "model.py", remote_path="/root/model.py", copy=True
    )
    .add_local_file(
        Path(__file__).parent / "train.py", remote_path="/root/train.py", copy=True
    )
    .add_local_file(
        Path(__file__).parent / "data.py", remote_path="/root/data.py", copy=True
    )
)


# Cloud bucket mount configuration
# https://modal.com/docs/reference/modal.CloudBucketMount
cloud_bucket_mount = modal.CloudBucketMount(
    bucket_name="gnn-training-533335672401",
    oidc_auth_role_arn="arn:aws:iam::533335672401:role/wayflow-dev-modal",
)


# ----------------------------------------
# Training function
# This function is used to train the GNN model
@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    cpu=8.0,
    volumes={"/root/storage": cloud_bucket_mount},
)
async def train_gnn():
    from train import train

    train()


# ----------------------------------------
# Local entrypoint for running the function
# This allows the function to be run locally using `modal run`
@app.local_entrypoint()
def main():
    train_gnn.remote()
