# ReplicaCAD scenes
echo "Downloading ReplicaCAD scenes..."
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets

# HSSD scenes
if [ -d "data/scene_datasets/hssd-hab" ]; then
    echo "HSSD exists. Skipping HSSD download."
else
    echo "Downloading HSSD scenes..."
    mkdir -p data/scene_datasets/
    pushd data/scene_datasets/

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/hssd/hssd-hab
    cd hssd-hab
    git checkout 34b5f48c343cd7eb65d46e45402c85a004d77c92
    git lfs pull
    git lfs prune -f

    popd
fi

# ExtObjNav
echo "Downloading ExtObjNav..."
huggingface-cli download aielawady/ExtObjNav --local-dir-use-symlinks False --repo-type dataset --local-dir data/datasets/ExtObjNav_HSSD
huggingface-cli download aielawady/ExtObjNav_ReplicaCAD --local-dir-use-symlinks False --repo-type dataset --local-dir data/datasets/ExtObjNav_replicaCAD

# VC1 finetuned
echo "Downloading VC1 finetuned checkpoint..."
huggingface-cli download aielawady/vc1-smallObj --local-dir-use-symlinks False --local-dir model_ckpts/cls
