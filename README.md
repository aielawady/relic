# ReLIC: A recipe for 64k steps In-Context Reinforcement Learning for Embodied AI

This is the official implementation for "ReLIC: A recipe for 64k steps In-Context Reinforcement Learning for Embodied AI".


**Abstract**: Intelligent embodied agents need to quickly adapt to new scenarios by integrating long histories of experience into decision-making. For instance, a robot in an unfamiliar house initially wouldn't know the locations of objects needed for tasks and might perform inefficiently. However, as it gathers more experience, it should learn the layout and remember where objects are, allowing it to complete new tasks more efficiently. To enable such rapid adaptation to new tasks, we present ReLIC, a new approach for in-context reinforcement learning (RL) for embodied agents. With ReLIC, agents are capable of adapting to new environments using 64,000 steps of in-context experience with full attention mechanism while being trained through self-generated experience via RL. We achieve this by proposing a novel policy update scheme for on-policy RL called "partial updates" as well as a Sink-KV mechanism which enables effective utilization of long observation history for embodied agents. Our method outperforms a variety of meta-RL baselines in adapting to unseen houses in an embodied multi-object navigation task. In addition, we find that ReLIC is capable of few-shot imitation learning despite never being trained with expert demonstrations. We also provide a comprehensive analysis ReLIC, highlighting that the combination of large-scale RL training, the proposed partial updates scheme, and the Sink-KV are essential for effective in-context learning.


# Getting Started

* Clone the repo.
    ```bash
    git clone https://github.com/aielawady/relic.git
    cd relic
    ```
* Install relic
    ```bash
    conda create -n relic -y python=3.9
    conda activate relic
    conda install -y habitat-sim==0.3.0 withbullet headless -c conda-forge -c aihabitat

    pip install packaging ninja
    pip install -e .
    ```
* Change the dir to `relic`.
    ```bash
    cd relic
    ```
* Download ExtObjNav data (ExtObjNav dataset, scenes dataset and VC1 finetuned checkpoint).
    ```bash
    bash download_datasets.sh
    ```
* To run the training, update the variables in `run.sh` to config wandb and run it.
    ```bash
    bash run.sh
    ```
* To run the evaluation, update the evaluation variables in `run.sh` run it with `--eval` flag.
    ```bash
    bash run.sh --eval
    ```
* To process the evaluation result follow these steps.
    ```python
    from relic.evaluator import read_csvs, extract_episodes_data_from_df

    # Each row in the df is a trial.
    df = read_csvs("exp_data/eval_data/relic/relic_ExtObjNav/latest_*.csv")

    # `data` is a dict. The keys are the metrics names.
    # The values are 2D array with size (# Trials x # Episodes).
    # data["counts"] is a 1D array with size (# Episodes).
    # data["counts"][i] is the number of trials that have ith episode.
    data = extract_episodes_data_from_df(df)


    # To plot the data, create a mask to exclude the episodes that aren't
    # represented in all trials.
    mask = data["counts"] == data["counts"].max()

    # Then plot the data
    plt.plot(data["extobjnav_success"][:, mask].mean(axis=0))
    plt.xlabel("# of In-context Episodes")
    plt.ylabel("Success")
    ```

# Paper Experiments

You can change `CONFIG_NAME` in `run.sh` to run other experiments. This is the description of the configs available.

| Description | Config name | Figure | Task-Scene |
| -- | -- | -- | -- |
| ReLIC (64k context length) | baseline/relic_HSSD_64k | 3.b | ExtObjNav-HSSD |
| ReLIC | baseline/baselines_comparison/relic_HSSD | 1, 2.b, 3.a | ExtObjNav-HSSD |
| Single episode transformer | baseline/baselines_comparison/single_episode_HSSD | 1 | ExtObjNav-HSSD |
| Our method without inter-episode attention | baseline/baselines_comparison/relic_HSSD_no_iea | 1 | ExtObjNav-HSSD |
| Transformer-XL | baseline/baselines_comparison/trxl_HSSD | 1 | ExtObjNav-HSSD |
| RL2 | baseline/baselines_comparison/rl2 | 1 | ExtObjNav-HSSD |
| ReLIC | baseline/partial_updates_ablation/relic_replicaCAD | 2.a | ExtObjNav-ReplicaCAD |
| ReLIC without partial updates | baseline/partial_updates_ablation/relic_replicaCAD_no_partial_updates | 2.a | ExtObjNav-ReplicaCAD |
| ReLIC with Sink KV | baseline/sink_ablation/relic_replicaCAD_sink_kv | 11, 2.c | ExtObjNav-ReplicaCAD |
| ReLIC with Sink Token | baseline/sink_ablation/relic_replicaCAD_sink_token | 2.c | ExtObjNav-ReplicaCAD |
| ReLIC with Sink KV (only K is trainable) | baseline/sink_ablation/relic_replicaCAD_sink_kv0 | 11 | ExtObjNav-ReplicaCAD |
| ReLIC with Sink KV (only V is trainable) | baseline/sink_ablation/relic_replicaCAD_sink_k0v | 11 | ExtObjNav-ReplicaCAD |
| ReLIC with Sink KV (both are not trainable) | baseline/sink_ablation/relic_replicaCAD_sink_k0v0 | 11, 2.c | ExtObjNav-ReplicaCAD |
| ReLIC without Sink mechanism | baseline/sink_ablation/relic_replicaCAD_no_sink | 2.c | ExtObjNav-ReplicaCAD |

# Use ReLIC on your own task

To run ReLIC on the [Darkroom](https://github.com/jon--lee/decision-pretrained-transformer) task,

* Change the `CONFIG_NAME` to `baseline/custom_task/darkroom_relic` in `run.sh`.
* Then run it `bash run.sh`.

ReLIC works with `gym` compatible environments. Adding new task requires two files, the environment definition and the configuration.

We provide `darkroom` environment wrapper with detailed comments in [example.py](relic/envs/custom_envs/example.py). We also provide two configs, one for ReLIC (Multi-episode trials) [baseline/custom_task/darkroom_relic](relic/configs/baseline/custom_task/darkroom_relic.yaml) and one for single-episode training [baseline/custom_task/darkroom_single_episode](relic/configs/baseline/custom_task/darkroom_single_episode.yaml).

Once you define the two required files, you can follow the steps in the getting started section.
