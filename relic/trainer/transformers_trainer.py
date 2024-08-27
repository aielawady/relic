import contextlib
import time
from typing import Dict, Tuple
import hydra
import numpy as np

import torch
from habitat.core.logging import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat.config.default import get_agent_config
from torch import distributed as distrib
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)

import torch.distributed as dist

from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
    REQUEUE,
    SAVE_STATE,
    DEFAULT_MAIN_ADDR,
    DEFAULT_PORT_RANGE,
    DEFAULT_PORT,
    SLURM_JOBID,
    RESUME_STATE_BASE_NAME,
    is_slurm_batch_job,
    is_slurm_job,
    get_ifname,
    get_distrib_size,
    get_main_addr,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer, get_device
from habitat_baselines.utils.timing import g_timer
from collections import defaultdict, deque
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet


import os
import random

from omegaconf import OmegaConf, open_dict

from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (
    init_distrib_slurm,
)
from relic.default_structured_configs import (
    SPLGeodiscMeasurementConfig,
    SoftSPLGeodiscMeasurementConfig,
)

from habitat_baselines.rl.ppo.evaluator import Evaluator


@baseline_registry.register_trainer(name="transformers")
class TransformersTrainer(PPOTrainer):
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # if self._is_distributed:
        #     raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint trained for {step_id} steps")
        else:
            ckpt_dict = {"config": None}

        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])
        config = OmegaConf.merge(self.config, config)
        with read_write(config):
            local_rank, world_rank, world_size = get_distrib_size()
            local_rank = 0
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
            config.habitat_baselines.torch_gpu_id = local_rank
            config.habitat_baselines.num_environments = 20
            config.habitat.seed = (
                config.habitat.seed
                + config.habitat_baselines.num_environments * world_rank
            )
            print(f"Worker {world_rank}, seed {config.habitat.seed}.")
            config.habitat.task.target_sampling_strategy = "object_type"
            config.habitat.environment.iterator_options.max_scene_repeat_steps = -1
            config.habitat.environment.iterator_options.max_scene_repeat_episodes = -1
            config.habitat_baselines.vector_env_factory._target_ = (
                "relic.envs.train_il_env_factory.HabitatVectorEnvFactory"
            )
            config.habitat.environment.iterator_options.shuffle = False

            evaluation_config = {
                "n_steps": 8200,
                "n_demos": -1,
                "fix_target_same_episode": False,
                "max_num_start_pos": -1,
            }

            evaluation_config.update(
                config.habitat_baselines.get("evaluation_config", {})
            )
            N_STEPS = evaluation_config["n_steps"]
            N_DEMOS = evaluation_config["n_demos"]
            config.habitat.task.fix_target_same_episode = evaluation_config[
                "fix_target_same_episode"
            ]
            config.habitat.task.max_num_start_pos = evaluation_config[
                "max_num_start_pos"
            ]
            MAX_N_EPS = evaluation_config.get("max_n_eps", -1)
            config.habitat_baselines.evaluate = self.config.habitat_baselines.evaluate

        with read_write(self.config):
            self.config.habitat_baselines.eval.evals_per_ep = 100
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
            self.config.habitat_baselines.torch_gpu_id = local_rank

        print("+" * 100)
        print(local_rank, world_rank, world_size)
        self._is_distributed = False
        self.device = torch.device("cuda", local_rank)

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(config.habitat.simulator, agent_i)

                agent_sensors = agent_config.sim_sensors
                extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
                with read_write(agent_sensors):
                    agent_sensors.update(extra_sensors)
                with read_write(config):
                    if config.habitat.gym.obs_keys is not None:
                        for render_view in extra_sensors.values():
                            if render_view.uuid not in config.habitat.gym.obs_keys:
                                if n_agents > 1:
                                    config.habitat.gym.obs_keys.append(
                                        f"{agent_name}_{render_view.uuid}"
                                    )
                                else:
                                    config.habitat.gym.obs_keys.append(render_view.uuid)

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            self._agent.load_state_dict(ckpt_dict)

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        assert isinstance(evaluator, Evaluator)
        evaluator.evaluate_agent(
            self._agent,
            self.envs,
            self.config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self.obs_transforms,
            self._env_spec,
            self._rank0_keys,
            n_steps=N_STEPS,
            n_demos=N_DEMOS,
            max_n_eps=MAX_N_EPS,
            suffix=f"rank{world_rank}",
            is_memory=config.habitat_baselines.rl.policy.main_agent.transformer_config.model_name
            == "transformerxl",
        )

        self.envs.close()

    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()
        if self.config.habitat_baselines.separate_envs_and_policy:
            with read_write(self.config):
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = 1

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                    % (torch.cuda.device_count() - torch.distributed.get_world_size())
                    + torch.distributed.get_world_size()
                    if self.config.habitat_baselines.separate_envs_and_policy
                    else local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(non_scalar_metric_root)
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        self._init_envs()

        self.device = get_device(self.config)

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        logs_dir = os.path.dirname(self.config.habitat_baselines.log_file)
        os.makedirs(logs_dir, exist_ok=True)
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.updater.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            assert (
                self._encoder is not None
            ), "Visual encoder is not specified for this actor"
            with inference_mode():
                batch[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY] = self._encoder(
                    batch
                )

        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )

        self.t_start = time.time()

    def _reset_envs_custom(self):
        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            assert (
                self._encoder is not None
            ), "Visual encoder is not specified for this actor"
            with inference_mode():
                batch[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY] = self._encoder(
                    batch
                )

        self._agent.rollouts.insert_first_observations(batch, reset_memory=True)

        self.current_episode_reward *= 0

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(
        self, loss_from_step=0, last_update=True, percent_envs_update=None, slice_from=0
    ):
        with inference_mode():
            step_batch = self._agent.rollouts.get_last_step()
            step_batch_lens = {
                k: v for k, v in step_batch.items() if k.startswith("index_len")
            }

            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )
            del step_batch

        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        self._agent.train()

        losses = self._agent.updater.update(
            self._agent.rollouts,
            loss_from_step,
            percent_envs_update=percent_envs_update,
            slice_from=slice_from,
        )
        if last_update:
            self._agent.rollouts.after_update()
            self._agent.after_update()

        if hasattr(self._agent, "_step_sched"):
            self._agent._step_sched(self.num_steps_done)

        return losses

    def _compute_actions_and_step_envs_il(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )
        actions = self.envs.call(["best_action"] * self.envs.num_envs)
        for index_env in range(env_slice.start, env_slice.stop):
            act = actions[index_env]

            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                act = np.clip(
                    act.numpy(),
                    self._env_spec.action_space.low,
                    self._env_spec.action_space.high,
                )
            else:
                act = act
            self.envs.async_step_at(index_env, act)
            # actions.append(actions)

        with g_timer.avg_time("trainer.obs_insert"):
            self._agent.rollouts.insert(
                actions=np.asarray(actions)[..., None],
                buffer_index=buffer_index,
            )

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        if self._ppo_cfg.get("shortest_path_follower", False):
            self._compute_actions_and_step_envs_il(buffer_index=buffer_index)
        else:
            super()._compute_actions_and_step_envs(buffer_index=buffer_index)

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """
        if self.config.habitat_baselines.rl.ppo.full_updates_per_rollout > 0:
            assert (
                self.config.habitat_baselines.rl.ppo.updates_per_rollout
                % self.config.habitat_baselines.rl.ppo.full_updates_per_rollout
                == 0
            )
            RATIO_P2F_UPDATES = (
                self.config.habitat_baselines.rl.ppo.updates_per_rollout
                // self.config.habitat_baselines.rl.ppo.full_updates_per_rollout
            )
        else:
            RATIO_P2F_UPDATES = 0

        UPDATE_EVERY_STEPS = (
            self.config.habitat_baselines.rl.ppo.num_steps
            // self.config.habitat_baselines.rl.ppo.updates_per_rollout
        )

        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            # self.running_episode_stats = requeue_stats["running_episode_stats"]
            # self.window_episode_stats.update(
            #     requeue_stats["window_episode_stats"]
            # )
            logger.info(
                "Last checkpoint percent: {} - {}".format(
                    self._last_checkpoint_percent, self.percent_done()
                )
            )
            self._last_checkpoint_percent = min(
                self._last_checkpoint_percent, self.percent_done()
            )
            logger.info("Resuming from previous checkpoint")
            logger.info(
                "Total num steps: {}/{} - {}".format(
                    self.num_steps_done,
                    self.config.habitat_baselines.total_num_steps,
                    self.percent_done(),
                )
            )
            logger.info(
                "[Override] Last checkpoint percent: {} - {}".format(
                    self._last_checkpoint_percent, self.percent_done()
                )
            )
            resume_run_id = requeue_stats.get("run_id", None)

        if self.config.habitat_baselines.rl.ppo.get("init_checkpoint", ""):
            print(
                f"loading from {self.config.habitat_baselines.rl.ppo.init_checkpoint}..."
            )
            resume_state_ = load_resume_state(
                self.config.habitat_baselines.rl.ppo.init_checkpoint
            )
            self._agent.load_state_dict(resume_state_)

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            rollouts_count = 0
            switched_envs = 0
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                last_update_step = self._agent._rollouts.old_context_length
                with g_timer.avg_time("trainer.rollout_collect"):
                    for buffer_index in range(self._agent.nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)
                    for step in range(self._ppo_cfg.num_steps):
                        is_last_step = (
                            self.should_end_early(step + 1)
                            or (step + 1) == self._ppo_cfg.num_steps
                        )

                        for buffer_index in range(self._agent.nbuffers):
                            count_steps_delta += self._collect_environment_result(
                                buffer_index
                            )

                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(buffer_index)

                        if (step + 1) % UPDATE_EVERY_STEPS == 0 and not is_last_step:
                            IS_FULL_UPDATE = (
                                RATIO_P2F_UPDATES != 0
                                and (step + 1)
                                % (RATIO_P2F_UPDATES * UPDATE_EVERY_STEPS)
                                == 0
                            )
                            update_from = (
                                0
                                if IS_FULL_UPDATE
                                or self._ppo_cfg.slice_in_partial_update
                                else last_update_step
                            )
                            slice_from = (
                                0
                                if IS_FULL_UPDATE
                                or not self._ppo_cfg.slice_in_partial_update
                                else last_update_step
                            )

                            losses = self._update_agent(
                                loss_from_step=update_from,
                                last_update=False,
                                percent_envs_update=self._ppo_cfg.percent_envs_update,
                                slice_from=slice_from,
                            )
                            self.num_updates_done += 1
                            losses = self._coalesce_post_step(
                                losses,
                                count_steps_delta,
                            )

                            self._training_log(writer, losses, prev_time)
                            count_steps_delta = 0

                            if rank0_only() and self._should_save_resume_state():
                                requeue_stats = dict(
                                    count_checkpoints=count_checkpoints,
                                    num_steps_done=self.num_steps_done,
                                    num_updates_done=self.num_updates_done,
                                    _last_checkpoint_percent=self._last_checkpoint_percent,
                                    prev_time=(time.time() - self.t_start) + prev_time,
                                    running_episode_stats=self.running_episode_stats,
                                    window_episode_stats=dict(
                                        self.window_episode_stats
                                    ),
                                    run_id=writer.get_run_id(),
                                )

                                save_resume_state(
                                    dict(
                                        **self._agent.get_resume_state(),
                                        config=self.config,
                                        requeue_stats=requeue_stats,
                                    ),
                                    self.config,
                                )

                            if EXIT.is_set():
                                profiling_wrapper.range_pop()  # train update

                                self.envs.close()

                                requeue_job()

                                return

                            self._agent.eval()
                            last_update_step = (
                                self._agent._rollouts.current_rollout_step_idx
                            )

                            if self._ppo_cfg.shuffle_old_episodes:
                                self._agent.rollouts.shuffle_episodes()

                            if (
                                self._ppo_cfg.update_stale_kv
                                or self._ppo_cfg.update_stale_values
                                or self._ppo_cfg.update_stale_action_probs
                            ):
                                with torch.inference_mode():
                                    for i in range(self.envs.num_envs):
                                        batch = self._agent._rollouts.get_context_step(
                                            env_id=i, n_steps=last_update_step + 2
                                        )

                                        (
                                            value_preds,
                                            action_log_probs,
                                            recurrent_hidden_states,
                                        ) = self._agent._actor_critic.get_value_action_prob(
                                            batch["observations"],
                                            None,
                                            batch["actions"],
                                            batch["prev_actions"],
                                            batch["masks"],
                                            batch["rnn_build_seq_info"],
                                            full_rnn_state=True,
                                        )
                                        value_preds = value_preds.unflatten(
                                            0,
                                            tuple(batch["rnn_build_seq_info"]["dims"]),
                                        )
                                        action_log_probs = action_log_probs.unflatten(
                                            0,
                                            tuple(batch["rnn_build_seq_info"]["dims"]),
                                        )

                                        if not self._ppo_cfg.update_stale_kv:
                                            recurrent_hidden_states = None

                                        if not self._ppo_cfg.update_stale_values:
                                            value_preds = None

                                        if not self._ppo_cfg.update_stale_action_probs:
                                            action_log_probs = None

                                        self._agent._rollouts.update_context_data(
                                            value_preds,
                                            action_log_probs,
                                            recurrent_hidden_states,
                                            env_id=i,
                                            n_steps=last_update_step + 2,
                                        )

                                    del (
                                        batch,
                                        value_preds,
                                        action_log_probs,
                                        recurrent_hidden_states,
                                    )
                        if is_last_step:
                            break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                losses = self._update_agent(
                    0
                    if self._ppo_cfg.full_updates_per_rollout > 0
                    else last_update_step,
                    last_update=True,
                )

                self._agent.eval()
                if self._agent._rollouts.context_length > 0 and (
                    self._ppo_cfg.update_stale_kv
                    or self._ppo_cfg.update_stale_values
                    or self._ppo_cfg.update_stale_action_probs
                ):
                    with torch.inference_mode():
                        for i in range(self.envs.num_envs):
                            batch = self._agent._rollouts.get_context_step(env_id=i)

                            (
                                value_preds,
                                action_log_probs,
                                recurrent_hidden_states,
                            ) = self._agent._actor_critic.get_value_action_prob(
                                batch["observations"],
                                None,
                                batch["actions"],
                                batch["prev_actions"],
                                batch["masks"],
                                batch["rnn_build_seq_info"],
                                full_rnn_state=True,
                            )
                            value_preds = value_preds.unflatten(
                                0, tuple(batch["rnn_build_seq_info"]["dims"])
                            )
                            action_log_probs = action_log_probs.unflatten(
                                0, tuple(batch["rnn_build_seq_info"]["dims"])
                            )

                            if not self._ppo_cfg.update_stale_kv:
                                recurrent_hidden_states = None

                            if not self._ppo_cfg.update_stale_values:
                                value_preds = None

                            if not self._ppo_cfg.update_stale_action_probs:
                                action_log_probs = None

                            self._agent._rollouts.update_context_data(
                                value_preds,
                                action_log_probs,
                                recurrent_hidden_states,
                                env_id=i,
                            )

                    del batch, value_preds, action_log_probs, recurrent_hidden_states

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)
                rollouts_count += 1

                if self.config.habitat_baselines.call_after_update_env:
                    if self._ppo_cfg.shift_scene_every <= 0:
                        self.envs.call(["after_update"] * self.envs.num_envs)
                    elif self._ppo_cfg.get("shift_scene_staggered", True):
                        shift_every_n_rollouts = np.ceil(
                            self._ppo_cfg.shift_scene_every / self._ppo_cfg.num_steps
                        )
                        to_switch = int(
                            np.ceil(
                                self.envs.num_envs
                                * (rollouts_count / shift_every_n_rollouts)
                            )
                        )
                        for iio in range(switched_envs, to_switch):
                            self.envs.call_at(iio, "after_update")
                        rollouts_count %= shift_every_n_rollouts
                        if rollouts_count == 0:
                            switched_envs = 0
                        else:
                            switched_envs = min(iio + 1, self.envs.num_envs)
                    else:
                        shift_every_n_rollouts = np.ceil(
                            self._ppo_cfg.shift_scene_every / self._ppo_cfg.num_steps
                        )
                        if (rollouts_count + 1) % shift_every_n_rollouts == 0:
                            self.envs.call(["after_update"] * self.envs.num_envs)
                            rollouts_count = 0
                if self._ppo_cfg.get("force_env_reset_every", -1) > 0:
                    force_reset_n_rollouts = np.ceil(
                        self._ppo_cfg.force_env_reset_every / self._ppo_cfg.num_steps
                    )
                else:
                    force_reset_n_rollouts = -1

                if self.config.habitat_baselines.reset_envs_after_update or (
                    force_reset_n_rollouts > 0
                    and (rollouts_count + 1) % force_reset_n_rollouts == 0
                ):
                    self._reset_envs_custom()

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    @rank0_only
    def _training_log(self, writer, losses: Dict[str, float], prev_time: int = 0):
        deltas = {
            k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        if self._agent._lr_scheduler:
            writer.add_scalar(
                "learning_rate",
                self._agent._lr_scheduler.get_last_lr()[0],
                self.num_steps_done,
            )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # Log perf metrics.
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        # log stats
        if self.num_updates_done % self.config.habitat_baselines.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames {self.num_steps_done}"
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
            perf_stats_str = " ".join(
                [f"{k}: {v.mean:.3f}" for k, v in g_timer.items()]
            )
            logger.info(f"\tPerf Stats: {perf_stats_str}")
            if self.config.habitat_baselines.should_log_single_proc_infos:
                for k, v in self._single_proc_infos.items():
                    logger.info(f" - {k}: {np.mean(v):.3f}")
