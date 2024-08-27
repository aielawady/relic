from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
    get_rollout_obs_space,
)
import gym.spaces as spaces
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat import logger

from torch import optim
from bisect import bisect_right
import torch
from habitat_baselines.rl.ppo.ppo import PPO

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CustomSequentialLR(optim.lr_scheduler.SequentialLR):
    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0:
            scheduler.step(self.last_epoch - self._milestones[idx - 1])
        else:
            scheduler.step(self.last_epoch)
        self._last_lr = scheduler.get_last_lr()


@baseline_registry.register_agent_access_mgr
class TransformerSingleAgentAccessMgr(SingleAgentAccessMgr):
    def _init_policy_and_updater(self, lr_schedule_fn, resume_state):
        self._actor_critic = self._create_policy()

        self._updater = self._create_updater(self._actor_critic)

        if self._updater.optimizer is None:
            self._lr_scheduler = None
        else:
            scheds = []
            milestones = []
            if self._ppo_cfg.warmup:
                scheds.append(
                    optim.lr_scheduler.LinearLR(
                        self._updater.optimizer,
                        start_factor=self._ppo_cfg.warmup_start_factor,
                        total_iters=self._ppo_cfg.warmup_total_iters,
                        end_factor=self._ppo_cfg.warmup_end_factor,
                    )
                )
                milestones.append(self._ppo_cfg.warmup_total_iters)

            if self._ppo_cfg.lr_scheduler == "cosine_decay":
                scheds.append(
                    optim.lr_scheduler.CosineAnnealingLR(
                        self._updater.optimizer,
                        T_max=self._ppo_cfg.lrsched_T_max,
                        eta_min=self._ppo_cfg.lrsched_eta_min,
                    )
                )
            elif self._ppo_cfg.lr_scheduler == "cosine_annealing_warm_restarts":
                scheds.append(
                    optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self._updater.optimizer,
                        T_0=self._ppo_cfg.lrsched_T_0,
                        eta_min=self._ppo_cfg.lrsched_eta_min,
                    )
                )
            elif not self._ppo_cfg.lr_scheduler:
                pass
            else:
                raise ValueError

            if scheds and len(scheds) > 1:
                self._lr_scheduler = CustomSequentialLR(
                    self._updater.optimizer, scheds, milestones
                )
            elif scheds:
                self._lr_scheduler = scheds[0]
            else:
                self._lr_scheduler = None

        if resume_state is not None:
            self._updater.load_state_dict(resume_state["state_dict"])
            self._updater.load_state_dict(
                {"actor_critic." + k: v for k, v, in resume_state["state_dict"].items()}
            )
        self._policy_action_space = self._env_spec.action_space

    def load_state_dict(self, state: Dict, strict=True) -> None:
        self._actor_critic.load_state_dict(state["state_dict"], strict=strict)
        if self._updater is not None:
            self._updater.load_state_dict(state)
            if "lr_sched_state" in state:
                self._lr_scheduler.load_state_dict(state["lr_sched_state"])

    def _create_storage(
        self,
        num_envs: int,
        env_spec: EnvironmentSpec,
        actor_critic: NetPolicy,
        policy_action_space: spaces.Space,
        config: "DictConfig",
        device,
    ) -> Storage:
        """
        Default behavior for setting up and initializing the rollout storage.
        """

        obs_space = get_rollout_obs_space(
            env_spec.observation_space, actor_critic, config
        )
        ppo_cfg = config.habitat_baselines.rl.ppo
        dtype = (
            torch.float16
            if config.habitat_baselines.rl.ppo.storage_low_precision
            else torch.float32
        )
        separate_rollout_and_policy = (
            config.habitat_baselines.separate_rollout_and_policy
        )
        rollout_on_cpu = config.habitat_baselines.rollout_on_cpu
        rollouts = baseline_registry.get_storage(
            config.habitat_baselines.rollout_storage_name
        )(
            numsteps=ppo_cfg.num_steps,
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=policy_action_space,
            actor_critic=actor_critic,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            device=device,
            separate_rollout_and_policy=separate_rollout_and_policy,
            dtype=dtype,
            freeze_visual_feats=not config.habitat_baselines.rl.ddppo.train_encoder,
            on_cpu=rollout_on_cpu,
            acting_context=ppo_cfg.acting_context,
            is_memory=config.habitat_baselines.rl.policy.main_agent.transformer_config.model_name
            == "transformerxl",
        )
        return rollouts

    def after_update(self):
        # if (
        #     self._lr_scheduler is not None
        # ):
        #     self._lr_scheduler.step()  # type: ignore
        self._updater.after_update()

    def _step_sched(self, steps):
        if self._lr_scheduler:
            self._lr_scheduler.step(steps)
