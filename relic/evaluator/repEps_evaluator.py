from functools import partial
import os
from collections import defaultdict

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
import pandas as pd

from habitat import VectorEnv
from torch import Tensor
from numpy import ndarray
from typing import Any, Dict, List, Tuple, Union
import pickle as pkl


class TransformersRepEpsHabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
        n_steps=8200,
        n_demos=-1,
        max_n_eps=-1,
        suffix="",
        is_memory=False,
    ):
        VIS_KEYS = set(["head_rgb"])
        n_envs_visits = np.zeros(envs.num_envs)
        observations = envs.reset()
        next_episodes_info = envs.current_episodes()
        _last_save_value = 0

        selected_eps = set()
        n_envs = envs.num_envs

        for i in range(n_envs):
            _curr_eps = (
                next_episodes_info[i].scene_id,
                next_episodes_info[i].episode_id,
            )
            while _curr_eps in selected_eps:
                envs.call_at(i, "after_update")
                observations[i] = envs.reset_at(i)[0]
                n_envs_visits[i] += 1
                _curr_eps = (
                    envs.current_episodes()[i].scene_id,
                    envs.current_episodes()[i].episode_id,
                )
            selected_eps.add(_curr_eps)

        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent._policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        if is_memory:
            mem = torch.zeros(
                (
                    agent.actor_critic.num_recurrent_layers,
                    agent.actor_critic.memory_size,
                    envs.num_envs,
                    agent.actor_critic.recurrent_hidden_size,
                ),
                device=device,
                dtype=torch.float16,
            )
            should_update_recurrent_hidden_states = False
        else:
            test_recurrent_hidden_states = torch.zeros(
                (
                    agent.actor_critic.num_recurrent_layers,
                    2,
                    envs.num_envs,
                    agent.actor_critic.num_heads,
                    n_steps + 1,
                    agent.actor_critic.recurrent_hidden_size
                    // agent.actor_critic.num_heads,
                ),
                device=device,
                dtype=torch.float16,
            )
            should_update_recurrent_hidden_states = (
                np.prod(test_recurrent_hidden_states.shape) != 0
            )

        not_done_masks = torch.zeros(
            envs.num_envs,
            n_steps + 1,
            1,
            device=device,
            dtype=torch.bool,
        )

        prev_actions = torch.zeros(
            envs.num_envs,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [observations_to_image({k: v[env_idx] for k, v in batch.items()}, {})]
                for env_idx in range(envs.num_envs)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = len(
                set(sum(envs.call(["episodes"] * envs.num_envs), []))
            )  # sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()

        stats = defaultdict(partial(defaultdict, list))

        loop_counter = 0
        while envs.num_envs:
            loop_counter += 1
            for _i in tqdm.tqdm(range(n_steps)):
                if (
                    max_n_eps > -1
                    and (
                        (~not_done_masks[:, : _i + 1, 0])
                        .sum(axis=1)
                        .detach()
                        .cpu()
                        .numpy()
                        > max_n_eps
                    ).all()
                ):
                    break
                current_episodes_info = envs.current_episodes()
                if "one_hot_target_sensor" in batch:
                    batch["one_hot_target_sensor"] = batch[
                        "one_hot_target_sensor"
                    ].half()
                if "one_hot_receptacle_sensor" in batch:
                    batch["one_hot_receptacle_sensor"] = batch[
                        "one_hot_receptacle_sensor"
                    ].half()
                for k in batch:
                    if torch.is_tensor(batch[k]) and torch.is_floating_point(batch[k]):
                        batch[k] = batch[k].half()

                with inference_mode():
                    action_data, outputs = agent.actor_critic.act(
                        batch,
                        mem
                        if is_memory
                        else test_recurrent_hidden_states[..., 1 : _i + 1, :],
                        prev_actions,
                        not_done_masks[:, : _i + 1],
                        deterministic=False,
                        output_attentions=False,
                    )
                    if n_demos > 0:
                        best_actions = envs.call(["best_action"] * envs.num_envs)
                        for i in np.arange(envs.num_envs)[
                            (~not_done_masks[:, : _i + 1, 0])
                            .sum(axis=1)
                            .detach()
                            .cpu()
                            .numpy()
                            <= n_demos
                        ]:
                            # breakpoint()
                            action_data.actions[i] = best_actions[i]
                    outputs.past_key_values = None
                    if action_data.should_inserts is None:
                        if is_memory:
                            mem = action_data.rnn_hidden_states
                        else:
                            test_recurrent_hidden_states[
                                ..., _i + 1, :
                            ] = action_data.rnn_hidden_states
                        prev_actions.copy_(action_data.actions)  # type: ignore
                    else:
                        for i, should_insert in enumerate(action_data.should_inserts):
                            assert False
                            if not should_insert.item():
                                continue
                            if should_update_recurrent_hidden_states:
                                if is_memory:
                                    mem[i] = action_data.rnn_hidden_states[i]
                                else:
                                    test_recurrent_hidden_states[
                                        :, :, i, ..., _i + 1, :
                                    ] = action_data.rnn_hidden_states[i]

                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
                # NB: Move actions to CPU.  If CUDA tensors are
                # sent in to env.step(), that will create CUDA contexts
                # in the subprocesses.
                if is_continuous_action_space(env_spec.action_space):
                    # Clipping actions to the specified limits
                    step_data = [
                        np.clip(
                            a.numpy(),
                            env_spec.action_space.low,
                            env_spec.action_space.high,
                        )
                        for a in action_data.env_actions.cpu()
                    ]
                else:
                    step_data = [a.item() for a in action_data.env_actions.cpu()]

                outputs = envs.step(step_data)

                observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
                # Note that `policy_infos` represents the information about the
                # action BEFORE `observations` (the action used to transition to
                # `observations`).
                policy_infos = agent.actor_critic.get_extra(action_data, infos, dones)
                for i in range(len(policy_infos)):
                    infos[i].update(policy_infos[i])

                observations = envs.post_step(observations)
                batch = batch_obs(  # type: ignore
                    observations,
                    device=device,
                )
                batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

                not_done_masks[:, _i + 1] = torch.tensor(
                    [[not done] for done in dones],
                    dtype=torch.bool,
                    device="cpu",
                )

                rewards = torch.tensor(
                    rewards_l, dtype=torch.float, device="cpu"
                ).unsqueeze(1)
                current_episode_reward += rewards
                next_episodes_info = envs.current_episodes()
                n_envs = envs.num_envs

                for info_i, info in enumerate(infos):
                    info = pd.json_normalize(info).iloc[0].to_dict()
                    key = "_".join(
                        [
                            current_episodes_info[info_i].scene_id,
                            current_episodes_info[info_i].episode_id,
                            str(
                                ep_eval_count.get(
                                    (
                                        current_episodes_info[info_i].scene_id,
                                        current_episodes_info[info_i].episode_id,
                                    ),
                                    0,
                                )
                            ),
                        ]
                    )
                    if dones[info_i]:
                        for k, v in info.items():
                            stats[key][k].append(v)

                        stats[key]["reward_m"].append(rewards_l[info_i])
                        stats[key]["return_m"].append(
                            current_episode_reward[info_i].item()
                        )
                        current_episode_reward[info_i] = 0
                        stats[key]["done"].append(dones[info_i])
                        stats[key]["obs"].append(
                            {
                                k: v[info_i].detach().cpu().numpy().tolist()
                                for k, v in batch.items()
                                if k not in VIS_KEYS
                            }
                        )

                for i in range(n_envs):
                    # Exclude the keys from `_rank0_keys` from displaying in the video
                    disp_info = {
                        k: v for k, v in infos[i].items() if k not in rank0_keys
                    }

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # TODO move normalization / channel changing out of the policy and undo it here
                        frame = observations_to_image(
                            {k: v[i] for k, v in batch.items()}, disp_info
                        )
                        if not not_done_masks[i, _i + 1].item():
                            # The last frame corresponds to the first frame of the next episode
                            # but the info is correct. So we use a black frame
                            final_frame = observations_to_image(
                                {k: v[i] * 0.0 for k, v in batch.items()},
                                disp_info,
                            )
                            final_frame = overlay_frame(final_frame, disp_info)
                            rgb_frames[i].append(final_frame)
                            # The starting frame of the next episode will be the final element..
                            rgb_frames[i].append(frame)
                        else:
                            frame = overlay_frame(frame, disp_info)
                            rgb_frames[i].append(frame)

            if is_memory:
                mem = torch.zeros(
                    (
                        agent.actor_critic.num_recurrent_layers,
                        agent.actor_critic.memory_size,
                        envs.num_envs,
                        agent.actor_critic.recurrent_hidden_size,
                    ),
                    device=device,
                    dtype=torch.float16,
                )

            if len(stats) > 100:
                df = pd.DataFrame.from_dict(stats, "index")
                os.makedirs(
                    os.path.join(config.habitat_baselines.eval_data_dir), exist_ok=True
                )
                df.to_csv(
                    os.path.join(
                        config.habitat_baselines.eval_data_dir,
                        f"latest_{loop_counter}_{suffix}.csv",
                    )
                )
                with open(
                    os.path.join(
                        config.habitat_baselines.eval_data_dir,
                        f"latest_{loop_counter}_{suffix}.pkl",
                    ),
                    "wb",
                ) as f:
                    pkl.dump(stats, f)
                _last_save_value += 1  # len(stats) // 200
                # stats = defaultdict(lambda: defaultdict(list))
                stats = defaultdict(partial(defaultdict, list))

            # episode ended
            pbar.update()

            for i in range(n_envs):
                episode_stats = {"reward": current_episode_reward[i].item()}
                episode_stats.update(extract_scalars_from_info(infos[i]))
                current_episode_reward[i] = 0
                k = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                )
                ep_eval_count[k] += 1
                # use scene_id + episode_id as unique id for storing stats
                stats_episodes[(k, ep_eval_count[k])] = episode_stats

                if len(config.habitat_baselines.eval.video_option) > 0:
                    generate_video(
                        video_option=config.habitat_baselines.eval.video_option,
                        video_dir=config.habitat_baselines.video_dir,
                        # Since the final frame is the start frame of the next episode.
                        images=rgb_frames[i][:-1],
                        episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                        checkpoint_idx=checkpoint_index,
                        metrics=extract_scalars_from_info(disp_info),
                        fps=config.habitat_baselines.video_fps,
                        tb_writer=writer,
                        keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                    )

                    # Since the starting frame of the next episode is the final frame.
                    rgb_frames[i] = rgb_frames[i][-1:]

                gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                if gfx_str != "":
                    write_gfx_replay(
                        gfx_str,
                        config.habitat.task,
                        current_episodes_info[i].episode_id,
                    )

            envs.call(["after_update"] * envs.num_envs)
            observations = envs.reset()
            n_envs_visits += 1
            next_episodes_info = envs.current_episodes()

            selected_eps = set()

            envs_to_pause = []
            for i in range(n_envs):
                if (
                    len(selected_eps) + sum(ep_eval_count.values())
                    >= number_of_eval_episodes * evals_per_ep
                ):
                    envs_to_pause.append(i)
                    continue

                _curr_eps = (
                    next_episodes_info[i].scene_id,
                    next_episodes_info[i].episode_id,
                )
                while (
                    ep_eval_count[_curr_eps] + int(_curr_eps in selected_eps)
                    >= evals_per_ep
                    and n_envs_visits[i] < envs.number_of_episodes[i] * evals_per_ep
                ):
                    envs.call_at(i, "after_update")
                    observations[i] = envs.reset_at(i)[0]
                    n_envs_visits[i] += 1
                    _curr_eps = (
                        envs.current_episodes()[i].scene_id,
                        envs.current_episodes()[i].episode_id,
                    )
                if n_envs_visits[i] >= envs.number_of_episodes[i] * evals_per_ep:
                    envs_to_pause.append(i)
                    continue

                selected_eps.add(_curr_eps)

            n_envs_visits = n_envs_visits[
                [x not in envs_to_pause for x in range(len(n_envs_visits))]
            ]

            observations = envs.post_step(observations)
            batch = batch_obs(observations, device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                None if is_memory else test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        df = pd.DataFrame.from_dict(stats, "index")
        os.makedirs(os.path.join(config.habitat_baselines.eval_data_dir), exist_ok=True)
        df.to_csv(
            os.path.join(
                config.habitat_baselines.eval_data_dir,
                f"latest_{loop_counter + 1}_{suffix}.csv",
            )
        )
        with open(
            os.path.join(
                config.habitat_baselines.eval_data_dir,
                f"latest_{loop_counter + 1}_{suffix}.pkl",
            ),
            "wb",
        ) as f:
            pkl.dump(stats, f)

        """
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."
        """
        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)


def pause_envs(
    envs_to_pause: List[int],
    envs: VectorEnv,
    test_recurrent_hidden_states: Tensor,
    not_done_masks: Tensor,
    current_episode_reward: Tensor,
    prev_actions: Tensor,
    batch: Dict[str, Tensor],
    rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
) -> Tuple[
    VectorEnv,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[str, Tensor],
    List[List[Any]],
]:
    # pausing self.envs with no new episode
    if len(envs_to_pause) > 0:
        state_index = list(range(envs.num_envs))
        for idx in reversed(envs_to_pause):
            state_index.pop(idx)
            envs.pause_at(idx)

        # indexing along the batch dimensions
        if test_recurrent_hidden_states is not None:
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, :, state_index
            ]
        not_done_masks = not_done_masks[state_index]
        current_episode_reward = current_episode_reward[state_index]
        prev_actions = prev_actions[state_index]

        for k, v in batch.items():
            batch[k] = v[state_index]

        if rgb_frames is not None:
            rgb_frames = [rgb_frames[i] for i in state_index]
        # actor_critic.do_pause(state_index)

    return (
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    )
