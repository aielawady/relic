from collections import deque
from itertools import groupby
import json
import random
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from habitat.core.dataset import Episode, EpisodeIterator, T
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from numpy import ndarray


class EpisodeIteratorRepeat(EpisodeIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_episode = next(self._iterator)
        if len(self.episodes) < 10:
            print([x.episode_id for x in self.episodes])

    def after_update(self):
        self._forced_scene_switch_if()

        self._next_episode = next(self._iterator, None)
        if self._next_episode is None:
            if not self.cycle:
                raise StopIteration

            self._iterator = iter(self.episodes)

            if self.shuffle:
                self._shuffle()

            self._next_episode = next(self._iterator)

        if (
            self._prev_scene_id != self._next_episode.scene_id
            and self._prev_scene_id is not None
        ):
            self._rep_count = 0
            self._step_count = 0

        self._prev_scene_id = self._next_episode.scene_id

    def __next__(self) -> Episode:
        return self._next_episode


# ==========================================================================================


class ObjNavEpisodeIterator(Iterator[T]):
    r"""Episode Iterator class that gives options for how a list of episodes
    should be iterated.

    Some of those options are desirable for the internal simulator to get
    higher performance. More context: simulator suffers overhead when switching
    between scenes, therefore episodes of the same scene should be loaded
    consecutively. However, if too many consecutive episodes from same scene
    are feed into RL model, the model will risk to overfit that scene.
    Therefore it's better to load same scene consecutively and switch once a
    number threshold is reached.

    Currently supports the following features:

    Cycling:
        when all episodes are iterated, cycle back to start instead of throwing
        StopIteration.
    Cycling with shuffle:
        when cycling back, shuffle episodes groups grouped by scene.
    Group by scene:
        episodes of same scene will be grouped and loaded consecutively.
    Set max scene repeat:
        set a number threshold on how many episodes from the same scene can be
        loaded consecutively.
    Sample episodes:
        sample the specified number of episodes.
    """

    def __init__(
        self,
        episodes: Sequence[T],
        cycle: bool = True,
        shuffle: bool = False,
        group_by_scene: bool = True,
        max_scene_repeat_episodes: int = -1,
        max_scene_repeat_steps: int = -1,
        num_episode_sample: int = -1,
        step_repetition_range: float = 0.2,
        seed: int = None,
    ) -> None:
        r"""..

        :param episodes: list of episodes.
        :param cycle: if :py:`True`, cycle back to first episodes when
            StopIteration.
        :param shuffle: if :py:`True`, shuffle scene groups when cycle. No
            effect if cycle is set to :py:`False`. Will shuffle grouped scenes
            if :p:`group_by_scene` is :py:`True`.
        :param group_by_scene: if :py:`True`, group episodes from same scene.
        :param max_scene_repeat_episodes: threshold of how many episodes from the same
            scene can be loaded consecutively. :py:`-1` for no limit
        :param max_scene_repeat_steps: threshold of how many steps from the same
            scene can be taken consecutively. :py:`-1` for no limit
        :param num_episode_sample: number of episodes to be sampled. :py:`-1`
            for no sampling.
        :param step_repetition_range: The maximum number of steps within each scene is
            uniformly drawn from
            [1 - step_repeat_range, 1 + step_repeat_range] * max_scene_repeat_steps
            on each scene switch.  This stops all workers from swapping scenes at
            the same time
        """
        assert group_by_scene
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        # sample episodes
        if num_episode_sample >= 0:
            episodes = np.random.choice(  # type: ignore[assignment]
                episodes, num_episode_sample, replace=False  # type: ignore[arg-type]
            )

        if not isinstance(episodes, list):
            episodes = list(episodes)

        print(f"Scenes: {set(x.scene_id for x in episodes)}")

        self.episodes = episodes
        self.cycle = cycle
        self.group_by_scene = group_by_scene
        self.shuffle = shuffle

        if shuffle:
            random.shuffle(self.episodes)

        self.episodes = self._group_scenes(self.episodes)

        self.max_scene_repetition_episodes = max_scene_repeat_episodes
        self.max_scene_repetition_steps = max_scene_repeat_steps

        self._rep_count = -1  # 0 corresponds to first episode already returned
        self._step_count = 0
        self._prev_scene_id: Optional[str] = None
        self._iterator = iter(self.episodes[0])

        self.step_repetition_range = step_repetition_range
        self._set_shuffle_intervals()
        self.should_switch_scene = False
        self.switches_count = 1

    def __iter__(self) -> "EpisodeIterator":
        return self

    def __next__(self) -> Episode:
        r"""The main logic for handling how episodes will be iterated.

        :return: next episode.
        """
        # self._forced_scene_switch_if()

        next_episode = next(self._iterator, None)
        if next_episode is None:
            self.should_switch_scene = True
            if not self.cycle:
                raise StopIteration

            self._iterator = iter(self.episodes[0])

            if self.shuffle:
                self._shuffle()

            next_episode = next(self._iterator)

        if (
            self._prev_scene_id != next_episode.scene_id
            and self._prev_scene_id is not None
        ):
            self._rep_count = 0
            self._step_count = 0

        self._prev_scene_id = next_episode.scene_id
        return next_episode

    def _forced_scene_switch(self) -> None:
        r"""Internal method to switch the scene. Moves remaining episodes
        from current scene to the end and switch to next scene episodes.
        """

        self.episodes.rotate(-1)
        self._iterator = iter(self.episodes[0])

    def _shuffle(self) -> None:
        r"""Internal method that shuffles the remaining episodes.
        If self.group_by_scene is true, then shuffle groups of scenes.
        """
        assert self.shuffle
        # random.shuffle(episodes)
        for e in self.episodes:
            random.shuffle(e)

    def _group_scenes(
        self, episodes: Union[Sequence[Episode], List[Episode], ndarray]
    ) -> List[T]:
        r"""Internal method that groups episodes by scene
        Groups will be ordered by the order the first episode of a given
        scene is in the list of episodes

        So if the episodes list shuffled before calling this method,
        the scenes will be in a random order
        """
        assert self.group_by_scene
        episodes = sorted(episodes, key=lambda x: (x.scene_id, x.object_category))
        print(f"There are {len(set(x.scene_id for x in episodes))} scenes.")
        print(
            f"There are {len(set((x.scene_id, x.object_category) for x in episodes))} scenes and targets."
        )
        groups = deque(
            [
                list(x[1])
                for x in groupby(episodes, lambda x: (x.scene_id, x.object_category))
            ]
        )
        print(f"Found {len(groups)} groups.")
        return groups

    def step_taken(self) -> None:
        self._step_count += 1

    @staticmethod
    def _randomize_value(value: int, value_range: float) -> int:
        return random.randint(
            int(value * (1 - value_range)), int(value * (1 + value_range))
        )

    def _set_shuffle_intervals(self) -> None:
        if self.max_scene_repetition_episodes > 0:
            self._max_rep_episode = self.max_scene_repetition_episodes
        else:
            self._max_rep_episode = None

        if self.max_scene_repetition_steps > 0:
            self._max_rep_step = self._randomize_value(
                self.max_scene_repetition_steps, self.step_repetition_range
            )
        else:
            self._max_rep_step = None

    def _forced_scene_switch_if(self) -> None:
        do_switch = False
        self._rep_count += 1

        # Shuffle if a scene has been selected more than _max_rep_episode times in a row
        if (
            self._max_rep_episode is not None
            and self._rep_count >= self._max_rep_episode
        ):
            do_switch = True

        # Shuffle if a scene has been used for more than _max_rep_step steps in a row
        if self._max_rep_step is not None and self._step_count >= self._max_rep_step:
            do_switch = True

        if do_switch or self.should_switch_scene:
            self._forced_scene_switch()
            self._set_shuffle_intervals()
            self.should_switch_scene = False
            self.switches_count += 1

    def after_update(self):
        self._forced_scene_switch_if()


# =================================================


@registry.register_dataset(name="RearrangeDatasetTransformers-v0")
class RearrangeDatasetTransformersV0(RearrangeDatasetV0):
    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator[T]:
        return EpisodeIteratorRepeat(self.episodes, *args, **kwargs)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, reset_episode_ids=True
    ) -> None:
        deserialized = json.loads(json_str)

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangeEpisode(**episode)
            if reset_episode_ids:
                rearrangement_episode.episode_id = str(i)

            self.episodes.append(rearrangement_episode)


@registry.register_dataset(name="ObjectNavTransformers-v1")
class ObjectNavTransformersV1(ObjectNavDatasetV1):
    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator[T]:
        return ObjNavEpisodeIterator(self.episodes, *args, **kwargs)
