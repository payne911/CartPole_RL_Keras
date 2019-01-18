import gym
from gym import spaces
from collections import deque
import numpy as np


# https://github.com/openai/gym/tree/master/gym/wrappers
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        gym.ObservationWrapper.__init__(self, env)
        self.env = env
        self.width = width
        self.height = height
        def to_grayscale(img): return np.mean(img, axis=2).astype(np.uint8)
        def downsample(img): return img[::2, ::2]
        self.preprocess = lambda img: to_grayscale(downsample(img))

        self.observation_space = spaces.Box(low=0, high=255,
                        shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        return self.preprocess(self.env.render(mode='rgb_array')[170:300, 100:500, :])


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
