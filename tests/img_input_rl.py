import gym  # https://gym.openai.com/docs/
from gym import spaces
import cv2
import numpy as np
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Input, Model


env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
num_actions = env.action_space.n


# https://github.com/openai/gym/tree/master/gym/wrappers
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# class WarpFrame(gym.ObservationWrapper):
#     def __init__(self, env, width=84, height=84, grayscale=True):
#         """Warp frames to 84x84 as done in the Nature paper and later work."""
#         gym.ObservationWrapper.__init__(self, env)
#         self.width = width
#         self.height = height
#         self.grayscale = grayscale
#         if self.grayscale:
#             self.observation_space = spaces.Box(low=0, high=255,
#                 shape=(self.height, self.width, 1), dtype=np.uint8)
#         else:
#             self.observation_space = spaces.Box(low=0, high=255,
#                 shape=(self.height, self.width, 3), dtype=np.uint8)
#
#     def observation(self, frame):
#         if self.grayscale:
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
#         if self.grayscale:
#             frame = np.expand_dims(frame, -1)
#         return frame

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
        return self.preprocess(self.env.render(mode='rgb_array')[170:300, :, :])


env = WarpFrame(env)
env.reset()

# # sanity check (screenshot saved as image)
# from PIL import Image
# img = Image.fromarray(test, 'L')  # grayscale
# img.save('img/my_crop.png')  # sanity check
# print(img.size)


def build_model(num_actions, width=84, height=84):
    frames_input = Input(shape=(1, width, height), name="frames")

    # As proposed by paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    conv1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4),
                   data_format='channels_first', activation='relu')(frames_input)
    conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                   data_format='channels_first', activation='relu')(conv1)

    # from image to action
    flat = Flatten()(conv2)
    dense = Dense(256, activation='relu')(flat)
    output = Dense(num_actions, activation='linear')(dense)

    model = Model(inputs=frames_input, outputs=output)
    print(model.summary())
    return model


# setting up the model
model = build_model(num_actions, 65, 300)
memory = SequentialMemory(limit=100000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)

# train and test
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=['mae'])  # todo: loss='mse'
dqn.fit(env, nb_steps=50000,
        visualize=False,
        verbose=2)
dqn.test(env, nb_episodes=10, visualize=True)

model.save('models/saved_img_model.h5')
