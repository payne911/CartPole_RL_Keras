import gym  # https://gym.openai.com/docs/
import numpy as np
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Conv2D, merge
from keras.models import Input, Model


env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
env.reset()

# To obtain a "screenshot" to use as input for the model
screenshot = env.render(mode='rgb_array')  # 600x400
screenshot = screenshot[170:300, :, :]  # vertical crop
def to_grayscale(img): return np.mean(img, axis=2).astype(np.uint8)
def downsample(img): return img[::2, ::2]
def preprocess(img): return to_grayscale(downsample(img))
screenshot = preprocess(screenshot)

# # sanity check (screenshot saved as image)
# from PIL import Image
# img = Image.fromarray(test, 'L')  # grayscale
# img.save('img/my_crop.png')  # sanity check
# print(img.size)

# extracting constants for the model
height, width = screenshot.shape
num_actions = env.action_space.n


def build_model(width, height, num_actions):  # todo: conv model (with rnn)
    frames_input = Input(shape=(width, height, 1), name="frames")

    # As proposed by paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    conv1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), activation='relu')(frames_input)
    conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu')(conv1)

    # from image to action
    flat = Flatten()(conv2)
    dense = Dense(256, activation='relu')(flat)
    output = Dense(num_actions, activation='linear')(dense)

    # https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    actions_input = Input(shape=(num_actions,), name="actions")
    merged_out = merge.multiply([output, actions_input])  # todo: why?
    model = Model(inputs=[actions_input, frames_input], outputs=merged_out)

    #model = Model(inputs=frames_input, outputs=output)

    print(model.summary())
    return model


# setting up the model
model = build_model(width, height, num_actions)
memory = SequentialMemory(limit=100000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)

# train and test
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=['mae'])  # todo: loss='mse'
dqn.fit([env, screenshot], nb_steps=50000,
        visualize=True,
        verbose=2)
dqn.test(env, nb_episodes=10, visualize=True)

model.save('models/saved_img_model.h5')
