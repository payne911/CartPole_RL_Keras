import gym  # https://gym.openai.com/docs/
from wrappers import FrameStack, WarpFrame
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Input, Model


env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
env = FrameStack(env, 4)
env = WarpFrame(env)
env.reset()

num_actions = env.action_space.n

# # sanity check (screenshot saved as image)
# from PIL import Image
# img = Image.fromarray(test, 'L')  # grayscale
# img.save('img/my_crop.png')  # sanity check
# print(img.size)


def build_model(num_actions, width=84, height=84):
    frames_input = Input(shape=(4, width, height), name="frames")

    # As proposed by paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    conv1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4),
                   data_format='channels_first', activation='relu')(frames_input)
    conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                   data_format='channels_first', activation='relu')(conv1)

    # from image to action
    flat = Flatten()(conv2)
    dense1 = Dense(256, activation='relu')(flat)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(num_actions, activation='linear')(dense2)

    model = Model(inputs=frames_input, outputs=output)
    print(model.summary())

    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss='mse', optimizer=optimizer)
    return model


# setting up the model
model = build_model(num_actions, 65, 200)
memory = SequentialMemory(limit=100000, window_length=4)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)

# train and test
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=400,
               target_model_update=1e-2, policy=policy)
dqn.compile(RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=['mae'])
dqn.fit(env, nb_steps=500000,
        visualize=False,
        verbose=2)
dqn.test(env, nb_episodes=25, visualize=True)

model.save('models/saved_img_model_5.h5')
