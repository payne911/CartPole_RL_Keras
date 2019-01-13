import gym  # https://gym.openai.com/docs/
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.models import Input, Model


### To obtain a screenshot to use as input for NN
#from PIL import Image
# screenshot = env.render(mode='rgb_array')  # 600x400 screenshot
# screenshot = screenshot[280:300, 200:400, :]
# img = Image.fromarray(screenshot, 'RGB')
# img.save(str(totalreward) + 'my_crop.png')
# print(screenshot.shape)


env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
env.reset()

# extracting env-specific constants
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n


def build_model(state_size, num_actions):
    input = Input(shape=(1, state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model


model = build_model(num_inputs, num_actions)  # 4 input, 2 actions
memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)

dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000,
        visualize=False,
        verbose=1)

dqn.test(env, nb_episodes=10, visualize=True)
