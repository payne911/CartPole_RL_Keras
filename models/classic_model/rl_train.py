import gym  # https://gym.openai.com/docs/
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.models import Input, Model


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


# setting up the model
model = build_model(num_inputs, num_actions)  # 4 input, 2 actions
memory = SequentialMemory(limit=10000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)

# train and test
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=400,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000,
        visualize=False,
        verbose=2)
dqn.test(env, nb_episodes=10, visualize=True)

model.save('saves/saved_model_3.h5')
