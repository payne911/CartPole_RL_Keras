import gym
import numpy as np
from keras.models import load_model


# load and set up
model = load_model('saves/saved_model_2.h5')
print(model.summary())
env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
env.reset()
print("\n\n\n\n\n\n\n\n\n")

for attempt in range(10):  # run 10 tests
    total_reward = 0
    observation, reward, done, _ = env.step(env.action_space.sample())  # random action
    while not done:
        # todo: use keyboard-arrow input to interact with agent
        env.render()  # show the animation (window)
        total_reward += reward
        input_obs = np.expand_dims(np.expand_dims(observation, 0), 0)  # hack for prediction dim
        predictions = model.predict(input_obs)
        observation, reward, done, _ = env.step(np.argmax(predictions))
    print("Attempt", attempt, "  |  Total reward:", total_reward)
    env.reset()

env.close()  # close the animation (window)
