import gym
import numpy as np
from keras.models import load_model


# load and set up
model = load_model('models/saved_model.h5')
env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
env.reset()

print(model.summary())

for _ in range(10):
    print("TEST")
    totalreward = 1
    observation, reward, done, _ = env.step(env.action_space.sample())  # random action
    print(observation)
    while not done:
        # todo: use keyboard-arrow input to interact with agent
        env.render()  # show the animation (window)
        input_action = np.expand_dims(np.expand_dims(observation, 0), 0)  # hack for prediction dim
        predictions = model.predict(input_action)
        observation, reward, done, _ = env.step(np.argmax(predictions))
        totalreward += reward
    print("Total reward:", totalreward)
    env.reset()

env.close()  # close the animation (window)
