import gym
import numpy as np
from keras.models import load_model
from models.img_model.wrappers import WarpFrame


# load and set up
model = load_model('saves/saved_img_model_2.h5')
print(model.summary())
env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
env = WarpFrame(env)
env.reset()

for attempt in range(10):  # run 10 tests
    total_reward = 0
    obs, reward, done, _ = env.step(env.action_space.sample())  # random action
    obs = np.array([obs, obs, obs, obs])  # 4 stacked frames for initialization

    while not done:
        # todo: use keyboard-arrow input to interact with agent
        env.render()  # show the animation (window)
        total_reward += reward
        input_obs = np.expand_dims(obs, 0)  # hack for prediction dim
        predictions = model.predict(input_obs)
        # Queue-like behavior on numpy array of 4-stacked frames
        new_frame, reward, done, _ = env.step(np.argmax(predictions))
        obs[:-1] = obs[1:]
        obs[-1] = new_frame

    print("Attempt", attempt, "  |  Total reward:", total_reward)
    env.reset()

env.close()  # close the animation (window)
