import gym  # https://gym.openai.com/docs/
from gym.utils.play import play


env = gym.make('CartPole-v1')  # https://gym.openai.com/envs/CartPole-v1/
play(env)
