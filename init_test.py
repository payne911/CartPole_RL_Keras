import gym  # https://gym.openai.com/docs/
from time import sleep


### To obtain a screenshot to use as input for NN
#from PIL import Image
# screenshot = env.render(mode='rgb_array')  # 600x400 screenshot
# screenshot = screenshot[280:300, 200:400, :]
# img = Image.fromarray(screenshot, 'RGB')
# img.save(str(totalreward) + 'my_crop.png')
# print(screenshot.shape)


env = gym.make('CartPole-v0')  # https://gym.openai.com/envs/CartPole-v1/
env.reset()

# extracting env-specific constants
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

print(type(env.observation_space))  # <class 'gym.spaces.box.Box'>
print(env.observation_space)  # Box(4,)


def run_episode(iter):
    totalreward = 0
    for _ in range(80):
        env.render()  # show the animation (window)
        observation, reward, done, _ = env.step(env.action_space.sample())  # random action
        print(observation, reward, done)  # np[4], float (1.0), bool
        totalreward += reward
        sleep(0.1)  # time in seconds
        if done:  # game is lost
            print("----Failure. Score:", totalreward)
            env.reset()
            break

    # env.env.ale.saveScreenPNG(b'test_image.png')  # screenshot (not working)
    if iter - 1 == 0:  # run a max of "iter" times
        return
    else:
        run_episode(iter - 1)


run_episode(3)
env.close()  # close the animation (window)


###### reports
#
# Observation:
#   Type: Box(4)
#   Num     Observation             Min         Max
#   0       Cart Position           -4.8        4.8
#   1       Cart Velocity           -Inf        Inf
#   2       Pole Angle              -24°        24°
#   3       Pole Velocity At Tip    -Inf       Inf
#
# Action:
#   Type: Discrete(2)
#   Num     Action
#   0       Push cart to the left
#   1       Push cart to the right
