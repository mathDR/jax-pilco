import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("InvertedDoublePendulum-v4", render_mode="rgb_array")
observation, info = env.reset()

for _ in range(1000):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    breakpoint()
    observation, reward, terminated, truncated, info = env.step(action)
    img = env.render()
    # plt.imshow(img)
    # plt.show()
    if terminated or truncated:
        observation, info = env.reset()

env.close()
