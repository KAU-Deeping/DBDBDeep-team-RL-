import gym

env = gym.make('Breakout-v0')

env.reset()
for _ in range(100000):
    env.render()
    done = False
    life = 5
    while life != 0:
        while not done:
            _, _, done, _  = env.step(env.action_space.sample())
        life -= 1