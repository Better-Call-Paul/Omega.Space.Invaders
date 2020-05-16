#necessary contingencies to run atari and AI

import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2

env = gym.make('SpaceInvaders-v0')

def evaluate(model, num_episodes=200):
  """Evaluate a PPO model.

  Args:
    model: (BaseRLModel object) the RL Agent.
    num_episodes: (int) number of time episodes to evaluate the model.
  Returns:
    (float) Mean reward for all the episodes
  """
  print("Start model evaluation.")
  all_rewards = []

  for i in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      episode_reward += reward

    print("Reward for episode {} is {}.".format(i+1, episode_reward))
    all_rewards.append(episode_reward)

  # Compute mean reward for the `num_episodes` episodes
  mean_reward = round(np.mean(all_rewards))
  print("Mean reward:", mean_reward, "Num episodes:", num_episodes)
  return mean_reward

rewards_array = []

#gets the average score of all trials

for i in range(10):
	model = PPO2.load("ppo_space_invaders_model" + str(i))
	average_reward = evaluate(model, num_episodes=1)
	rewards_array.append(average_reward)


print ("this prints the average reward array things")
print (rewards_array)



