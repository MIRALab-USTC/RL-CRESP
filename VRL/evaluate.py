import torch
import numpy as np

import utils

REF_MIN_SCORE = 0
REF_MAX_SCORE = 1000

d4rl_score = lambda rew_mean: (rew_mean - REF_MIN_SCORE) / (REF_MAX_SCORE - REF_MIN_SCORE) * 100


def d4rl_eval_fn(video, action_repeat):

    def d4rl_eval(env, env_name, agent, global_episode, global_step, global_frame, record_video, num_eval_episodes=100):
        episode = 0
        episode_returns = []
        episode_lengths = []
        eval_until_episode = utils.Until(num_eval_episodes)
        # evaluate the agent under num_eval_episodes episodes

        while eval_until_episode(episode):
            time_step, episode_len, total_reward = env.reset(), 0, 0
            video.init(env, enabled=(episode == 0) and record_video)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(time_step.observation,
                                       eval_mode=True)
                time_step = env.step(action)
                video.record(env)
                total_reward += time_step.reward
                episode_len += 1

            episode += 1
            video.save(f'{env_name}{global_frame}.mp4')
            episode_returns.append(total_reward)
            episode_lengths.append(episode_len)

        rew_mean = np.mean(episode_returns)
        rew_std  = np.std(episode_returns)
        len_mean = np.mean(episode_lengths)
        score = d4rl_score(rew_mean)

        eval_info = {f'{env_name}episode_reward': rew_mean,
                     f'{env_name}episode_std': rew_std,
                     f'{env_name}episode_length': len_mean * action_repeat,
                     f'{env_name}episode_score': score,
                     'episode': global_episode,
                     'step': global_step}
        return eval_info
    
    return d4rl_eval
