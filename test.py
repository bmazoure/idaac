import numpy as np
import torch

from procgen import ProcgenEnv

from ppo_daac_idaac.envs import VecPyTorchProcgen, ProcgenVecEnvCustom


def evaluate(args, actor_critic, device):
    actor_critic.eval()

    # Sample Levels From the Full Distribution 
    eval_envs = ProcgenVecEnvCustom(num_envs=1, env_name=args.env_name, \
        num_levels=0, start_level=0, mode=args.distribution_mode, device=device)
    # eval_envs = VecPyTorchProcgen(venv, device)

    eval_episode_rewards = []
    obs = eval_envs.reset()

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            if args.algo == 'ppo':
                _, action, _ = actor_critic.act(obs)
            else:
                _, _, action, _ = actor_critic.act(obs)

        obs, _, done, infos = eval_envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
        .format(len(eval_episode_rewards), \
        np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    return eval_episode_rewards

