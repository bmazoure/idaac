import os
import torch
import numpy as np
from collections import deque
import wandb

import hyperparams as hps
from test import evaluate
from procgen import ProcgenEnv

from baselines import logger
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

from ppo_daac_idaac import algo, utils
from ppo_daac_idaac.arguments import parser
from ppo_daac_idaac.model import PPOnet, IDAACnet, \
    LinearOrderClassifier, NonlinearOrderClassifier, CTRL
from ppo_daac_idaac.storage import DAACRolloutStorage, \
    IDAACRolloutStorage, RolloutStorage
from ppo_daac_idaac.envs import VecPyTorchProcgen


def train(args):
    group_name = "%s_%s_%s" % (args.algo, args.env_name, args.run_id)
    name = "%s_%s_%s_%d" % (args.algo, args.env_name, args.run_id,
                            np.random.randint(100000000))

    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               config=args,
               group=group_name,
               name=name,
               sync_tensorboard=False,
               mode=args.wandb_mode)
    # torch.autograd.set_detect_anomaly(True)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.use_best_hps:
        args.value_epoch = hps.value_epoch[args.env_name]
        args.value_freq = hps.value_freq[args.env_name]
        args.adv_loss_coef = hps.adv_loss_coef[args.env_name]
        args.clf_hidden_size = hps.clf_hidden_size[args.env_name]
        args.order_loss_coef = hps.order_loss_coef[args.env_name]
        if args.env_name in hps.nonlin_envs:
            args.use_nonlinear_clf = True
        else:
            args.use_nonlinear_clf = False
    print("\nArguments: ", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name, \
        num_levels=args.num_levels, start_level=args.start_level, \
        distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)

    obs_shape = envs.observation_space.shape     
    if 'ppo' in args.algo:
        actor_critic = PPOnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})    
    else:           
        actor_critic = IDAACnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})
    if 'ctrl' in args.algo:
        ctrl = CTRL(dims=[256,256,256], cluster_len=args.cluster_len, num_protos=args.num_clusters, k=args.k, temp=args.temp)
        ctrl.to(device)
    else:
        ctrl = None
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)
    
    if 'idaac' in args.algo:
        if args.use_nonlinear_clf:
            order_classifier = NonlinearOrderClassifier(emb_size=args.hidden_size, \
                hidden_size=args.clf_hidden_size).to(device)       
        else:
            order_classifier = LinearOrderClassifier(emb_size=args.hidden_size)
        order_classifier.to(device)
        print("\n Order Classifier: ", order_classifier)

    if 'idaac' in args.algo:
        rollouts = IDAACRolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)
    elif 'daac' in args.algo:
        rollouts = DAACRolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space, args.cluster_len)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    if 'idaac' in args.algo:
        agent = algo.IDAAC(
            actor_critic,
            order_classifier,
            args.clip_param,
            args.ppo_epoch,
            args.value_epoch, 
            args.value_freq,
            args.num_mini_batch,
            args.value_loss_coef,
            args.adv_loss_coef,
            args.order_loss_coef, 
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif 'daac' in args.algo:
        agent = algo.DAAC(
            actor_critic,
            ctrl,
            args.clip_param,
            args.ppo_epoch,
            args.value_epoch,
            args.value_freq,
            args.num_mini_batch,
            args.value_loss_coef,
            args.adv_loss_coef,
            args.entropy_coef,
            myow_k=args.myow_k,
            lr=args.lr,
            lr_ctrl=args.lr_ctrl,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    else: 
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    returns_ood_acc = []
    nsteps = torch.zeros(args.num_processes)
    for j in range(num_updates):
        actor_critic.train()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if 'ppo' in args.algo:
                    value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                else:
                    adv, value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                                        
            obs, reward, done, infos = envs.step(action)

            if 'idaac' in args.algo:
                levels = torch.LongTensor([info['level_seed'] for info in infos])
                if j == 0 and step == 0:
                    rollouts.levels[0].copy_(levels)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1 
            nsteps[done == True] = 0
            if 'idaac' in args.algo:
                rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks, adv, levels, nsteps)
            elif 'daac' in args.algo:
                rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks, adv)
            else:
                rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        if 'idaac' in args.algo:
            rollouts.before_update()
            order_acc, order_loss, clf_loss, adv_loss, value_loss, \
                action_loss, dist_entropy = agent.update(rollouts)    
        elif 'daac' in args.algo:
            adv_loss, value_loss, action_loss, dist_entropy, ctrl_loss = agent.update(rollouts)    
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)    
        rollouts.after_update()

        # Save Model
        if j == num_updates - 1 and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(envs, 'ob_rms', None)
            ], os.path.join(args.save_dir, "agent{}.pt".format(log_file))) 

        # Save Logs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps 
            print("\nUpdate {}, step {}:".format(j, total_num_steps))
            print("Last {} training episodes, mean/median reward {:.2f}/{:.2f}"\
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards)))

            eval_episode_rewards = evaluate(args, actor_critic, device)


            wandb.log({
                "%s/ep_return_200" % (args.env_name):np.mean(episode_rewards)
            },step=total_num_steps)

            wandb.log({
                "%s/ep_return_all" % (args.env_name):np.mean(eval_episode_rewards)
            },step=total_num_steps)

            wandb.log({
                "%s/adv_loss" % (args.env_name):adv_loss,
                "%s/action_loss" % (args.env_name):action_loss,
                "%s/value_loss" % (args.env_name):value_loss,
                "%s/dist_entropy" % (args.env_name):dist_entropy,
                "%s/ctrl_loss" % (args.env_name):ctrl_loss,
            },step=total_num_steps)

            returns_ood_acc.append(np.mean(eval_episode_rewards))

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
