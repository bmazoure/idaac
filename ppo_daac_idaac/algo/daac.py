import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DAAC():
    """
    DAAC
    """
    def __init__(self,
                 actor_critic,
                 ctrl,
                 clip_param,
                 ppo_epoch,
                 value_epoch, 
                 value_freq, 
                 num_mini_batch,
                 value_loss_coef,
                 adv_loss_coef,
                 entropy_coef,
                 lr=None,
                 lr_ctrl=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic
        self.ctrl = ctrl

        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch
        self.value_epoch = value_epoch 
        self.value_freq = value_freq
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.adv_loss_coef = adv_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.policy_parameters = list(actor_critic.base.parameters()) + \
            list(actor_critic.dist.parameters())
        self.value_parameters = list(actor_critic.value_net.parameters())
        self.ctrl_parameters = list(ctrl.parameters())+list(actor_critic.parameters())
        
        self.policy_optimizer = optim.Adam(\
            self.policy_parameters, lr=lr, eps=eps)
        self.value_optimizer = optim.Adam(\
            self.value_parameters, lr=lr, eps=eps)
        self.ctrl_optimizer = optim.Adam(\
            self.ctrl_parameters, lr=lr_ctrl, eps=eps)

        self.num_policy_updates = 0

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        # CTRL
        for e in range(self.ppo_epoch):
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

                for sample in data_generator:
                    obs_batch, actions_batch, value_preds_batch, return_batch, \
                        old_action_log_probs_batch, adv_targ, adv_preds_batch,  obs_seq, actions_seq, returns_seq = \
                        sample
                    # z: cluster_len x ((n_timesteps-cluster_len) * n_processes) x n_rkhs
                    z_seq = self.actor_critic.base(obs_seq.reshape(-1,*obs_batch.size()[1:]))[1].reshape(*obs_seq.size()[:2], -1)
                    v_clust, w_clust, v_pred, w_pred = self.ctrl(z_seq, actions_seq, returns_seq)
                    C = self.ctrl.protos.weight.data.clone()
                    C = F.normalize(C, dim=1, p=2)
                    self.ctrl.protos.weight.data.copy_(C)

                    v_clust = F.normalize(v_clust, dim=1, p=2)
                    w_clust = F.normalize(v_clust, dim=1, p=2)

                    scores_v = self.ctrl.protos(v_clust)
                    log_p = F.log_softmax(scores_v / self.ctrl.temp, dim=1)

                    scores_v_target = self.ctrl.protos(v_clust)
                    scores_w_target = self.ctrl.protos(w_clust)
                    with torch.no_grad():
                        q_target = self.ctrl.sinkhorn(scores_w_target)
                    proto_loss = -(q_target * log_p).sum(axis=1).mean()

                    ctrl_loss = proto_loss # + myow # todo

                    self.ctrl_optimizer.zero_grad()
                    ctrl_loss.backward()
                    nn.utils.clip_grad_norm_(self.ctrl_parameters, \
                                             self.max_grad_norm)
                    self.ctrl_optimizer.step()  

        # Update the Policy Network
        adv_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ, adv_preds_batch, obs_seq, actions_seq, returns_seq = \
                    sample

                _, adv, _, action_log_probs, dist_entropy = \
                    self.actor_critic.evaluate_actions(obs_batch, actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                adv_loss = (adv - adv_targ).pow(2).mean()
                
                # Update actor-critic using both PPO Loss
                self.policy_optimizer.zero_grad()
                (adv_loss * self.adv_loss_coef + 
                    action_loss - 
                    dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.policy_parameters, \
                                         self.max_grad_norm)
                self.policy_optimizer.step()
                                
                adv_loss_epoch += adv_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        num_policy_updates = self.ppo_epoch * self.num_mini_batch

        adv_loss_epoch /= num_policy_updates
        action_loss_epoch /= num_policy_updates
        dist_entropy_epoch /= num_policy_updates

        # Update the Value Netowrk
        if self.num_policy_updates % self.value_freq == 0:
            value_loss_epoch = 0
            for e in range(self.value_epoch):
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

                for sample in data_generator:
                    obs_batch, actions_batch, value_preds_batch, return_batch, \
                        old_action_log_probs_batch, adv_targ, adv_preds_batch, obs_seq, actions_seq, returns_seq = \
                        sample
                    
                    _, _, values, _, _ = self.actor_critic.evaluate_actions(
                        obs_batch, actions_batch)                            

                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, \
                                                           self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()

                    # Update actor-critic using both PPO Loss
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_parameters, \
                                             self.max_grad_norm)
                    self.value_optimizer.step()  
                                    
                    value_loss_epoch += value_loss.item()

            num_value_updates = self.value_epoch * self.num_mini_batch
            value_loss_epoch /= num_value_updates
            self.prev_value_loss_epoch = value_loss_epoch 
            
        else:
            value_loss_epoch = self.prev_value_loss_epoch 

        self.num_policy_updates += 1

        return adv_loss_epoch, value_loss_epoch, \
            action_loss_epoch, dist_entropy_epoch
