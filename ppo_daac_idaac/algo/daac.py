import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_distance(A):
    similarity = (A.T @ A)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = torch.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    # inv_square_mag[jnp.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = torch.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag

def cos_loss(p, z):
    p = F.normalize(p, dim=1, p=2)
    z = F.normalize(z, dim=1, p=2)
    dist = 2 - 2 * (p * z.detach()).sum(dim=1)
    return dist

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
                 myow_k,
                 lr=None,
                 lr_ctrl=None,
                 eps=None,
                 max_grad_norm=None):
        self.myow_k = myow_k
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
        if ctrl is not None:
            self.ctrl_parameters = list(ctrl.parameters())+list(actor_critic.parameters())
        
        self.policy_optimizer = optim.Adam(\
            self.policy_parameters, lr=lr, eps=eps)
        self.value_optimizer = optim.Adam(\
            self.value_parameters, lr=lr, eps=eps)
        if ctrl is not None:
            self.ctrl_optimizer = optim.Adam(\
                self.ctrl_parameters, lr=lr_ctrl, eps=eps)

        self.num_policy_updates = 0

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        # CTRL
        if self.ctrl is not None:
            ctrl_loss_epoch = 0
            for e in range(self.ppo_epoch):
                    data_generator = rollouts.feed_forward_generator(
                        advantages, 32)

                    for sample in data_generator:
                        obs_batch, actions_batch, value_preds_batch, return_batch, \
                            old_action_log_probs_batch, adv_targ, adv_preds_batch,  obs_seq, actions_seq, returns_seq = \
                            sample
                        # z: cluster_len x ((n_timesteps-cluster_len) * n_processes) x n_rkhs
                        z_seq = self.actor_critic.base(obs_seq.reshape(-1,*obs_batch.size()[1:]))[1].reshape(*obs_seq.size()[:2], -1)
                        v_clust, w_clust, v_pred, w_pred = self.ctrl(z_seq, actions_seq, returns_seq)
                        # C = self.ctrl.protos.weight.data.clone()
                        # C = F.normalize(C, dim=1, p=2)
                        # self.ctrl.protos.weight.data.copy_(C)

                        # v_clust = F.normalize(v_clust, dim=1, p=2)
                        # w_clust = F.normalize(w_clust, dim=1, p=2)

                        scores_v = self.ctrl.protos(v_clust)
                        log_p = F.log_softmax(scores_v / self.ctrl.temp, dim=1)

                        scores_v_target = self.ctrl.protos(v_clust)
                        scores_w_target = self.ctrl.protos(w_clust)
                        q_target = self.ctrl.sinkhorn(scores_w_target)
                        proto_loss = -(q_target * log_p).sum(axis=1).mean()

                        # MYOW
                        dist = compute_distance(self.ctrl.protos.weight.data.clone().T)
                        vals, indx = torch.topk(-dist, self.myow_k + 1)
                        cluster_idx = torch.argmax(q_target, 1)
                        
                        w_pred_target = w_pred
                        cluster_membership_list = []
                        for c in range(self.ctrl.num_protos):
                            idxes = torch.where(cluster_idx==c)[0]
                            if len(idxes) < self.myow_k:
                                cluster_membership_list.append(torch.randint(0,cluster_idx.shape[0],(self.myow_k,)).to(device))
                            else:
                                cluster_membership_list.append(idxes[:self.myow_k])
                        cluster_membership_list = torch.stack(cluster_membership_list)
                        myow_loss = 0.
                        for k_idx in range(self.myow_k):
                            nearby_cluster_idx = indx[:, k_idx+1][cluster_idx]
                            idx = torch.randint(0,self.myow_k+1,(self.ctrl.num_protos, 1)).to(device)
                            
                            nearby_vec_idx = torch.gather(torch.gather(cluster_membership_list,0,idx),0,nearby_cluster_idx.unsqueeze(1))
                            nearby_vec = w_pred_target[nearby_vec_idx][:,0]
                            myow_loss += cos_loss(v_pred, nearby_vec).mean()

                        ctrl_loss = proto_loss + myow_loss
                        print(proto_loss, myow_loss, ctrl_loss)

                        self.ctrl_optimizer.zero_grad()
                        ctrl_loss.backward()
                        nn.utils.clip_grad_norm_(self.ctrl_parameters, \
                                                self.max_grad_norm)
                        self.ctrl_optimizer.step()  
                        ctrl_loss_epoch += ctrl_loss.item()
            ctrl_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)
        else:
            ctrl_loss_epoch = 0.
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
            action_loss_epoch, dist_entropy_epoch, ctrl_loss_epoch
