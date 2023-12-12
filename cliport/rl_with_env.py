"""Ravens main training script."""
'''
1. Both QLearning and PPO require two networks, eval and target, besides the replay buffer.
   So the memory would be an issue.
2. QLearning requires a Q network and multi-step task.
3. PPO requires the prob. of action.
'''

import os
import pickle
import json
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import normalize

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment

class RolloutBuffer:
    def __init__(self):
        self.act = []
        self.pick = []
        self.place = []
        self.states = []
        self.obs = []
        self.info = []
        self.goal = []
        self.pick_logprob = []
        self.place_logprob = []   
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.act[:]
        del self.pick[:]
        del self.place[:]
        del self.states[:]
        del self.obs[:]
        del self.info[:]
        del self.goal[:]
        del self.pick_logprob[:]
        del self.place_logprob[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, vcfg, name,tcfg, ds, model_file, lr, 
                 gamma, K_epochs, eps_clip, has_continuous_action_space, 
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        
        agent = agents.names[vcfg['agent']](name, tcfg, None, ds)
        # Load checkpoint
        agent.load(model_file)
        print(f"Loaded: {model_file}")
        
        # self.critic = nn.Sequential(
        #                 nn.Linear(6 * 320 * 160, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 1)
        #             )
        # self.critic = self.critic.to('cuda')

        self.agent = agent
        self.agent_old = agent
        self.agent_old.load_state_dict(self.agent.state_dict())
        self.pick_optimizer = torch.optim.Adam([
                        {'params': self.agent.attention.parameters(), 'lr': lr},
                    ])
        self.place_optimizer = torch.optim.Adam([
                        {'params': self.agent.transport.parameters(), 'lr': lr},
                    ])
        # self.critic_optimizer = torch.optim.Adam([
        #                 {'params': self.critic.parameters(), 'lr': lr},
        #             ])
        
        self.MseLoss = nn.MSELoss()
        
    def act(self, obs, info, goal):
        img, pick_conf, place_conf, act = self.agent_old.act_for_rl(obs, info, goal)
        
        pick_dist = Categorical(pick_conf.flatten())
        pick = pick_dist.sample()
        pick_logprob = pick_dist.log_prob(pick)
        
        # import pdb; pdb.set_trace()
        place_conf = torch.max(place_conf, dim=0)[0].flatten()
        place_dist = Categorical(place_conf)
        place = place_dist.sample()
        # import pdb; pdb.set_trace()
        place_logprob = pick_dist.log_prob(place)
        
        img = torch.tensor(img.flatten()).to('cuda')
        state_val = torch.tensor([0.01]) # should be checked if constant is correct or should be learned
        # state_val = self.critic(img)
        img = img.to('cpu')
        
        return img, pick.detach(), pick_logprob.detach(), \
               place, place_logprob.detach(), \
               state_val.detach(), act
    
    def select_action(self, obs, info, goal):
        with torch.no_grad():
            img, pick, pick_logprob, place, place_logprob, state_val, act = self.act(obs, info, goal)
        
        pick = pick.to('cpu')
        pick_logprob = pick_logprob.to('cpu')
        place = place.to('cpu')
        place_logprob = place_logprob.to('cpu')
        state_val = state_val.to('cpu')
        
        # import pdb; pdb.set_trace()
        self.buffer.obs.append(obs)
        self.buffer.info.append(info)
        self.buffer.states.append(img)
        self.buffer.pick.append(pick)
        self.buffer.pick_logprob.append(pick_logprob)
        self.buffer.place.append(place)
        self.buffer.place_logprob.append(place_logprob)
        self.buffer.state_values.append(state_val) # should be checked if constant is correct or should be learned
        self.buffer.act.append(act)
        self.buffer.goal.append(goal)
        
        return act
    
    def evaluate(self, chunk_i, chunk, device='cuda'):
        
        # number of sample to evaluate
        num_of_sample = len(self.buffer.obs)
        old_pick = torch.squeeze(torch.stack(self.buffer.pick, dim=0)).detach().to(device)
        old_place = torch.squeeze(torch.stack(self.buffer.place, dim=0)).detach().to(device)
        list_pick = []
        list_place = []
        
        for i in range(chunk_i, chunk_i + chunk):
            buffer_obs = self.buffer.obs[i]
            buffer_info = self.buffer.info[i]
            buffer_goal = self.buffer.goal[i]
            
            img, pick_conf, place_conf, act = self.agent.act_for_rl(buffer_obs, buffer_info, buffer_goal)
            
            # import pdb; pdb.set_trace()
            list_pick.append(pick_conf.flatten())
            list_place.append(torch.max(place_conf, dim=0)[0].flatten())
            
        pick_dist = Categorical(torch.squeeze(torch.stack(list_pick, dim=0)))
        place_dist = Categorical(torch.squeeze(torch.stack(list_place, dim=0)))
        
        pick_logprobs = pick_dist.log_prob(old_pick[chunk_i:chunk_i+chunk])
        place_logprobs = place_dist.log_prob(old_place[chunk_i:chunk_i+chunk])
        # import pdb; pdb.set_trace()
        
        pick_dist_entropy = pick_dist.entropy()
        place_dist_entropy = place_dist.entropy()
        
        # memory management
        list_pick = []
        list_place = []
        
        return pick_logprobs, pick_dist_entropy, place_logprobs, place_dist_entropy, \
            torch.squeeze(torch.stack(self.buffer.state_values[chunk_i:chunk_i+chunk], dim=0))
    
    def update(self, device='cuda'):
        # Monte Carlo estimate of returns
        loss_list = {'pick_loss': [], 'place_loss': []}
        self.agent.train()
        self.agent_old.train()
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards)
        rewards = normalize(rewards, dim=0)

        # convert list to tensor
        old_pick_logprob = torch.squeeze(torch.stack(self.buffer.pick_logprob, dim=0))
        old_place_logprob = torch.squeeze(torch.stack(self.buffer.place_logprob, dim=0))
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0))

        # calculate advantages
        advantages = rewards - old_state_values
        
        # Optimize policy for K epochs
        print("PPO Optimizing...")
        for _ in range(self.K_epochs):
            print(f"PPO Epoch: {_}")
            # gradient accumulation
            num_of_sample = len(self.buffer.obs)
            chunk = 2
            for chunk_i in range(0, num_of_sample, chunk):
                if chunk_i + chunk > num_of_sample:
                    break
                else:
                    pick_logprob, pick_entropy, place_logprob, place_entropy, state_values = self.evaluate(chunk_i, chunk)
                    
                    # match state_values tensor dimensions with rewards tensor
                    state_values = torch.squeeze(state_values).to(device)
                    
                    # Finding the ratio (pi_theta / pi_theta__old)
                    pick_logprob = pick_logprob.to(device)
                    chunk_old_pick_logprob = old_pick_logprob[chunk_i:chunk_i+chunk].to(device)
                    place_logprob = place_logprob.to(device)
                    chunk_old_place_logprob = old_place_logprob[chunk_i:chunk_i+chunk].to(device)
                    chunk_advantages = advantages[chunk_i:chunk_i+chunk].to(device)
                    chunk_rewards = rewards[chunk_i:chunk_i+chunk].to(device)
                    
                    pick_ratios = torch.exp(pick_logprob - chunk_old_pick_logprob.detach())
                    place_ratios = torch.exp(place_logprob - chunk_old_place_logprob.detach())

                    # Finding Surrogate Loss   
                    pick_surr1 = pick_ratios * chunk_advantages
                    pick_surr2 = torch.clamp(pick_ratios, 1-self.eps_clip, 1+self.eps_clip) * chunk_advantages
                    place_surr1 = place_ratios * chunk_advantages
                    place_surr2 = torch.clamp(place_ratios, 1-self.eps_clip, 1+self.eps_clip) * chunk_advantages

                    # final loss of clipped objective PPO
                    pick_loss = -torch.min(pick_surr1, pick_surr2) - 0.01 * pick_entropy + 0.5 * self.MseLoss(state_values, chunk_rewards) 
                    place_loss = -torch.min(place_surr1, place_surr2) - 0.01 * place_entropy + 0.5 * self.MseLoss(state_values, chunk_rewards) 
                    
                    pick_loss.mean().backward()
                    
                    place_loss.mean().backward()

                    
                    # self.critic_optimizer.zero_grad()
                    # self.MseLoss(state_values, rewards).backward()
                    # self.critic_optimizer.step()
                    
                    print(f"Pick Loss: {pick_loss.mean().item()} | Place Loss: {place_loss.mean().item()}")
                    loss_list['pick_loss'].append(pick_loss.mean().item())
                    loss_list['place_loss'].append(place_loss.mean().item())

            self.pick_optimizer.step()
            self.pick_optimizer.zero_grad()

            self.place_optimizer.step()
            self.place_optimizer.zero_grad()
            # # Evaluating old actions and values
            # pick_logprob, pick_entropy, place_logprob, place_entropy, state_values = self.evaluate()
            
            # # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)
            
            # # Finding the ratio (pi_theta / pi_theta__old)
            # pick_logprob = pick_logprob.to(device)
            # old_pick_logprob = old_pick_logprob.to(device)
            # place_logprob = place_logprob.to(device)
            # old_place_logprob = old_place_logprob.to(device)
            
            # pick_ratios = torch.exp(pick_logprob - old_pick_logprob.detach())
            # place_ratios = torch.exp(place_logprob - old_place_logprob.detach())

            # # Finding Surrogate Loss   
            # pick_surr1 = pick_ratios * advantages
            # pick_surr2 = torch.clamp(pick_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # place_surr1 = place_ratios * advantages
            # place_surr2 = torch.clamp(place_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # # final loss of clipped objective PPO
            # pick_loss = -torch.min(pick_surr1, pick_surr2) - 0.01 * pick_entropy + 0.5 * self.MseLoss(state_values, rewards) 
            # place_loss = -torch.min(place_surr1, place_surr2) - 0.01 * place_entropy + 0.5 * self.MseLoss(state_values, rewards) 
            
            # self.pick_optimizer.zero_grad()
            # pick_loss.mean().backward()
            # self.pick_optimizer.step()
            
            # self.place_optimizer.zero_grad()
            # place_loss.mean().backward()
            # self.place_optimizer.step()
            
            # # self.critic_optimizer.zero_grad()
            # # self.MseLoss(state_values, rewards).backward()
            # # self.critic_optimizer.step()
            
            # print(f"Pick Loss: {pick_loss.mean().item()} | Place Loss: {place_loss.mean().item()}")
            # loss_list['pick_loss'].append(pick_loss.mean().item())
            # loss_list['place_loss'].append(place_loss.mean().item())
            
            
        # Copy new weights into old policy
        self.agent_old.load_state_dict(self.agent.state_dict())

        # clear buffer
        self.buffer.clear()        
        
        print(f"Mean Pick Loss: {np.mean(loss_list['pick_loss'])} | Mean Place Loss: {np.mean(loss_list['place_loss'])}")


@hydra.main(config_path='./cfg', config_name='PPO')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    PPO_task = vcfg['PPO_task']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=PPO_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{PPO_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    all_results = {}
    name = '{}-{}-n{}'.format(PPO_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Training loop
    print(f"Training with PPO: {str(ckpts_to_eval)}")
    
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        
        # Initialize PPO
        ppo_agent = PPO(vcfg, name, tcfg, ds, model_file, lr=3e-4, gamma=0.99, 
                        K_epochs=40, eps_clip=0.2, has_continuous_action_space=False)
        
        for train_run in range(vcfg['outer_epoch']): # how many training epochs to run
            
            print(f'Training: {train_run + 1}')
            utils.set_seed(train_run, torch=True)
            n_demos = vcfg['n_demos']

            # Run testing and save total rewards with last transition info.
            for i in range(0, n_demos): #how many demos to run
                print(f'Demos: {i + 1}/{n_demos}')
                update_timestep = n_demos // 50 # buffer size and update frequency
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)

                # set task
                if 'multi' in dataset_type:
                    task_name = ds.get_curr_task()
                    task = tasks.names[task_name]()
                    print(f'Evaluating on {task_name}')
                else:
                    task_name = vcfg['PPO_task']
                    task = tasks.names[task_name]()

                task.mode = mode
                env.seed(seed)
                env.set_task(task)
                obs = env.reset()
                info = env.info
                reward = 0

                for _ in range(task.max_steps):
                    # import pdb; pdb.set_trace()
                    act = ppo_agent.select_action(obs, info, goal)
                    obs, reward, done, info = env.step(act)
                    
                    ppo_agent.buffer.rewards.append(reward)
                    ppo_agent.buffer.is_terminals.append(done)
                    total_reward += reward
                    lang_goal = info['lang_goal']
                    
                    print(f'Lang Goal: {lang_goal}')
                    print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')

                    if done:
                        break
                
                # update PPO agent
                if (i + 1) % update_timestep == 0:
                    ppo_agent.update()
                    ckpt_path = os.path.join(vcfg['model_path'], f"PPO/PPO_steps={train_run}.ckpt")
                    torch.save({"state_dict": ppo_agent.agent.state_dict()}, ckpt_path)


            all_results[ckpt] = {
                'episodes': results,
                'mean_reward': mean_reward,
            }


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


if __name__ == '__main__':
    main()
