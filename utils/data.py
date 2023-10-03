import torch
import numpy as np
import torch.distributions as td
from copy import deepcopy


def preprocess_sac_batch_oto(offline_buffer, model_buffer, online_buffer, batch_size, real_ratio, online_ratio):
    # TODO: fix the below to be dependent on batch_size kwarg
    offline_bs = int(batch_size * real_ratio)
    model_bs = int(batch_size * (1 - real_ratio))
    online_bs = int(batch_size * online_ratio)

    offline_batch = offline_buffer.sample(offline_bs, rl=True)
    model_batch = model_buffer.sample(model_bs, rl=True)
    online_batch = online_buffer.sample(online_bs, rl=True)

    batch = [
        torch.cat((offline_item, model_item, online_item), dim=0)
        for offline_item, model_item, online_item in zip(offline_batch, model_batch, online_batch)
    ]

    return batch


def preprocess_sac_batch(env_replay_buffer, model_replay_buffer, batch_size, real_ratio):
    """"""
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_replay_buffer.sample(env_batch_size, rl=True)
    model_batch = model_replay_buffer.sample(model_batch_size, rl=True)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
             zip(env_batch, model_batch)]
    return batch


def preprocess_sac_batch_latent(env_replay_buffer, model_replay_buffer, batch_size, real_ratio, scaler, state_encoder):
    """"""
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_replay_buffer.sample(env_batch_size, rl=True)
    model_batch = model_replay_buffer.sample(model_batch_size, rl=True)

    # States in the model-replay buffer are already in the transformed latent space. However,
    # we need to perform the transformation to the real states in the env_replay_buffer
    s, a, ns, r, nd = env_batch
    s = scaler.transform(torch.cat([s, torch.ones_like(a)], dim=-1))[:, :s.shape[-1]]
    ns = scaler.transform(torch.cat([ns, torch.ones_like(a)], dim=-1))[:, :ns.shape[-1]]

    with torch.no_grad():
        s = state_encoder(s)
        ns = state_encoder(ns)

    env_batch = (s, a, ns, r, nd)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
             zip(env_batch, model_batch)]
    return batch


def preprocess_sac_batch_penalty(
        env_replay_buffer,
        model_replay_buffer,
        batch_size,
        real_ratio,
        penalty_fn):
    """"""
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_replay_buffer.sample(env_batch_size)
    model_batch = model_replay_buffer.sample(model_batch_size)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
             zip(env_batch, model_batch)]
    return batch


def find_neighbors(replay, train_inputs, dynamics_ens, scaler, n_sample, percentile):
    # TODO: also try weighting each components' standard deviation in the replay buffer. Then scale each
    # entry by i / std_i
    #     sa = torch.cat([s, a], dim=-1).to('cuda') #.unsqueeze(0)

    batch = replay.sample(n_sample)
    obs, action, next_obs, reward, not_done = batch

    #     sa_replay = torch.cat([obs, action], dim=-1)

    #     scaling_stds = sa_replay.std(0)

    sample_sa, sample_spr = dynamics_ens.preprocess_training_batch(batch)
    sample_sa = scaler.transform(sample_sa)
    #     scaling_stds = sample_sa.std(0)
    #     sample_sa = sample_sa / scaling_stds

    # Need to repeat this along dim 1! One for each row in sa
    # entry for sa[i, :] is diff[:, i, :]
    diff = train_inputs - sample_sa.unsqueeze(1).repeat(1, train_inputs.shape[0], 1)
    #     diff = diff / scaling_stds

    # Now for the fully-vectorized version!
    l2_norm_vec = torch.linalg.norm(diff, dim=2).T
    bottom_10_p_vec = np.percentile(l2_norm_vec.cpu().numpy(), percentile, axis=-1)
    percentile_mask_vec = (l2_norm_vec <= torch.from_numpy(bottom_10_p_vec.reshape(-1, 1)).to(l2_norm_vec.device))

    # Sometimes, there could be a non-equal length masks as TRUE
    # If so, simply randomly select one of the extra TRUEs and change to false
    # ISSUE: will make the original and vectorized NON-EQUAL! Also will make the thing potentially slow
    if percentile_mask_vec.float().sum(-1).max() != percentile_mask_vec.float().sum(-1).min():
        try:
            index = percentile_mask_vec.float().sum(-1).argmax()

            while percentile_mask_vec.float().sum(-1).max().item() != percentile_mask_vec.float().sum(-1).min().item():
                rnd = np.random.choice(percentile_mask_vec[index].nonzero().cpu().numpy().reshape(-1))
                percentile_mask_vec[index][rnd] = False

        except:
            print(f'MAX: {percentile_mask_vec.float().sum(-1).max()}')
            print(f'MIN: {percentile_mask_vec.float().sum(-1).min()}')
            print(f'PERCENTILE VALUE: {bottom_10_p_vec}')
            print(f'INDEX: {index}')
            qqq

    # We

    prototypes_vec = (next_obs.unsqueeze(0).repeat(train_inputs.shape[0], 1, 1)[percentile_mask_vec] -
                      obs.unsqueeze(0).repeat(train_inputs.shape[0], 1, 1)[percentile_mask_vec]).reshape(
        percentile_mask_vec.shape[0], -1, next_obs.shape[-1])

    # Now we need to concatenate with actions (then chop these off) and standardize
    prototypes_vec = prototypes_vec.mean(1)
    #     prototypes_vec = torch.cat([prototypes_vec, action[:prototypes_vec.shape[0]]], dim=-1)
    #     prototypes_vec = scaler.transform(prototypes_vec)[:, :obs.shape[-1]]

    return prototypes_vec


def find_neighbors_displacement(replay, train_targets, dynamics_ens, scaler, n_sample, percentile):
    # TODO: also try weighting each components' standard deviation in the replay buffer. Then scale each
    # entry by i / std_i
    #     sa = torch.cat([s, a], dim=-1).to('cuda') #.unsqueeze(0)

    batch = replay.sample(n_sample)
    obs, action, next_obs, reward, not_done = batch

    #     sa_replay = torch.cat([obs, action], dim=-1)

    #     scaling_stds = sa_replay.std(0)

    sample_sa, sample_spr = dynamics_ens.preprocess_training_batch(batch)
    sample_sa = scaler.transform(sample_sa)
    #     scaling_stds = sample_sa.std(0)
    #     sample_sa = sample_sa / scaling_stds

    # Need to repeat this along dim 1! One for each row in sa
    # entry for sa[i, :] is diff[:, i, :]
    diff = train_targets - sample_spr[:, :-1].unsqueeze(1).repeat(1, train_targets.shape[0], 1)
    #     diff = diff / scaling_stds

    # Now for the fully-vectorized version!
    l2_norm_vec = torch.linalg.norm(diff, dim=2).T
    bottom_10_p_vec = np.percentile(l2_norm_vec.cpu().numpy(), percentile, axis=-1)
    percentile_mask_vec = (l2_norm_vec <= torch.from_numpy(bottom_10_p_vec.reshape(-1, 1)).to(l2_norm_vec.device))

    # Sometimes, there could be a non-equal length masks as TRUE
    # If so, simply randomly select one of the extra TRUEs and change to false
    # ISSUE: will make the original and vectorized NON-EQUAL! Also will make the thing potentially slow
    if percentile_mask_vec.float().sum(-1).max() != percentile_mask_vec.float().sum(-1).min():
        try:
            index = percentile_mask_vec.float().sum(-1).argmax()

            while percentile_mask_vec.float().sum(-1).max().item() != percentile_mask_vec.float().sum(-1).min().item():
                rnd = np.random.choice(percentile_mask_vec[index].nonzero().cpu().numpy().reshape(-1))
                percentile_mask_vec[index][rnd] = False

        except:
            print(f'MAX: {percentile_mask_vec.float().sum(-1).max()}')
            print(f'MIN: {percentile_mask_vec.float().sum(-1).min()}')
            print(f'PERCENTILE VALUE: {bottom_10_p_vec}')
            print(f'INDEX: {index}')
            qqq

    # We

    prototypes_vec = (next_obs.unsqueeze(0).repeat(train_targets.shape[0], 1, 1)[percentile_mask_vec] -
                      obs.unsqueeze(0).repeat(train_targets.shape[0], 1, 1)[percentile_mask_vec]).reshape(
        percentile_mask_vec.shape[0], -1, next_obs.shape[-1])

    # Now we need to concatenate with actions (then chop these off) and standardize
    prototypes_vec = prototypes_vec.mean(1)
    #     prototypes_vec = torch.cat([prototypes_vec, action[:prototypes_vec.shape[0]]], dim=-1)
    #     prototypes_vec = scaler.transform(prototypes_vec)[:, :obs.shape[-1]]

    return prototypes_vec


def find_neighbors_weighted(replay, train_inputs, dynamics_ens, scaler, n_sample, percentile):
    # TODO: also try weighting each components' standard deviation in the replay buffer. Then scale each
    # entry by i / std_i
    #     sa = torch.cat([s, a], dim=-1).to('cuda') #.unsqueeze(0)

    batch = replay.sample(n_sample)
    obs, action, next_obs, reward, not_done = batch

    #     sa_replay = torch.cat([obs, action], dim=-1)

    #     scaling_stds = sa_replay.std(0)

    sample_sa, sample_spr = dynamics_ens.preprocess_training_batch(batch)
    sample_sa = scaler.transform(sample_sa)
    #     scaling_stds = sample_sa.std(0)
    #     sample_sa = sample_sa / scaling_stds

    # Need to repeat this along dim 1! One for each row in sa
    # entry for sa[i, :] is diff[:, i, :]
    diff = train_inputs - sample_sa.unsqueeze(1).repeat(1, train_inputs.shape[0], 1)
    #     diff = diff / scaling_stds

    # Now for the fully-vectorized version!
    l2_norm_vec = torch.linalg.norm(diff, dim=2).T
    bottom_10_p_vec = np.percentile(l2_norm_vec.cpu().numpy(), percentile, axis=-1)
    percentile_mask_vec = (l2_norm_vec <= torch.from_numpy(bottom_10_p_vec.reshape(-1, 1)).to(l2_norm_vec.device))

    # Sometimes, there could be a non-equal length masks as TRUE
    # If so, simply randomly select one of the extra TRUEs and change to false
    # ISSUE: will make the original and vectorized NON-EQUAL! Also will make the thing potentially slow
    if percentile_mask_vec.float().sum(-1).max() != percentile_mask_vec.float().sum(-1).min():
        index = percentile_mask_vec.float().sum(-1).argmax()

        while percentile_mask_vec.float().sum(-1).max().item() != percentile_mask_vec.float().sum(-1).min().item():
            rnd = np.random.choice(percentile_mask_vec[index].nonzero().cpu().numpy().reshape(-1))
            percentile_mask_vec[index][rnd] = False

    # We
    norm_values = l2_norm_vec[percentile_mask_vec]

    norm_values = torch.softmax(
        -norm_values.reshape(train_inputs.shape[0], int(percentile_mask_vec.float().sum(-1).max().item())),
        dim=-1
    )

    prototypes_vec = (next_obs.unsqueeze(0).repeat(train_inputs.shape[0], 1, 1)[percentile_mask_vec] -
                      obs.unsqueeze(0).repeat(train_inputs.shape[0], 1, 1)[percentile_mask_vec]).reshape(
        percentile_mask_vec.shape[0], -1, next_obs.shape[-1])

    prototypes_vec = prototypes_vec * norm_values.unsqueeze(-1)

    # Now we need to concatenate with actions (then chop these off) and standardize
    prototypes_vec = prototypes_vec.mean(1)
    #     prototypes_vec = torch.cat([prototypes_vec, action[:prototypes_vec.shape[0]]], dim=-1)
    #     prototypes_vec = scaler.transform(prototypes_vec)[:, :obs.shape[-1]]

    return prototypes_vec


def find_neighbors____(replay, train_inputs, dynamics_ens, scaler, n_sample, percentile):
    # TODO: also try weighting each components' standard deviation in the replay buffer. Then scale each
    # entry by i / std_i
    #     sa = torch.cat([s, a], dim=-1).to('cuda') #.unsqueeze(0)

    batch = replay.sample(n_sample)
    obs, action, next_obs, reward, not_done = batch

    #     sa_replay = torch.cat([obs, action], dim=-1)

    #     scaling_stds = sa_replay.std(0)

    sample_sa, sample_spr = dynamics_ens.preprocess_training_batch(batch)
    sample_sa = scaler.transform(sample_sa)
    #     scaling_stds = sample_sa.std(0)
    #     sample_sa = sample_sa / scaling_stds

    # Need to repeat this along dim 1! One for each row in sa
    # entry for sa[i, :] is diff[:, i, :]
    diff = train_inputs - sample_sa.unsqueeze(1).repeat(1, train_inputs.shape[0], 1)
    #     diff = diff / scaling_stds

    # Now for the fully-vectorized version!
    l2_norm_vec = torch.linalg.norm(diff, dim=2).T
    bottom_10_p_vec = np.percentile(l2_norm_vec.cpu().numpy(), percentile, axis=-1)
    percentile_mask_vec = (l2_norm_vec <= torch.from_numpy(bottom_10_p_vec.reshape(-1, 1)).to(l2_norm_vec.device))

    # Sometimes, there could be a non-equal length masks as TRUE
    # If so, simply randomly select one of the extra TRUEs and change to false
    # ISSUE: will make the original and vectorized NON-EQUAL! Also will make the thing potentially slow
    if percentile_mask_vec.float().sum(-1).max() != percentile_mask_vec.float().sum(-1).min():
        index = percentile_mask_vec.float().sum(-1).argmax()

        while percentile_mask_vec.float().sum(-1).max().item() != percentile_mask_vec.float().sum(-1).min().item():
            rnd = np.random.choice(percentile_mask_vec[index].nonzero().cpu().numpy().reshape(-1))
            percentile_mask_vec[index][rnd] = False

    # We

    prototypes_vec = (next_obs.unsqueeze(0).repeat(train_inputs.shape[0], 1, 1)[percentile_mask_vec] -
                      obs.unsqueeze(0).repeat(train_inputs.shape[0], 1, 1)[percentile_mask_vec]).reshape(
        percentile_mask_vec.shape[0], -1, next_obs.shape[-1])

    # Now we need to concatenate with actions (then chop these off) and standardize
    prototypes_vec = prototypes_vec.mean(1)
    #     prototypes_vec = torch.cat([prototypes_vec, action[:prototypes_vec.shape[0]]], dim=-1)
    #     prototypes_vec = scaler.transform(prototypes_vec)[:, :obs.shape[-1]]

    return prototypes_vec


def perform_search(dynamics_ens, depth, n_actions, agent, starting_states, pct_random, env):
    actions_selected = []
    states_predicted = []
    rewards_predicted = []

    # [B, obs_dim]
    s_currents = starting_states
    B = starting_states.shape[0]

    # Setting actions
    random_actions = int(n_actions * pct_random)
    real_actions = n_actions - random_actions

    for i in range(depth):
        # Sampling actions from the agent in the current state. These starting_states need to be torch.Tensors
        # [B * n_actions, obs_dim]
        pi_actions = [agent.actor(s_currents).sample().clamp(*agent.action_range) for _ in range(real_actions)]
        actions = torch.cat(pi_actions, dim=0)

        if random_actions > 0:
            rnd_actions = [
                torch.from_numpy(env.action_space.sample()).float().to(starting_states.device).unsqueeze(0)
                for _ in range(random_actions * B)
            ]
            rnd_actions = torch.cat(rnd_actions, dim=0)
            actions = torch.cat([actions, rnd_actions], dim=0)

        # Combining the current state and actions. First, we need to duplicate the current state along the batch
        # dimension. Then, we concat with actions along dim=-1. Finally, we normalize for input to the dynamics
        # ensemble. [B * n_actions, obs_dim + action_dim]
        sa = torch.cat([torch.cat([s_currents for _ in range(n_actions)], dim=0), actions], dim=-1)
        sa = dynamics_ens.scaler.transform(sa)

        # Measure disagreement at each "proposal" transition. We also want to measure disagreement along the way,
        # as it can give us an indication for MPE. As such, we need to loop over each elite in the ensemble.
        # We don't want to track the autograd graph here.
        with torch.no_grad():
            samples = []

            for mem in dynamics_ens.selected_elites:
                dist = dynamics_ens.forward_models[mem](sa, moments=False)
                s = dist.sample()
                # s = dist.mean
                samples.append(s.unsqueeze(0))

        # [n_elites, B * n_actions, obs_dim + reward_included]
        samples = torch.cat(samples, dim=0)

        # [B * n_actions]
        # Now we have a disagreement number for each of the proposed transitions
        disagreement = (torch.norm(samples - samples.mean(0), dim=-1)).mean(0)

        # Now we reshape to [B, n_actions] because we need to select the action for each input state in the
        # input batch's trajector withh the smallest disagreement
        # disagreement = disagreement.reshape(s_currents.shape[0], n_actions)
        disagreement = torch.cat([x.reshape(B, -1) for x in disagreement.split(B)], dim=-1)

        # [B,]
        min_disagreement_idx = disagreement.argmin(-1)

        # [B, n_actions, action_dim] -> [B, action_dim]
        min_disagreement_actions = torch.cat([x.reshape(B, -1).unsqueeze(1) for x in actions.split(B)], dim=1)[
                                   torch.arange(s_currents.shape[0]), min_disagreement_idx, :
                                   ]
        # min_disagreement_actions = actions.reshape(min_disagreement_idx.shape[0], n_actions, -1)[
        #                            torch.arange(s_currents.shape[0]), min_disagreement_idx, :
        #                            ]

        # Now we want to randomly select one of the members of the ensemble for each step
        # [n_elites, B, n_actions, obs_dim + reward_included]
        samples = torch.cat([x.unsqueeze(2) for x in samples.split(B, 1)], dim=2)
        # samples = samples.reshape(samples.shape[0], s_currents.shape[0], n_actions, -1)

        # Selecting only the states that correspond to the lowest disagreement between ensemble members
        # [n_elites, B, obs_dims + reward_included]
        samples = samples[:, torch.arange(B), min_disagreement_idx, :]
        # samples = samples[:, torch.arange(s_currents.shape[0]), min_disagreement_idx, :]

        # Finally randomly selecting an ensemble member's predicted transition for each minibatch element
        # [B, obs_dim + reward_included]
        samples = samples[
                      [np.random.choice(dynamics_ens.selected_elites) for _ in range(samples.shape[1])],
                      torch.arange(samples.shape[1]),
                      :
                  ]

        states_predicted.append(s_currents.unsqueeze(1))
        rewards_predicted.append(samples[:, -1].unsqueeze(1))
        actions_selected.append(min_disagreement_actions.unsqueeze(1))

        # Adding the predicted state displacement to the current state to get the predicted next-state
        s_currents = s_currents + samples[:, :-1]

    states_predicted.append(s_currents.unsqueeze(1))

    return torch.cat(states_predicted, 1), torch.cat(rewards_predicted, 1), torch.cat(actions_selected, 1)


def augment_minibatch(sa, ds, augment_pct):
    """

    Args:
        sa (tensor): a [B, obs_dim + action_dim] tensor that has already been normalized
        ds (tensor) a [B, obs_dim + reward_included] tensor that represents state displacement for each element in sa
        augment_pct (float): what percent of the minibatch to augment
    Returns:

    """
    sa = deepcopy(sa)
    ds = deepcopy(ds)

    augment_indices = np.random.choice(sa.shape[0], int(sa.shape[0] * augment_pct), replace=False)

    """
    Some states may be "invalid" (e.g., some joint angles may not be possible). I don't think this issue
    occurs for actions. Since the sa input is standardized, we should perhaps clip the values at [-3, 3] or something
    similar. 
    
    Types of sa -> d(s) augmentation:
        (1) Augment sa only. This would lead to many sa mapping to a single d(s)
        (2) Augment d(s) only. This would lead to a single sa mapping to many d(s).
        (3) Augment both. This would lead to many sa mapping to many d(s).
        
    Is (3) the most desirable?
    """
    # First, let's just try them all!
    augmentation_type = np.random.choice(3)

    if augmentation_type == 0:
        # sa only. Small uniform random noise?
        sa[augment_indices] += td.Uniform(
            low=torch.zeros_like(sa[augment_indices]) - 0.5,
            high=torch.zeros_like(sa[augment_indices]) + 0.5
        ).sample().to(sa.device)

    elif augmentation_type == 1:
        # d(s) only (not reward at the moment)
        ds[augment_indices, :-1] += td.Uniform(
            low=torch.zeros_like(ds[augment_indices, :-1]) - 0.5,
            high=torch.zeros_like(ds[augment_indices, :-1]) + 0.5
        ).sample().to(sa.device)

    else:
        # both sa and d(s)
        sa[augment_indices] += td.Uniform(
            low=torch.zeros_like(sa[augment_indices]) - 0.5,
            high=torch.zeros_like(sa[augment_indices]) + 0.5
        ).sample().to(sa.device)

        ds[augment_indices, :-1] += td.Uniform(
            low=torch.zeros_like(ds[augment_indices, :-1]) - 0.5,
            high=torch.zeros_like(ds[augment_indices, :-1]) + 0.5
        ).sample().to(sa.device)

    return sa, ds


def symlog(x):
    return torch.sign(x) * torch.log(x.abs() + 1)


def inv_symlog(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)

