import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from networks.forward_models import MLP, GRU
from networks.distributions import DistLayer
import torch.distributions as td
from utils.scalers import StandardScaler
from copy import deepcopy


class DynamicsEnsemble(nn.Module):
    def __init__(self, n_ensemble_members, obs_dim, act_dim, hidden_dims, activation, norm, dist, max_logging,
                 reward_included, predict_difference, batch_size, lr, early_stop_patience, n_elites, terminal_fn,
                 rnn, reward_penalty, reward_penalty_weight, classifier, replay, logger, threshold, lcc, device):
        """

        Args:
            n_ensemble_members:
            obs_dim:
            act_dim:
            hidden_dims:
            activation:
            norm (bool):
            dist (str):
            max_logging (int): # TODO: rename this??? orig impl doesn't explain
            reward_included (bool): whether the reward prediction is included directly in forward model output
            predict_difference (bool): whether the forward models predict s' - s
            batch_size
            lr
            early_stop_patience
            n_elites
            terminal_fn
            rnn:
            reward_penalty:
            reward_penalty_weight:
            classifier:
            replay:
            logger:
            threshold:
            lcc:
            device:
        """
        super().__init__()
        self.n_ensemble_members = n_ensemble_members
        self.reward_included = reward_included
        self.predict_difference = predict_difference
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.n_elites = n_elites
        self.terminal_fn = terminal_fn
        self.rnn = rnn
        # self.reward_penalty = reward_penalty
        # self.reward_penalty_weight = reward_penalty_weight
        self.classifier = classifier
        self.replay = replay
        self.threshold = threshold
        self.logger = logger
        self.lcc = lcc
        self.obs_dim = obs_dim
        self.reward_included = reward_included

        if not rnn:
            self.forward_models = nn.ModuleList([
                MLP(obs_dim + act_dim, hidden_dims, obs_dim + reward_included, activation, norm, dist)
                for _ in range(n_ensemble_members)
            ])

        else:
            # If we have a GRU model: prev_state X action -> next_state
            self.forward_models = nn.ModuleList([
                GRU(obs_dim, act_dim, reward_included, norm, activation, dist)
                for _ in range(n_ensemble_members)
            ])

        # Currently only support functional form where the reward is the last index of the forward model's output
        # Instead, we may wish to predict the reward with a separate MLP
        if not reward_included:
            raise NotImplemented('Sorry!')

        self.max_logging = max_logging
        self.scaler = StandardScaler()

        self.device = device
        self.to(device)

        self.ensemble_optim = torch.optim.Adam(self.forward_models.parameters(), lr=lr, weight_decay=1e-5)

        self.selected_elites = np.array([i for i in range(n_ensemble_members)])

    def step(self, observations, actions, deterministic):
        """
        Performs a single step

        Args:
            observations:
            actions:
            deterministic:

        Returns:

        """

        obs_act = torch.cat([observations, actions], dim=-1)
        obs_act = self.scaler.transform(obs_act)

        with torch.no_grad():
            if deterministic:
                means = []
                # logvars = []
                for i in range(self.n_ensemble_members):
                    mean, _ = self.forward_models[i](obs_act)
                    means.append(mean.unsqueeze(0))
                    # logvars.append(logvar.unsqueeze(0))

                means = torch.cat(means, dim=0)
                # logvars = torch.cat(logvars, dim=0)

                # Take mean over ensemble members' means
                samples = torch.mean(means, dim=0)

                # reward_penalty = self.compute_reward_penalty(
                #     means,
                #     self.reward_penalty,
                #     obs_act,
                #     samples,
                # )

            else:
                samples = []
                means = []
                for i in range(self.n_ensemble_members):
                    dist = self.forward_models[i](obs_act, moments=False)
                    samples.append(dist.rsample().unsqueeze(0))
                    means.append(dist.loc.unsqueeze(0))
                    # samples.append(self.forward_models[i](obs_act, moments=False).rsample().unsqueeze(0))

                # [n_ens, B, obs_dim + reward_included]
                samples = torch.cat(samples, dim=0)
                means = torch.cat(means, dim=0)

                # reward_penalty = self.compute_reward_penalty(means, self.reward_penalty)
                idxs = np.random.choice(self.selected_elites, size=samples[0].shape[0])

                # [B, obs_dim + reward_included] where each i \in B is from a randomly sampled ensemble member
                samples = samples[idxs, np.arange(0, samples[0].shape[0])]

                # reward_penalty = self.compute_reward_penalty(
                #     means,
                #     self.reward_penalty,
                #     obs_act,
                #     samples,
                # )

        # if self.reward_penalty == 'dual_classifier':
        #     reward_penalty, prototypes, prototype_mask = reward_penalty
        #     if prototypes is not None:
        #         samples[prototype_mask, :-1] = prototypes
        #
        # if self.reward_penalty == 'disagreement_threshold':
        #     prototypes, prototype_mask = reward_penalty
        #     if prototypes is not None:
        #         samples[prototype_mask, :-1] = prototypes
        #
        #     reward_penalty = 0
        #
        # if self.reward_penalty == 'plaus_no_disagreement':
        #     prototypes, prototype_mask = reward_penalty
        #     if prototypes is not None:
        #         samples[prototype_mask, :-1] = prototypes
        #
        #     reward_penalty = 0

        if self.reward_included:
            # Chopping of rewards
            rewards = samples[:, -1]
            # rewards = rewards - self.reward_penalty_weight * reward_penalty

        else:
            raise NotImplemented('Sorry!')

        if self.predict_difference:
            next_obs = samples[:, :-1] + observations
        else:
            next_obs = samples[:, :-1]

        if not self.terminal_fn:
            terminals = torch.zeros_like(rewards).bool()
        else:
            # TODO: ensure we have all of the correct terminal functions for our envs
            terminals = self.terminal_fn(observations, actions, next_obs).squeeze(dim=-1)

        return next_obs, rewards, terminals, {}

    def rnn_step(self, observations, actions, deterministic):
        """"""
        with torch.no_grad():
            if deterministic:
                means = []
                for i in range(self.n_ensemble_members):
                    mean, _ = self.forward_models[i](
                        actions, observations
                    )
                    means.append(mean.unsqueeze(0))

                means = torch.cat(means, dim=0)
                samples = torch.mean(means, dim=0)

            else:
                samples = []
                for i in range(self.n_ensemble_members):
                    state = self.forward_models[i](
                        actions, observations, moments=False
                    ).rsample()
                    samples.append(state.unsqueeze(0))

                # [n_ens, B, obs_dim + reward_included]
                samples = torch.cat(samples, dim=0)
                idxs = np.random.choice(self.selected_elites, size=samples[0].shape[0])

                # [B, obs_dim + reward_included] where each i \in B is from a randomly sampled ensemble member
                samples = samples[idxs, np.arange(0, samples[0].shape[0])]

        if self.reward_included:
            # Chopping of rewards
            rewards = samples[:, -1]

        else:
            raise NotImplemented('Sorry!')

        if self.predict_difference:
            next_obs = samples[:, :-1] + observations
        else:
            next_obs = samples[:, :-1]

        if not self.terminal_fn:
            terminals = torch.zeros_like(rewards).bool()
        else:
            # TODO: ???
            terminals = self.terminal_fn(observations, actions, next_obs).squeeze(dim=-1)

        return next_obs, rewards, terminals, {}

    def train_single_step(self, replay_buffer, validation_ratio, batch_size, online_buffer=None):
        """
        Trains the ensemble of dynamics models with single-step predictions only!
        Here, each member of the ensemble is guaranteed to see the different training data?
        Args:
            replay_buffer:
            validation_ratio:
            batch_size:

        Returns:

        """

        # data_size = min(replay_buffer.size, self.max_logging)
        val_size = int(batch_size * validation_ratio)
        train_size = batch_size - val_size
        # val_size = min(int(data_size * validation_ratio), self.max_logging)
        # train_size = data_size - val_size

        if online_buffer is not None:
            if isinstance(online_buffer, tuple):
                train_batch, val_batch = replay_buffer.random_split(val_size, batch_size * 10)

                train_batch = [torch.cat((offline_item, online_item), dim=0) for offline_item, online_item in
                               zip(train_batch, online_buffer)]


            else:
                train_batch, val_batch = replay_buffer.random_split(val_size, batch_size * 10)
                train_batch_online, val_batch_online = online_buffer.random_split(val_size, batch_size * 10)

                train_batch = [torch.cat((offline_item, online_item), dim=0) for offline_item, online_item in
                         zip(train_batch, train_batch_online)]

                val_batch = [torch.cat((offline_item, online_item), dim=0) for offline_item, online_item in
                               zip(val_batch, val_batch_online)]

        else:
            train_batch, val_batch = replay_buffer.random_split(val_size, batch_size * 10)
            # train_batch, val_batch = replay_buffer.random_split(val_size)

        train_inputs, train_targets = self.preprocess_training_batch(train_batch)
        val_inputs, val_targets = self.preprocess_training_batch(val_batch)

        # need to re-adjust train size in case we are querying for more examples than exist
        train_size = train_inputs.shape[0]

        # calculate mean and var used for normalizing inputs
        # MOVED THIS TO A GLOBAL COMPUTATION IN THE OFFLINE DATA
        # if update_scaler:
        #     self.scaler.fit(train_inputs)
        train_inputs, val_inputs = self.scaler.transform(train_inputs), self.scaler.transform(val_inputs)

        # Entering the actual training loop
        self.val_loss = [1e5 for _ in range(self.n_ensemble_members)]
        epoch = 0
        self.cnt = 0
        early_stop = False

        idxs = np.random.randint(train_size, size=[self.n_ensemble_members, train_size])

        loss_hist = []
        while not early_stop:
            for b in range(int(np.ceil(train_size / self.batch_size))):
                batch_idxs = idxs[:, b * self.batch_size:(b + 1) * self.batch_size]

                # In the next-step prediction process, we do not sample
                means = []
                logvars = []

                for i in range(self.n_ensemble_members):
                    mean, logvar = self.forward_models[i](train_inputs[batch_idxs[i], :])
                    means.append(mean.unsqueeze(0))
                    logvars.append(logvar.unsqueeze(0))

                # [n_ens, B, obs_dim + include_reward]
                means = torch.concat(means, dim=0)
                logvars = torch.concat(logvars, dim=0)

                inv_var = torch.exp(-logvars)
                mse_loss = (((means - train_targets[batch_idxs, :]) ** 2) * inv_var).mean(dim=[1, 2])
                var_loss = logvars.mean(dim=[1, 2])

                self.logger.log({
                    'mse_loss': mse_loss.sum().detach().cpu().item(),
                    'var_loss': var_loss.sum().detach().cpu().item()
                })

                # Summing across each member of the ensemble
                loss = (mse_loss + var_loss).sum()

                for i in range(self.n_ensemble_members):
                    loss += 0.01 * self.forward_models[i].max_logvar.sum() - 0.01 * self.forward_models[i].min_logvar.sum()

                self.ensemble_optim.zero_grad()
                loss.backward()
                loss_hist.append(loss.item())
                self.ensemble_optim.step()

                if self.lcc:
                    # Projecting the weight matrices back into the feasible set of the constrained optimization problem.
                    for fm in self.forward_models:
                        fm.apply(self.lcc)

            # Shuffling the idxs
            self.shuffle_rows(idxs)

            new_val_loss = self.evaluate(val_inputs, val_targets, None)

            # print("In Epoch {e}, \nthe model_loss is : {m}, \nthe val_loss is: {v}".format(
            #     e=epoch, m=model_loss, v=new_val_loss)
            # )

            early_stop = self._is_early_stop(new_val_loss)
            epoch += 1

        # When we reach here, training is over!
        # Now, let's select the new "elites"!
        val_losses = self.evaluate(val_inputs, val_targets, None)
        sorted_idxs = np.argsort(val_losses)
        self.set_elites(sorted_idxs[:self.n_elites])
        return loss_hist

    def train_multi_step(self, replay_buffer, validation_ratio, horizon, episode_length):
        """
        Trains the ensemble of dynamics models with multiple single-step predictions strung together
        Here, each member of the ensemble is guaranteed to see different transitions
        Args:
            replay_buffer:
            validation_ratio:
            horizon:
            episode_length

        Returns:

        """
        # TODO: does this need to be scaled (e.g., // horizon)?
        data_size = min(replay_buffer.n_traj, self.max_logging)
        val_size = min(int(data_size * validation_ratio), self.max_logging)
        train_size = data_size - val_size

        # Sampling trajectories instead of single-step transitions
        train_batch, val_batch = replay_buffer.random_split_traj(val_size, horizon, episode_length)

        recurrent_states = train_batch[0]
        recurrent_states_val = val_batch[0]

        train_inputs, train_targets = self.preprocess_training_batch_traj(train_batch)
        val_inputs, val_targets = self.preprocess_training_batch_traj(val_batch)

        # TODO: the inputs are actions. These are already [-1, 1], so don't need to do this?
        # self.scaler.fit(train_inputs, traj=True)
        # train_inputs, val_inputs = self.scaler.transform(train_inputs), self.scaler.transform(val_inputs)

        # Entering the actual training loop
        self.val_loss = [1e5 for _ in range(self.n_ensemble_members)]
        epoch = 0
        self.cnt = 0
        early_stop = False

        # These indices: [n_ens, train_size]
        # Instead of niavely going through every traj with every ensemble member, we will
        # simply randomly sample!
        idxs = np.random.randint(train_size - 1, size=train_size)

        # TODO: currently only supports probabilistic transitions. Is this what we want?
        # From PlaNet paper:
        """
        (a) Transitions in a recurrent neural network are purely deterministic. This prevents
        the model from capturing multiple futures and makes it easy for the planner to exploit inaccuracies. (b) Transitions in a
        state-space model are purely stochastic. This makes it difficult to remember information over multiple time steps
        """
        loss_hist = []
        # print(f'NUM BATCHES: {range(int(np.ceil(train_size / self.batch_size)))} // {train_inputs.shape}')
        while not early_stop:
            for b in range(int(np.ceil(train_size / self.batch_size))):
                batch_idxs = idxs[b * self.batch_size: (b + 1) * self.batch_size]
                ens_member_order = np.random.choice(self.n_ensemble_members, size=horizon)
                traj_batch = train_inputs[batch_idxs, :, :]

                ns_preds = []
                nr_preds = []
                logvars = []
                for i in range(horizon):
                    if i == 0:
                        # Creates RNN initial state of all 0s
                        state = self.forward_models[0].get_initial_state(traj_batch.shape[0]).to(self.device)
                        # state = recurrent_states[batch_idxs, i, :]
                    # state = self.forward_models[ens_member_order[i]](
                    #     traj_batch[:, i, :], state, moments=False
                    # ).rsample()
                    state, logvar = self.forward_models[ens_member_order[i]](
                        traj_batch[:, i, :], state
                    )

                    ns_preds.append(state[:, :-1].unsqueeze(1))
                    nr_preds.append(state[:, -1].unsqueeze(1))
                    logvars.append(logvar.unsqueeze(1))

                    state = state[:, :-1] # + recurrent_states[batch_idxs, i, :]
                    # print(f'STATE: {state.shape}')
                    # print(f'RECURR: {recurrent_states[batch_idxs, i, :].shape}')
                    # qqq

                # ns_preds = [[B, 1, state_dim]] -> [B, T, state_dim]

                """
                ns_preds = torch.cat(ns_preds, dim=1)
                nr_preds = torch.cat(nr_preds, dim=1)
                state_loss = ((ns_preds - train_targets[batch_idxs, :, :-1]) ** 2).sum(dim=[1, 2]).mean()
                reward_loss = ((nr_preds - train_targets[batch_idxs, :, -1]) ** 2).sum(dim=1).mean()
                loss = state_loss + reward_loss

                for i in range(self.n_ensemble_members):
                    loss += 0.01 * self.forward_models[i].max_logvar.sum() - 0.01 * self.forward_models[i].min_logvar.sum()
                """
                ns_preds = torch.cat(ns_preds, dim=1)
                nr_preds = torch.cat(nr_preds, dim=1)

                logvars = torch.concat(logvars, dim=0)
                # inv_var = torch.exp(-logvars)

                state_loss = (
                        ((ns_preds - train_targets[batch_idxs, :, :-1]) ** 2)
                ).sum(dim=[1, 2]).mean()

                reward_loss = (
                        (nr_preds - train_targets[batch_idxs, :, -1]) ** 2
               ).sum(dim=1).mean()

                # var_loss = logvars.sum(dim=[1, 2]).mean()

                loss = state_loss + reward_loss # + var_loss

                for i in range(self.n_ensemble_members):
                    loss += 0.01 * self.forward_models[i].max_logvar.sum() - 0.01 * self.forward_models[i].min_logvar.sum()

                self.ensemble_optim.zero_grad()
                loss.backward()
                loss_hist.append(loss.item())
                self.ensemble_optim.step()

            # print(np.mean(loss_hist[-5:]))
            new_val_loss = self.evaluate_traj(val_inputs, val_targets, recurrent_states_val)
            early_stop = self._is_early_stop(new_val_loss)
            epoch += 1

        # When we reach here, training is over!
        # Now, let's select the new "elites"!
        val_losses = self.evaluate_traj(val_inputs, val_targets, recurrent_states_val)
        sorted_idxs = np.argsort(val_losses)
        self.set_elites(sorted_idxs[:self.n_elites])

    def set_elites(self, selected_idxs):
        self.selected_elites = np.array(selected_idxs)

    def _is_early_stop(self, new_val_loss):
        # print(new_val_loss)
        changed = False
        for i, old_loss, new_loss in zip(range(len(self.val_loss)), self.val_loss, new_val_loss):
            if (old_loss - new_loss) / old_loss > 0.01:
                changed = True
                self.val_loss[i] = new_loss

        if changed:
            self.cnt = 0
        else:
            self.cnt += 1

        if self.cnt >= self.early_stop_patience:
            return True
        else:
            return False

    @torch.no_grad()
    def evaluate(self, inputs, targets, idxs):
        if idxs is not None:
            means = []
            for i in range(self.n_ensemble_members):
                mean, _ = self.forward_models[i](inputs[idxs[i, :self.max_logging]])
                means.append(mean.unsqueeze(0))

            means = torch.cat(means, dim=0)
            loss = ((means - targets) ** 2).mean(dim=[1, 2])

        else:
            means = []
            for i in range(self.n_ensemble_members):
                mean, _ = self.forward_models[i](inputs)
                means.append(mean.unsqueeze(0))

            means = torch.cat(means, dim=0)
            loss = ((means - targets) ** 2).mean(dim=[1, 2])

        return loss.cpu().numpy()

    @torch.no_grad()
    def evaluate_traj(self, inputs, targets, real_states):
        losses = []
        # TODO: do we eval based off of the means?
        means = []
        for i in range(self.n_ensemble_members):
            inner = []

            for j in range(inputs.shape[1]):
                if j == 0:
                    state = self.forward_models[0].get_initial_state(inputs.shape[0]).to(self.device)
                    # state = real_states[:, 0, :]
                state, _ = self.forward_models[i](inputs[:, j, :], state)
                inner.append(state.unsqueeze(1))
                state = state[:, :-1] #+ real_states[:, j, :]

            means.append(torch.cat(inner, dim=1))

        for i in range(self.n_ensemble_members):
            losses.append(((means[i] - targets) ** 2).sum(dim=[1, 2]).mean().cpu().numpy())

        return np.array(losses)

    @staticmethod
    def shuffle_rows(arr):
        """ Shuffle among rows. This will keep distinct training for each ensemble."""
        idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
        return arr[np.arange(arr.shape[0])[:, None], idxs]

    def preprocess_training_batch(self, data):
        # TODO: rename. also used for val
        """

        Args:
            data:
            predict_difference (bool): changes target. Should the dynamics models be trained to predict the **change**
                in the state?

        Returns:

        """
        # TODO: check if this still works well for traj data
        states, actions, next_states, rewards, not_dones = data

        inputs = torch.cat([states, actions], dim=-1)

        if self.predict_difference:
            target_state = next_states - states

        else:
            target_state = next_states

        if self.reward_included:
            target = torch.cat([target_state, rewards], dim=-1)

        else:
            target = target_state

        return inputs, target

    def preprocess_training_batch_traj(self, data):
        """"""
        # For a trajectory, the inputs are simply the [recurrent state, action] vectors
        states, actions, next_states, rewards, not_dones = data
        inputs = actions

        if self.predict_difference:
            target_state = next_states - states

        else:
            target_state = next_states

        if self.reward_included:
            target = torch.cat([target_state, rewards], dim=-1)

        else:
            target = target_state

        return inputs, target

    def imagine(self, rollout_batch_size, horizon, policy, env_replay_buffer, model_replay_buffer, termination_fn, rnd, search_replay_buffer=None):
        """
        Adds experiences to the model_replay_buffer using imagined trajectories
        Args:
            rollout_batch_size:
            horizon:
            policy:
            env_replay_buffer:
            model_replay_buffer:
            termination_fn:

        Returns:

        """
        # Starting the rollout from a real state, from anywhere in a trajectory
        # index [0] selects state from the given sample
        if not search_replay_buffer:
            states = env_replay_buffer.sample(rollout_batch_size)[0]

        if search_replay_buffer is not None:
            states = env_replay_buffer.sample(rollout_batch_size // 2)[0]
            scan_states = search_replay_buffer.sample(rollout_batch_size // 2)[0]
            states = torch.cat([states, scan_states], dim=0)

        steps_added = []
        steps_taken = []
        for i in range(horizon):
            """
            From MBPO paper:
            To generate a prediction from the ensemble, we simply select a model uniformly at random, allowing for 
            different transitions along a single model rollout to be sampled from different dynamics models.
            """
            if rnd:
                actions = rnd(states)

            else:
                with torch.no_grad():
                    actions = policy(states).sample()
                # qqq
            # if rnd:
            #     actions = td.Uniform(
            #         low=torch.zeros_like(actions) - 1.,
            #         high=torch.zeros_like(actions) + 1.
            #     ).sample().float().to(actions.device)

            if self.rnn:
                next_states, rewards, dones, info = self.rnn_step(
                    states, actions, deterministic=False
                )
            else:
                next_states, rewards, dones, info = self.step(
                    states, actions, deterministic=False
                )

            if termination_fn:
                # print(f'NS: {next_states.shape}')
                dones = termination_fn(states, actions, next_states)
                # print(f'{1 - dones.float().mean()}% kept')
                # print(f'R: {rewards.mean()} +- {rewards.std()}')
            # If a model has reached a terminal state, we need to make sure we don't keep rolling out along the
            # queried trajectory
            steps_taken.append([
                states.cpu().numpy(),
                actions.cpu().numpy(),
                rewards.cpu().numpy(),
                next_states.cpu().numpy(),
                dones.cpu().numpy()
            ])

            # steps_added.append(states.shape[0])
            # nonterm_mask = torch.logical_not(dones)
            #
            # if nonterm_mask.sum() == 0:
            #     print(
            #         '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
            #     break
            #
            # states = next_states[nonterm_mask]
            states = next_states

        # The allowed_flag variable control whether a predicted transition is entered into the model's
        # replay buffer. We will not allow a transition into the replay buffer once a trajectory has
        # terminated.
        allowed_flag = np.ones_like(steps_taken[-1][-1]).astype(np.bool_)
        # print(f'b4: {model_replay_buffer.pointer}')
        for step in steps_taken:
            states, actions, rewards, next_states, dones = step
            # print(f'DONES: {dones}')
            model_replay_buffer.add_batch(
                states[allowed_flag.flatten(), :],
                actions[allowed_flag.flatten(), :],
                rewards[allowed_flag.flatten()],
                next_states[allowed_flag.flatten(), :],
                dones[allowed_flag.flatten()]
            )

            # The below logic will only allow future steps to be added IFF both the current step and next step are True
            # As soon as one is False, the followup steps will be false. We need to use ~dones because the dones
            # returned by the model/termination_fn is True if terminated and False if not. The replay buffer stores
            # dones as not_dones = 1 - dones
            allowed_flag = (allowed_flag & ~dones)
            # print(f'ALLOWED: {allowed_flag}')

        # print(f'after: {model_replay_buffer.pointer}')
        # qqqq
        # mean_rollout_length = sum(steps_added) / rollout_batch_size
        # return {"Rollout": mean_rollout_length}

    @torch.no_grad()
    def measure_disagreement(self, input, target):
        """"""
        samples = []

        # Here, we are working with non-trajectory data
        if not self.rnn:

            # encoded_train_inputs = torch.cat([
            #     self.encoder(input[:, :self.obs_dim]), input[:, self.obs_dim:]
            # ], dim=-1)

            for i in range(self.n_ensemble_members):
                mean, logvar = self.forward_models[i](input)
                # mean, logvar = self.forward_models[i](encoded_train_inputs)
                samples.append(mean.unsqueeze(0))

            # [n_ens, B, state_dim + with_reward]
            samples = torch.cat(samples, dim=0)

            mean = samples.mean(0)

            # penalty = torch.norm(mean - mean_pred, dim=[-1, -2]).mean(0)
            # print(torch.norm(samples - mean, dim=-1).mean(0).shape)
            # qqq
            # disagreement = torch.norm()
            # mean_diff = samples - mean

            # print(f'Mean diff: {mean_diff.pow(2).sum(-1).mean().shape}')
            #
            #
            # for i in range(self.n_ensemble_members):
            #     assert ((samples[i] - mean) - (mean_diff[i])).sum() <= 1e-5
            # qqq

            disagreement = (torch.norm(samples - mean, dim=-1)).mean()#.sum()
            # disagreement = mean_diff.pow(2).sum(-1).mean()
            # disagreement =

            if target is not None:
                error_dict = {}
                errors = []
                for i in range(self.n_ensemble_members):
                    error = (samples[i] - target).abs()
                    # error = samples[i] - target
                    # error = (samples[i] - torch.cat([self.encoder(target[:, :-1]), target[:, -1].unsqueeze(-1)], dim=-1)).abs()
                    errors.append(error.mean().cpu().item())
                    error_dict[i] = error.mean().cpu().item()
                return disagreement, np.mean(errors), error_dict
            return disagreement

        # Here be trajectories
        else:
            for i in range(self.n_ensemble_members):
                inner = []
                for j in range(input.shape[1]):
                    if j == 0:
                        state = self.forward_models[0].get_initial_state(input.shape[0]).to(self.device)
                    state = self.forward_models[i](
                        input[:, j, :], state, moments=False
                    ).rsample()

                    inner.append(state.unsqueeze(1))
                    state = state[:, :-1]

                samples.append(torch.cat(inner, dim=1).unsqueeze(0))

            # [n_ens, B, T, state_dim + with_reward]
            samples = torch.cat(samples, dim=0)

            # mean across all ensemble members [B, T, state_dim + with_reward]
            mean = samples.mean(0)

            return (samples - mean).pow(2).sum(-1).mean([0, 1])
