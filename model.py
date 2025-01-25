import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
from pytorch_lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, IterableDataset

from swap_env import SwapEnv
from utils import to_device


def collate_fn_ppo(batch):
    states, actions, logp_olds, v_olds, qvals, advs = zip(*batch)
    new_states = {}
    new_states["mask"] = torch.stack([state["mask"] for state in states])
    new_states["fac_data"] = torch_geometric.data.Batch.from_data_list(
        [state["fac_data"] for state in states]
    )
    actions = torch.stack(actions)
    logp_olds = torch.stack(logp_olds)
    v_olds = torch.as_tensor(v_olds, dtype=torch.float).unsqueeze(-1)
    qvals = torch.as_tensor(qvals, dtype=torch.float).unsqueeze(-1)
    advs = torch.as_tensor(advs, dtype=torch.float).unsqueeze(-1)
    return (new_states, actions, logp_olds, v_olds, qvals, advs)



class GraphFeatureExtractor(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", **kwargs):
        super().__init__()

        gnn_layer = getattr(geom_nn, layer_name)

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, out_channels, **kwargs),
                nn.ReLU(inplace=True),
            ]
            in_channels = c_hidden * kwargs["heads"]
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            if isinstance(layer, geom_nn.GATv2Conv):
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index, edge_weight=edge_attr)
            else:
                x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2):
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ActorCritic(nn.Module):
    def __init__(
        self, fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
    ) -> None:
        super().__init__()
        if "heads" not in kwargs:
            kwargs["heads"] = 1
        emb_size = c_out * kwargs["heads"] * 2

        self.actor_gnn = GraphFeatureExtractor(
            fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
        )
        self.actor_prob = MLP(emb_size, c_hidden, 1, num_layers)
        self.att = nn.Linear(emb_size, emb_size, bias=False)

        self.critic_gnn = GraphFeatureExtractor(
            fac_c_in, c_hidden, c_out, num_layers, layer_name, **kwargs
        )
        self.critic = MLP(emb_size, c_hidden, 1, num_layers)

    def actor_forward(self, state, action1=None):
        batch_fac, mask = state["fac_data"], state["mask"]
        batch = batch_fac.batch
        if batch is None:
            batch = torch.zeros(
                batch_fac.num_nodes, dtype=torch.long, device=batch_fac.x.device
            )

        emb_fac = self.actor_gnn(batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr)
        pooling = geom_nn.global_mean_pool(emb_fac, batch)
        emb_fac = torch.cat([emb_fac, pooling[batch]], dim=-1)

        act_scores1 = self.actor_prob(emb_fac).reshape(pooling.shape[0], -1)
        mask1 = torch.where(mask, -float("inf"), 0)
        logits1 = act_scores1 + mask1
        pi1 = Categorical(logits=logits1)
        if action1 is None:
            action1 = pi1.sample()

        if batch_fac.batch is not None:
            action1_inc = action1 + batch_fac.ptr[:-1]
        else:
            action1_inc = action1

        feat_act = torch.tanh(self.att(emb_fac[action1_inc]))
        act_scores2 = torch.matmul(emb_fac, feat_act.T)
        act_scores2 = act_scores2[torch.arange(act_scores2.shape[0]), batch]
        act_scores2 = act_scores2.reshape(pooling.shape[0], -1)
        mask2 = torch.where(mask, 0, -float("inf"))
        logits2 = act_scores2 + mask2
        pi2 = Categorical(logits=logits2)
        action2 = pi2.sample()

        logits = torch.stack([logits1, logits2], dim=1).squeeze(0)
        action = torch.stack([action1, action2], dim=1).squeeze(0)
        return Categorical(logits=logits), action

    def critic_forward(self, state):
        batch_fac = state["fac_data"]
        emb_fac = self.critic_gnn(
            batch_fac.x, batch_fac.edge_index, batch_fac.edge_attr
        )
        mean_pool = geom_nn.global_mean_pool(emb_fac, batch_fac.batch)
        max_pool = geom_nn.global_max_pool(emb_fac, batch_fac.batch)
        emb_global = torch.cat([mean_pool, max_pool], dim=-1) 
        score = self.critic(emb_global).squeeze(0)
        return score

    def get_log_prob(self, pi: Categorical, actions):
        return pi.log_prob(actions)

    def actor_loss(self, state, action, logp_old, qval, adv, clip_ratio):
        pi, _ = self.actor_forward(state)
        logp = self.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        with torch.no_grad():
            log_ratio = logp - logp_old
            approx_kl_div = (
                torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy().item()
            )

        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_ratio).float()).item()

        entropy_loss = -pi.entropy().mean()
        return loss_actor, entropy_loss, approx_kl_div, clip_fraction

    def critic_loss(self, state, action, logp_old, v_old, qval, adv, clip_ratio):
        value = self.critic_forward(state)
        clip_value = v_old + torch.clamp(value - v_old, -clip_ratio, clip_ratio)
        v_max = torch.max((qval - value).pow(2), (qval - clip_value).pow(2))
        loss_critic = v_max.mean()
        return loss_critic


class ExperienceSourceDataset(IterableDataset):
    def __init__(self, generate_batch):
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator


class PPOLightning(LightningModule):
    def __init__(
        self,
        model_params: dict = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 5e-4,
        lr_gamma: float = 0.99,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        clip_decay: float = 1,
        ent_weight: float = 0.01,
        critic_weight: float = 0.5,
        gradient_clip_val: float = None,
        mode: str = "train",
        data_path: str = "./data/train/",
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        if mode == "test":
            self.actor_critic = ActorCritic(**self.hparams.model_params)
            return

        self.automatic_optimization = False

        self.env = SwapEnv(data_path=data_path)

        self.actor_critic = ActorCritic(**self.hparams.model_params)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.batch_v = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = self.env.reset()[0]

    def forward(self, x: torch.Tensor):
        pi, action = self.actor_critic.actor_forward(x)
        value = self.actor_critic.critic_forward(x)

        return pi, action, value

    def predict(self, x):
        return self.actor_critic(x)

    def discount_rewards(self, rewards, discount):
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards, values, last_value):
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [
            rews[i] + self.hparams.gamma * vals[i + 1] - vals[i]
            for i in range(len(rews) - 1)
        ]
        adv = self.discount_rewards(delta, self.hparams.gamma * self.hparams.lam)

        return adv

    def generate_trajectory_samples(self):
        for step in range(self.hparams.steps_per_epoch):
            to_device(self.state, self.device)

            with torch.no_grad():
                pi, action, value = self(self.state)
                log_prob = self.actor_critic.get_log_prob(pi, action)
                self.logger.log_metrics(
                    {"entropy/pi": pi.entropy().mean()}, self.global_step
                )

            next_state, reward, done, truncated, _ = self.env.step(
                action.squeeze().cpu().numpy()
            )

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)
            self.batch_v.append(value)

            self.ep_rewards.append(reward.item())
            self.ep_values.append(value.item())

            self.state = next_state

            epoch_end = step == (self.hparams.steps_per_epoch - 1)

            if epoch_end or done or truncated:
                if not done:
                    to_device(self.state, self.device)
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.hparams.gamma
                )[:-1]
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )
                self.epoch_rewards.append(sum(self.ep_rewards))
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = self.env.reset()[0]

            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_v,
                    self.batch_qvals,
                    self.batch_adv,
                )
                for state, action, logp_old, v_old, qval, adv in train_data:
                    yield state, action, logp_old, v_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_v.clear()
                self.batch_qvals.clear()

                self.avg_reward = sum(self.epoch_rewards) / self.hparams.steps_per_epoch

                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.hparams.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/avg_ep_reward": -1})

    def training_step(self, batch, batch_idx):
        state, action, old_logp, v_old, qval, adv = batch

        if self.hparams.batch_size > 1:
            adv = (adv - adv.mean()) / adv.std()

        self.log(
            "hp/avg_ep_reward",
            self.avg_ep_reward,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        (
            loss_actor,
            entropy_loss,
            approx_kl_div,
            clip_fraction,
        ) = self.actor_critic.actor_loss(
            state, action, old_logp, qval, adv, self.hparams.clip_ratio
        )

        loss_critic = self.actor_critic.critic_loss(
            state, action, old_logp, v_old, qval, adv, self.hparams.clip_ratio
        )
        loss = (
            self.hparams.ent_weight * entropy_loss
            + loss_actor
            + self.hparams.critic_weight * loss_critic
        )

        self.manual_backward(loss)
        if self.hparams.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx + 1 == self.hparams.steps_per_epoch // self.hparams.batch_size:
            scheduler.step()
            self.hparams.clip_ratio *= self.hparams.clip_decay

        self.log(
            "loss/loss_critic",
            loss_critic,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/loss_actor",
            loss_actor,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "loss/loss_entropy",
            entropy_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_gamma
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        for _ in range(self.hparams.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn_ppo,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()

