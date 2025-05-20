import math
import os
from abc import ABC
from typing import Dict, Optional, Union

import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from models.loss import ValueLoss
from models.model import get_llm_for_sequence_regression
from models.utils import masked_mean
from trainer.ppo_utils.experience_maker import Experience
from utils.utils import get_tokenizer


from ..ppo_utils.replay_buffer import NaiveReplayBuffer
from .launcher import BasePPORole




from torch import nn
import torch.optim as optim
from models.actor import Actor




class CriticPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        critic: torch.nn.Module,
        critic_optim: Optimizer,
        critic_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        value_clip: float = 0.2,
        dataloader_pin_memory: bool = True,
        **kwargs,
    ):
        self.strategy = None
        self.args = strategy
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler
        self.micro_train_batch_size = micro_train_batch_size
        self.buffer_limit = buffer_limit
        self.buffer_cpu_offload = buffer_cpu_offload
        self.value_clip = value_clip
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_epochs = self.args.max_epochs

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, getattr(self.args, "packing_samples", False)
        )

        self.critic_loss_fn = ValueLoss(value_clip)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)
                # experience.to_device("cpu")

                # for DP
                # status = self.strategy.all_reduce(status)

                status_list.append(status)
                pbar.set_postfix(status)
            
            import pdb; pdb.set_trace()

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        action_mask = experience.action_mask
        packed_seq_lens = None
        attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=None,
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        # print(f"type model: {type(self.critic)}")
        # print(self.critic)
        # import pdb; pdb.set_trace()
        # print(f"type model 1: {type(self.critic.model)}")
        # print(f"type model 2: {type(self.critic.model.model)}")
        self.backward(loss, self.critic, self.critic_optim)
        self.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.detach().item(),
            "values": masked_mean(values, experience.action_mask).detach().item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status
    
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        # model.backward(loss)
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        # model.step()
        optimizer.step()


class CriticModelRayActor():
    def __init__(self, strategy_args, pretrain, max_steps):
        args = strategy_args

        # self._setup_distributed(strategy)
        self.critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=None,
            value_head_prefix=args.value_head_prefix,
            init_value_head=args.pretrain == args.critic_pretrain,
            packing_samples=args.packing_samples,
            device_map="auto",
        )
        print(self.critic)
        print("reward normalization status: {}".format(args.normalize_reward))
        print("mean: {}, std {}".format(self.critic.mean, self.critic.std))

        # configure tokenizer
        if args.save_value_network:
            self.tokenizer = get_tokenizer(
                pretrain, self.critic, "left", None, use_fast=not args.disable_fast_tokenizer
            )

        # configure optimizer
        self.critic_optim = torch.optim.AdamW(
            self.critic.parameters(), lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        # configure scheduler
        self.critic_scheduler = get_scheduler(
            "cosine_with_min_lr",
            self.critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )

        # if args.gradient_checkpointing:
        #     critic.gradient_checkpointing_enable(
        #         gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        #     )

        # prepare models/optimizers...
        # self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare(
        #     (critic, critic_optim, critic_scheduler),
        #     is_rlhf=True,
        # )

        # load checkpoint
        # if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        #     ckpt_path = os.path.join(args.ckpt_path, "_critic")
        #     strategy.print(f"Loading the checkpoint: {ckpt_path}")
        #     strategy.load_ckpt(self.critic, ckpt_path)

        # initial offload
        # if strategy.args.deepspeed_enable_sleep:
        #     self.offload_states()

        # configure Trainer
        self.trainer = CriticPPOTrainer(
            args,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            value_clip=args.value_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        # print("Put all tensors to device :", device)
        # print(f"seq device: {sequences.device}")
        # print(f"seq after device: {sequences.to(device).device}")
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=False,
                values_allgather=True,
            )
        self.critic.train()  # reset model state
        return value.to("cpu")

    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)

    def fit(self):
        """Train critic model with the replay buffer."""
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def _save_model(self):
        args = self.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.critic,
            self.tokenizer,
            args.save_path + "_critic",
        )

    def _save_checkpoint(self, tag):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
        )

    # def reload_states(self):
    #     reload_deepspeed_states(self.critic)

    # def offload_states(self):
    #     offload_deepspeed_states(self.critic)
    
    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
