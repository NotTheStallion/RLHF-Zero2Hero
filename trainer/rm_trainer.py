import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from models.loss import LogExpLoss, PairWiseLoss



class RewardModelTrainer(ABC):
    """
    Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        strategy_args,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy_args
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy_args

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            # self.strategy.print("LogSigmoid Loss")
            print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            # self.strategy.print("LogExp Loss")
            print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = self.args.packing_samples

        self.margin_loss = self.args.margin_loss
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.args.use_wandb : #  and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.args.use_wandb)
            wandb.init(
                entity=self.args.wandb_org,
                project=self.args.wandb_project,
                group=self.args.wandb_group,
                name=self.args.wandb_run_name,
                config=self.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        # if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
        #     from torch.utils.tensorboard import SummaryWriter

        #     os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
        #     log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
        #     self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = float("inf")  # Disable evaluation if not specified
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step
        accumulated_gradient = args.train_batch_size // args.micro_train_batch_size
        step = consumed_samples // args.train_batch_size * accumulated_gradient + 1

        epoch_bar = tqdm(range(self.epochs), desc="Train epoch")
        acc_sum = 0
        loss_sum = 0
        
        for epoch in range(self.epochs):
            # if isinstance(self.train_dataloader.sampler, DistributedSampler):
            #     self.train_dataloader.sampler.set_epoch(epoch)

            # train
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
            )
            loss_sum = 0
            acc_sum = 0
            self.model.train()
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                
                # Standard backward pass and optimizer step
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                
                # logging info
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                # print(f"step: {step}, accumulated_gradient: {accumulated_gradient}, loss_sum: {loss_sum}, acc_sum: {acc_sum}")
                # if step % accumulated_gradient == 0:
                #     logs_dict["loss_mean"] = loss_sum / accumulated_gradient
                #     logs_dict["acc_mean"] = acc_sum / accumulated_gradient
                #     loss_sum = 0
                #     acc_sum = 0
                #     global_step = step // accumulated_gradient
                #     client_states = {"consumed_samples": global_step * args.train_batch_size}
                # self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                    
                if self._wandb is not None:
                    self._wandb.log({"loss": logs_dict["loss"]})
                    self._wandb.log({"acc": logs_dict["acc"]})
                    self._wandb.log({"chosen_reward": logs_dict["chosen_reward"]})
                    self._wandb.log({"reject_reward": logs_dict["reject_reward"]})
                    self._wandb.log({"lr": logs_dict["lr"]})
                # print(logs_dict["loss"])

                step += 1
            epoch_bar.update()
            if self._wandb is not None:
                self._wandb.log({"loss_mean": loss_sum / len(self.train_dataloader)})
                self._wandb.log({"acc_mean": acc_sum / len(self.train_dataloader)})
                self.evaluate(self.eval_dataloader, epoch)
            # print(f"loss_sum: {loss_sum}, len(self.train_dataloader): {len(self.train_dataloader)}")
            

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            # elif self._tensorboard is not None and self.strategy.is_rank_0():
            #     for k, v in logs_dict.items():
            #         self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if (
            global_step % args.eval_steps == 0 # or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        # if global_step % args.save_steps == 0:
        #     tag = f"global_step{global_step}"
        #     if not self.disable_ds_ckpt:
        #         self.strategy.save_ckpt(
        #             self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
        #         )
        #     if self.save_hf_ckpt:
        #         save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
        #         self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_reward, reject_reward, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                loss = self.loss_fn(chosen_reward, reject_reward, margin)

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                # print("="*20)
                # print((chosen_reward > reject_reward))
                # print((chosen_reward > reject_reward).float().mean().item())
                # import pdb; pdb.set_trace()
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()

            # print("Eval acc: ", acc)
            acc_mean = acc / eval_dataloader.__len__()
            loss_mean = loss_sum / eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            # rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            # print("Set reward mean std")
            # unwrap_model = self.strategy._unwrap_model(self.model)
            # unwrap_model.config.mean = reward_mean.item()
            # unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = bar_dict
            # logs = self.strategy.all_reduce(bar_dict)
            # step_bar.set_postfix(logs)

            # histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            # self.strategy.print("histgram")
            # self.strategy.print(histgram)

            # if self.strategy.is_rank_0():
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                print(logs)
                self._wandb.log(logs)
                # elif self._tensorboard is not None:
                #     for k, v in logs.items():
                #         self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # # Determine the device of the model
        # model_device = next(model.parameters()).device

        # # Check the device of the inputs
        # chosen_ids_device = chosen_ids.device
        # c_mask_device = c_mask.device
        # reject_ids_device = reject_ids.device
        # r_mask_device = r_mask.device

        # # Print or log the devices if needed
        # print(f"Model is on device: {model_device}")
        # print(f"chosen_ids is on device: {chosen_ids_device}")
        # print(f"c_mask is on device: {c_mask_device}")
        # print(f"reject_ids is on device: {reject_ids_device}")
        # print(f"r_mask is on device: {r_mask_device}")
        
        
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
