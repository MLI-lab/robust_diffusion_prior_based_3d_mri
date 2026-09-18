from typing import Any, Dict, Optional
import logging

import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from src.utils.wandb_utils import tensor_to_wandbimages_dict

#from .loss import epsilon_based_loss_fn
from .loss import loss_fn_resolver
from ..sde import SDE
from ..utils_save import save_model

from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo

from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.ema import ExponentialMovingAverage

from src.diffmodels.sampler.base_sampler import BaseSampler
from src.sample_logger.base_sample_logger import BaseSampleLogger
from src.representations.fixed_grid_representation import FixedGridRepresentation

def score_model_trainer(
    score: UNetModel,
    sde: SDE,
    dataloader_train: DataLoader,
    dataloader_val: Optional[DataLoader],
    optim_kwargs: Dict,
    val_kwargs: Dict,
    prior_trafo: BasePriorTrafo,
    sampler : BaseSampler,
    sample_logger : BaseSampleLogger,
    switch_dir_and_upload_directory_on_exit_mgr,
    device: Optional[Any] = None,
    ):
    
    optimizer = Adam(score.parameters(), lr=optim_kwargs['lr'])
    #loss_fn = epsilon_based_loss_fn 
    loss_fn_train = loss_fn_resolver(**optim_kwargs['loss_fn'])
    loss_fn_val = loss_fn_resolver(**val_kwargs['loss_fn']) if dataloader_val is not None else None

    ema = None
    if optim_kwargs.use_ema: 
        ema = ExponentialMovingAverage(
            score.parameters(),
            decay=optim_kwargs['ema_decay']
            )

    batch_size = optim_kwargs['batch_size']

    log_cfg = optim_kwargs["log_dataset_stats_before_training"]
    if log_cfg["enabled"]:
        num_samples = len(dataloader_train) * batch_size if log_cfg["num_dataloader_stat_samples"] < 0 else log_cfg["num_dataloader_stat_samples"]
        num_images = len(dataloader_train) * batch_size if log_cfg["num_dataloader_image_samples"] < 0 else log_cfg["num_dataloader_image_samples"]
        samples_mean = torch.zeros(num_samples)
        samples_std = torch.zeros(num_samples)
        samples_norm = torch.zeros(num_samples)

        with tqdm(enumerate(dataloader_train), total=len(dataloader_train)) as pbar:
            for i, x in pbar:
                if i < num_images:
                    wandb.log({
                        'global_step': i,
                        'step' : i,
                        **tensor_to_wandbimages_dict(f"data_samples_{i}", x, show_phase=False)
                    })

                if i < num_samples:
                    if x.shape[0] != batch_size:
                        continue
                    x = x.view(batch_size,-1)
                    samples_mean[i*batch_size:(i+1)*batch_size] = x.mean(dim=-1)
                    samples_std[i*batch_size:(i+1)*batch_size] = x.std(dim=-1)
                    samples_norm[i*batch_size:(i+1)*batch_size] = torch.linalg.norm(x, dim=-1)

                if i > num_images and i > num_samples:
                    break
        
        wandb.run.summary.update({
            'samples_count': len(dataloader_train),
            'sample_mean_mean': samples_mean.mean(),
            'sample_mean_std': samples_mean.std(),
            'sample_std_mean': samples_std.mean(),
            'sample_std_std': samples_std.std(),
            'sample_norm_mean': samples_norm.mean(),
            'sample_norm_std': samples_norm.std(),
            })

    
    with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="training_run"):
        sample_logger.init_run()

    grad_step = 0
    steps_per_epoch = max(len(dataloader_train), 1)
    epochs_cfg = int(optim_kwargs.get('epochs', 1))

    training_steps_cfg = optim_kwargs.get('training_steps', None)
    if training_steps_cfg is None:
        max_steps = epochs_cfg * steps_per_epoch
    else:
        max_steps = int(training_steps_cfg)

    n_log_total = int(optim_kwargs.get('n_log_total', 10))
    log_every_n_steps = max(max_steps // n_log_total, 1)

    n_save_total = int(optim_kwargs.get('n_save_total', 10))
    save_every_n_steps = max(max_steps // n_save_total, 1)

    n_eval_total = int(val_kwargs.get('n_eval_total', 10))
    eval_every_n_steps = max(max_steps // n_eval_total, 1)

    n_sample_total = int(val_kwargs.get('n_sample_total', 10))
    sample_every_n_steps = max(max_steps // n_sample_total, 1)

    train_iter = iter(dataloader_train)
    running_loss = 0.0
    running_items = 0
    _shape_logged = False
    gradient_accumulation_steps = max(int(optim_kwargs.get('gradient_accumulation_steps', 1)), 1)

    def eval_on_validation_set(model):
        if dataloader_val is None or loss_fn_val is None:
            return None
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_num_items = 0
            num_max_items = 1000 # todo: make configurable
            for x in dataloader_val:
                x = x.to(device)
                x = prior_trafo(x)
                loss = loss_fn_val(
                    x=x,
                    model=model,
                    sde=sde
                )
                val_loss += loss.item() * x.shape[0]
                val_num_items += x.shape[0]
                if val_num_items >= num_max_items:
                    break
            return val_loss / max(val_num_items, 1)

    with tqdm(total=max_steps, desc="train_steps") as pbar:
        while grad_step < max_steps:
            score.train()
            optimizer.zero_grad()
            latest_loss_value = None
            for _ in range(gradient_accumulation_steps):
                try:
                    x = next(train_iter)
                except StopIteration:
                    train_iter = iter(dataloader_train)
                    x = next(train_iter)

                x = x.to(device)
                if not _shape_logged:
                    logging.info("train batch incoming shape: %s", tuple(x.shape))
                x = prior_trafo(x)
                if not _shape_logged:
                    logging.info("train batch after prior_trafo shape: %s", tuple(x.shape))
                    _shape_logged = True

                loss = loss_fn_train(
                    x=x,
                    model=score,
                    sde=sde
                )

                latest_loss_value = loss.item()
                running_loss += latest_loss_value * x.shape[0]
                running_items += x.shape[0]
                (loss / gradient_accumulation_steps).backward()

            optimizer.step()

            grad_step += 1
            pbar.update(1)
            if latest_loss_value is not None:
                pbar.set_description(f"loss={latest_loss_value:.3f}", refresh=False)

            if optim_kwargs.use_ema and grad_step > int(optim_kwargs['ema_warm_start_steps']):
                ema.update(score.parameters())

            should_log = (grad_step % log_every_n_steps == 0) or (grad_step == max_steps)
            if should_log and wandb.run is not None:
                wandb.log({
                    'train_loss': running_loss / max(running_items, 1),
                    'global_step': grad_step,
                    'step': grad_step,
                    'epoch_equivalent': grad_step / steps_per_epoch,
                })
                running_loss = 0.0
                running_items = 0

            should_eval = (grad_step % eval_every_n_steps == 0) or (grad_step == max_steps)
            if should_eval and dataloader_val is not None:
                val_loss = eval_on_validation_set(score)
                if val_loss is not None and wandb.run is not None:
                    wandb.log({
                        'val_loss': val_loss,
                        'global_step': grad_step,
                        'step': grad_step,
                        'epoch_equivalent': grad_step / steps_per_epoch,
                    })

            should_sample = (grad_step % sample_every_n_steps == 0) or (grad_step == max_steps)
            if should_sample:
                if optim_kwargs.use_ema:
                    ema.store(score.parameters())
                    ema.copy_to(score.parameters())
                    score = score.to(device)
                score.eval()

                # Release fragmented reserved-but-unallocated CUDA memory before
                # sampling so the UNet forward passes have enough contiguous space.
                torch.cuda.empty_cache()

                sample_logger.init_sample_log(sample_nr=grad_step, mesh=None)
                sample = sampler.sample()
                representation = FixedGridRepresentation(
                    in_shape=tuple(sample.shape[:-1]), out_features=sample.shape[-1], warm_start=sample
                )
                sample_logger.close_sample_log(representation=representation)

                val_loss_ema = eval_on_validation_set(score)
                if wandb.run is not None:
                    log_payload = {
                        'sample_mean': sample.mean(),
                        'sample_std': sample.std(),
                        'global_step': grad_step,
                        'step': grad_step,
                    }
                    if val_loss_ema is not None:
                        log_payload['val_loss_ema'] = val_loss_ema
                    wandb.log(log_payload)

                if optim_kwargs.use_ema:
                    ema.restore(score.parameters())

            should_save_model = (grad_step % save_every_n_steps == 0) or (grad_step == max_steps)
            if should_save_model:
                with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder=f"step_{grad_step}"):
                    save_model(score=score, epoch=grad_step, optim_kwargs=optim_kwargs, ema=ema)

    with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="final_model"):
        save_model(score=score, epoch=max_steps, optim_kwargs=optim_kwargs, ema=ema)
        sample_logger.close_run()
