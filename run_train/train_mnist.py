import logging
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
import numpy as np
import sys

print(sys.path)
sys.path.append(r"/home/motterbe/CataSynth")

from lib.model.ddim import DDIM
from lib.model.emnist_cond_unet import ConditionalEMNISTUNet
from lib.losses.losses import noise_estimation_loss
from lib.utils.factory import get_MNIST_data, get_optimizer, get_scheduler
from lib.utils.ema import EMAHelper
from lib.utils.logging import save_checkpoint, save_examples
from lib.utils.pre_train import parse_args, get_configs, get_log_path, dump_configs


#
# Pre-train
#
print("##### Pre-train")

args = parse_args()

data_conf, model_conf, diffusion_conf, train_conf = get_configs(args)

DEV = args.device_list[0]

LOG_PATH = get_log_path(args)

dump_configs(data_conf, model_conf, diffusion_conf, train_conf, LOG_PATH)

EMA = model_conf['EMA']
EPOCHS = train_conf['EPOCHS']
DATA_SHAPE = eval(data_conf['SHAPE'])

writer = SummaryWriter(log_dir=LOG_PATH)

print(f"Avail. GPUs: ", torch.cuda.device_count())

#
# Data
#
print("##### Loading data.")

train_ds, train_dl, val_ds, val_dl = get_MNIST_data(args, data_conf, train_conf, "/home/motterbe/repos/denoising-diffusion-pytorch/datasets/emnist/EMNIST/raw/dl")

#
# Model etc.
#
print("##### Loading model etc.")

diffusion = DDIM(diffusion_config=diffusion_conf, model_config=model_conf, device=DEV)
m = ConditionalEMNISTUNet(data_config=data_conf, model_config=model_conf, diffusion_config=diffusion_conf,
                    num_classes=10).to(DEV)  # w/o ignore label
# m = torch.nn.DataParallel(m, device_ids=args.device_list)
optim = get_optimizer(m.parameters(), train_conf)
sched = get_scheduler(optim, train_conf)
# scaler = GradScaler()
if EMA:
    ema_helper = EMAHelper(mu=model_conf['EMA_RATE'])
    ema_helper.register(m)
else:
    ema_helper = None
kid = KernelInceptionDistance(subset_size=train_conf['VAL_SAMPLES'] // 4).to(DEV)

#
# Training loop
#
print("##### Training")

best_kid = 1e6
start_epoch, step = 0, 0
for epoch in range(EPOCHS):

    # Training
    m.train()

    loss_sum = 0.

    for i, ((sample, bg_rad), label) in enumerate(tqdm(train_dl)):

        bg_rad = bg_rad.to(DEV)
        N = sample.shape[0]
        step += 1
        sample = sample.to(DEV)
        label = label.to(DEV)
        # phase_label = phase_label.long().to(DEV)
        # semantic_label = semantic_label.float().to(DEV)

        e = torch.randn_like(sample)
        b = diffusion.betas

        # Antithetic sampling
        t = torch.randint(low=0, high=diffusion.num_timesteps, size=(N // 2 + 1,)).to(DEV)
        t = torch.cat([t, diffusion.num_timesteps - t - 1], dim=0)[:N]

        # Classifier-free guidance
        if np.random.random() < diffusion_conf['GUIDANCE_P_UNCOND']:
            phase_label = None
        if np.random.random() < diffusion_conf['GUIDANCE_P_UNCOND']:
            bg_rad = None

        optim.zero_grad()


        loss = noise_estimation_loss(model=m, x0=sample, t=t, e=e, b=b,
                                                 label=label, current_color_cycle=bg_rad)

        loss_sum += loss.item()

        loss.backward()
        # scaler.scale(loss).backward()

        try:
            torch.nn.utils.clip_grad_norm_(
                m.parameters(), train_conf['CLIP_GRAD_NORM']
            )
        except Exception:
            pass

        optim.step()
        # scaler.step(optim)
        # scaler.update()

        if EMA:
            ema_helper.update(m)

    writer.add_scalar(tag='train/avg_loss', scalar_value=loss_sum / i, global_step=epoch)
    writer.add_scalar(tag='train/lr', scalar_value=optim.param_groups[0]['lr'], global_step=epoch)
    train_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                    f"\tEpoch {epoch}|{EPOCHS}" \
                    f"\tAvg. loss: {loss_sum / i}" \
                    f"\tLR: {optim.param_groups[0]['lr']}"
    print(train_log_str)
    logging.info(msg=train_log_str)

    # Validation
    m.eval()
    if not epoch % train_conf['VAL_FREQ']:

        assert train_conf['VAL_SAMPLES'] % 3 == 0

        with torch.no_grad():
            x = torch.randn(
                train_conf['VAL_SAMPLES'],
                *DATA_SHAPE,
                device=DEV,
            )

            # Idle phase
            # target_phase_label = torch.zeros(size=(train_conf['VAL_SAMPLES'], 1)).long().to(DEV)

            # target_semantic_label = torch.zeros(size=(train_conf['VAL_SAMPLES'],
            #                                           train_ds.num_tool_classes)).to(DEV)

            #TODO: CONFIG NUM CLASSES
            ((real_sample, real_bg_rad), real_label) = next(iter(val_dl))

            real_bg_rad = real_bg_rad.to(DEV)
            target_label = real_label.to(DEV)
            # target_label = torch.randint(0,10,size=(train_conf['VAL_SAMPLES'], 1)).long().to(DEV)

            gen_sample = diffusion.sample_image(x=x, model=m,
                                                guidance=True,
                                                guidance_strength=diffusion_conf['GUIDANCE_STRENGTH'],
                                                mask=None,
                                                label=target_label,
                                                current_color_cycle=real_bg_rad)

            _gen_sample = (gen_sample + 1.0) * 0.5
            _gen_sample = (_gen_sample * 255).type(torch.uint8).to(DEV)

            _real_sample = (real_sample + 1.0) * 0.5
            _real_sample = (_real_sample * 255).type(torch.uint8).to(DEV)

            kid.update(_real_sample, real=True)
            kid.update(_gen_sample, real=False)

            kid_mean, kid_std = kid.compute()
            writer.add_scalar(tag='val/KID mean', scalar_value=kid_mean, global_step=epoch)
            writer.add_scalar(tag='val/KID std', scalar_value=kid_std, global_step=epoch)
            val_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                          f"\tKID mean: {kid_mean}" \
                          f"\tKID std: {kid_std}"
            print(val_log_str)
            logging.info(msg=val_log_str)

            if kid_mean < best_kid:
                best_kid = kid_mean

                #TODO: CONFIGURE OUTPUT SIZE
                save_examples([gen_sample, real_sample],
                              target_shape=np.array((3,32,32)),#//np.array([1, 4, 4]),
                              log_path=LOG_PATH,
                              epoch=epoch,
                              names=['gen', 'real'],
                              device=DEV)

                states = [m.state_dict(), optim.state_dict(), sched.state_dict()]
                if EMA:
                    states.append(ema_helper.state_dict())

                save_checkpoint(states, LOG_PATH)

    # End of epoch
    if type(sched) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        sched.step(kid_mean)
    else:
        sched.step()
