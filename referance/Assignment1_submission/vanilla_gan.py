# Assignment 1 - GAN Playground
# vanilla_gan.py  --  DCGAN training loop
#
# Usage:
#   python vanilla_gan.py                          # baseline
#   python vanilla_gan.py --use_diffaug            # with DiffAugment
#   python vanilla_gan.py --data_preprocess vanilla

import argparse
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
import wandb

import utils
from data_loader import get_data_loader
from models import DCGenerator, DCDiscriminator
from diff_augment import DiffAugment

policy = 'color,translation,cutout'

SEED = 11
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_models(G, D):
    print("=" * 40, "Generator", "=" * 40)
    print(G)
    print("=" * 40, "Discriminator", "=" * 40)
    print(D)


def create_model(opts):
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
    D = DCDiscriminator(conv_dim=opts.conv_dim)

    print_models(G, D)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D


def create_image_grid(array, ncols=None):
    """Convert (N, C, H, W) array to an (H*nrows, W*ncols, C) grid."""
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels),
                      dtype=array.dtype)
    for i in range(nrows):
        for j in range(ncols):
            result[i * cell_h:(i + 1) * cell_h,
                   j * cell_w:(j + 1) * cell_w, :] = \
                array[i * ncols + j].transpose(1, 2, 0)
    return result.squeeze() if channels == 1 else result


def checkpoint(iteration, G, D, opts):
    torch.save(G.state_dict(),
               os.path.join(opts.checkpoint_dir, f'G_iter{iteration}.pkl'))
    torch.save(D.state_dict(),
               os.path.join(opts.checkpoint_dir, f'D_iter{iteration}.pkl'))




def to_uint8_grid(images):
    """Convert (N, C, H, W) tensor/array with values in [-1,1] to uint8 grid."""
    arr = utils.to_data(images)          # -> numpy (N, C, H, W)
    grid = create_image_grid(arr)        # -> (H, W, C) in [-1, 1]
    return np.uint8(255 * (grid + 1) / 2)


def save_samples(G, fixed_noise, iteration, opts):
    """Save a grid of generated images to disk and log it to W&B."""
    generated = G(fixed_noise)
    grid = to_uint8_grid(generated)
    path = os.path.join(opts.sample_dir, f'sample-{iteration:06d}.png')
    imageio.imwrite(path, grid)
    print(f'Saved {path}')

    wandb.log({'samples': wandb.Image(grid)}, step=iteration)


def sample_noise(batch_size, dim):
    """Uniform noise in [-1, 1], shape (batch_size, dim, 1, 1)."""
    return utils.to_var(
        torch.rand(batch_size, dim) * 2 - 1
    ).unsqueeze(2).unsqueeze(3)


# ---------------------------------------------------------------------------
# Image preparation helper
# ---------------------------------------------------------------------------

def prepare_images(images, opts):
    """Prepare images before they are passed to the discriminator.

    When --use_diffaug is set, apply the differentiable augmentation policy
    ('color,translation,cutout') to BOTH real and fake images.  Because the
    transforms are differentiable, gradients flow through them back to the
    generator during the G update, forcing G to produce samples that remain
    realistic under random color/translation/cutout perturbations.
    Otherwise, return the images unchanged.
    """
    if opts.use_diffaug:
        return DiffAugment(images, policy=policy)
    return images



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training_loop(train_dataloader, opts):

    G, D = create_model(opts)

    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)

    wandb.init(
        entity=opts.wandb_entity,
        project='assignment1-dcgan',
        name=f'dcgan_diffaug-{opts.use_diffaug}',
        config=vars(opts),
    )

    iteration = 1
    total_train_iters = opts.num_epochs * len(train_dataloader)

    for _ in range(opts.num_epochs):
        for batch in train_dataloader:

            real_images = utils.to_var(batch)

            # ==============================================================
            # TRAIN THE DISCRIMINATOR
            # ==============================================================

            # 1. Discriminator loss on real images: (D(x) - 1)^2
            real_images_processed = prepare_images(real_images, opts)

            D_real_loss = torch.mean((D(real_images_processed) - 1) ** 2)

            # 2. Sample a batch of noise vectors z.
            noise = sample_noise(real_images.size(0), opts.noise_size)

            # 3. Generate fake images G(z).
            fake_images = G(noise)

            # 4. Discriminator loss on fake images: (D(G(z)))^2
            # Note:
            # We detach fake_images so that gradients from the discriminator
            # update do not flow back into the generator parameters.

            fake_images_processed = prepare_images(fake_images.detach(), opts)

            D_fake_loss = torch.mean(D(fake_images_processed) ** 2)

            # 5. Total discriminator loss and update step.
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            d_optimizer.zero_grad()
            D_total_loss.backward()
            d_optimizer.step()

            # ==============================================================
            # TRAIN THE GENERATOR
            # ==============================================================

            # 1. Sample a fresh batch of noise vectors z.
            noise = sample_noise(real_images.size(0), opts.noise_size)

            # 2. Generate fake images G(z).
            fake_images = G(noise)

            # 3. Generator loss: (D(G(z)) - 1)^2
            fake_images_processed = prepare_images(fake_images, opts)

            G_loss = torch.mean((D(fake_images_processed) - 1) ** 2)

            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

            # ==============================================================
            # Logging
            # ==============================================================

            if iteration % opts.log_step == 0:
                print(
                    f'Iter [{iteration:4d}/{total_train_iters}] | '
                    f'D_real: {D_real_loss.item():.4f} | '
                    f'D_fake: {D_fake_loss.item():.4f} | '
                    f'G: {G_loss.item():.4f}'
                )

                wandb.log({
                    'D/real_loss':  D_real_loss.item(),
                    'D/fake_loss':  D_fake_loss.item(),
                    'D/total_loss': D_total_loss.item(),
                    'G/loss':       G_loss.item(),
                }, step=iteration)

            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)

            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1

    wandb.finish()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(opts):
    dataloader = get_data_loader(opts.data, opts)
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)
    training_loop(dataloader, opts)


def create_parser():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--conv_dim',   type=int, default=64)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training
    parser.add_argument('--num_epochs',       type=int,   default=500)
    parser.add_argument('--batch_size',       type=int,   default=16)
    parser.add_argument('--num_workers',      type=int,   default=2)
    parser.add_argument('--lr',               type=float, default=0.0002)
    parser.add_argument('--beta1',            type=float, default=0.5)
    parser.add_argument('--beta2',            type=float, default=0.999)

    # Data
    parser.add_argument('--data',            type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--data_preprocess', type=str, default='vanilla')
    parser.add_argument('--use_diffaug',     action='store_true')
    parser.add_argument('--ext',             type=str, default='*.png')

    # Directories / intervals
    parser.add_argument('--checkpoint_dir',   type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir',       type=str, default='./vanilla')
    parser.add_argument('--log_step',         type=int, default=10)
    parser.add_argument('--sample_every',     type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=400)

    # W&B
    parser.add_argument('--wandb_entity',     type=str, default='NN-Graphics')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    opts.sample_dir = os.path.join(
        'output/', opts.sample_dir,
        f'{os.path.basename(opts.data)}_{opts.data_preprocess}'
    )
    if opts.use_diffaug:
        opts.sample_dir += '_diffaug'

    if os.path.exists(opts.sample_dir):
        os.system(f'rm {opts.sample_dir}/*')

    print(opts)
    main(opts)
