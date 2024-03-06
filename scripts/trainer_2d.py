from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a diffusion model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the training images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in the input image')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.output_dir
    channel = args.channels

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = channel # 3 for color images, 1 for grayscale images
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 1000,           
        sampling_timesteps = 250,   
    ).cuda()

    trainer = Trainer(
        diffusion,
        data_dir,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 50000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        results_folder = model_dir,
        num_samples=16,
        calculate_fid=False,
        save_and_sample_every = 1000,
    )

    trainer.train()
