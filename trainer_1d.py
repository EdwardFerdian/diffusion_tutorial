from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a diffusion model')
    parser.add_argument('--input-file', type=str, required=True, help='Input .h5 file containing the training data with column "input"')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--seq-length', type=int, default=480, help='Length of the sequence')
    args = parser.parse_args()

    input_filepath = args.input_file
    model_dir = args.output_dir
    seq_length = args.seq_length

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    ).cuda()

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_length,
        timesteps = 1000,
        sampling_timesteps = 250,
    ).cuda()

    trainer = Trainer1D(
        diffusion,
        input_filepath,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 10000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        results_folder = model_dir,
        save_and_sample_every = 1000,
    )

    trainer.train()


