from denoising_diffusion_pytorch.classifier_free_guidance_1d import Unet1D, GaussianDiffusion, Trainer1D
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a diffusion model')
    parser.add_argument('--input-file', type=str, required=True, help='Directory containing the training images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--seq-length', type=int, default=480, help='Length of the sequence')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in the dataset')
    args = parser.parse_args()

    input_filepath = args.input_file
    model_dir = args.output_dir
    seq_length = args.seq_length
    num_classes = args.num_classes
    

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.1,
        channels = 1
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        seq_length = seq_length,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ).cuda()

    trainer = Trainer1D(
        diffusion,
        input_filepath,
        num_classes = num_classes,
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 10000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        results_folder = model_dir,
        save_and_sample_every = 1000,
    )

    trainer.train()
