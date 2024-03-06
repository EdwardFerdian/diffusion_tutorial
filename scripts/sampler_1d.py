from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from a diffusion model')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing the trained model')
    args = parser.parse_args()

    model_dir = args.model_dir

    # get all model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    # get the latest one
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    last_model = model_files[-1]

    batch_size = 16
    total_samples = 16

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    ).cuda()

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 480,
        timesteps = 1000,
        sampling_timesteps = 250,
    ).cuda()


    trainer = Trainer1D(
            diffusion,
            input_filepath = None,
            results_folder = model_dir,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_id = int(last_model.split("-")[1].split(".")[0])

    print("Loading model with id: ", model_id)

    trainer.load(model_id)
    trainer.ema.to(device)
    trainer.ema.ema_model.eval()

    with torch.no_grad():
        # generate samples
        for i in range(total_samples//batch_size):
            sampled_images = trainer.ema.ema_model.sample(batch_size=batch_size)

            samples = sampled_images.cpu().detach().numpy() # b, c, seq_length

            # squeeze channel dimension
            samples = samples.squeeze(1)

            nrow = ncol = np.sqrt(len(samples)).astype(int)
            
            
            # show samples in a grid
            fig, axs = plt.subplots(nrow, ncol)
            for i in range(nrow):
                for j in range(ncol):
                    axs[i, j].plot(samples[i * ncol + j])
            plt.show()
