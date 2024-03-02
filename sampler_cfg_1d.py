from denoising_diffusion_pytorch.classifier_free_guidance_1d import Unet1D, GaussianDiffusion, Trainer1D
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from a classifier free guidance 1D')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing the trained model')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in the dataset')
    args = parser.parse_args()

    model_dir = args.model_dir
    num_classes = args.num_classes

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
        num_classes = num_classes,
        cond_drop_prob = 0.,
        channels = 1
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        seq_length = 480,
        timesteps = 1000,
        sampling_timesteps = 250,
    ).cuda()


    trainer = Trainer1D(
            diffusion,
            input_filepath = None,
            num_classes=num_classes,
            results_folder = model_dir,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_id = int(last_model.split("-")[1].split(".")[0])
    print("Loading model with id: ", model_id)

    trainer.load(model_id)
    trainer.ema.to(device)
    trainer.ema.ema_model.eval()

        
    # generate random classes as much as batch
    classes = np.random.randint(0, num_classes, batch_size)
    classes_tensor = torch.tensor(classes).to(device)
        
    nr_iter = total_samples//batch_size
    with torch.no_grad():
        # generate samples
        for i in range(nr_iter):
            print(f"\nProcessing batch {i+1}/{nr_iter}")
            sampled_images = trainer.ema.ema_model.sample(classes=classes_tensor)

            samples = sampled_images.cpu().detach().numpy()
            # squeeze dim 1
            samples = np.squeeze(samples, axis=1)
            
            nrow = ncol = np.sqrt(len(samples)).astype(int)
            # show samples in a grid
            fig, axs = plt.subplots(nrow, ncol)
            for i in range(nrow):
                for j in range(ncol):
                    axs[i, j].plot(samples[i * ncol + j])
                    axs[i, j].set_title(classes[i * ncol + j])
            plt.show()