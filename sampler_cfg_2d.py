from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion, Trainer
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from a classifier free guidance 2D')
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

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        num_classes = num_classes,
        cond_drop_prob = 0,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,           
        sampling_timesteps = 250,   
    ).cuda()


    trainer = Trainer(
            diffusion,
            folder = None,
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

    with torch.no_grad():
        # generate samples
        for i in range(total_samples//batch_size):
            sampled_images = trainer.ema.ema_model.sample(classes=classes_tensor)
            samples = sampled_images.cpu().detach().numpy() # (b, c, w, h)
            # permute
            samples = np.transpose(samples, (0, 2, 3, 1))

            # show samples in a grid
            nrow = ncol = np.sqrt(len(samples)).astype(int)
            fig, axs = plt.subplots(nrow, ncol)
            for i in range(nrow):
                for j in range(ncol):
                    axs[i, j].imshow(samples[i*nrow+j])
                    axs[i, j].axis('off')
                    axs[i, j].set_title(classes[i*nrow+j])
            plt.show()
