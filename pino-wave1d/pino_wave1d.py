import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.pdes.wave_equation import WaveEquation  # Assumed available for 1D wave
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils import HDF5MapStyleDataset

def validation_step(model, dataloader, epoch):
    """Run validation and save comparison plots."""
    model.eval()
    with torch.no_grad():
        loss_epoch = 0.0
        for data in dataloader:
            invar, outvar, _, _ = data
            # For prediction, assume invar contains the initial condition in channel 0.
            pred = model(invar[:, 0].unsqueeze(dim=1))
            loss_epoch += F.mse_loss(outvar, pred)
        
        # Convert to numpy arrays for plotting (using the last batch)
        true_field = outvar.detach().cpu().numpy()
        pred_field = pred.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(25, 5))
        d_min = np.min(true_field[0, 0])
        d_max = np.max(true_field[0, 0])
        
        im0 = ax[0].imshow(true_field[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(pred_field[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(np.abs(true_field[0, 0] - pred_field[0, 0]))
        plt.colorbar(im2, ax=ax[2])
        
        ax[0].set_title("True")
        ax[1].set_title("Predicted")
        ax[2].set_title("Difference")
        
        fig.savefig(f"results_wave_epoch_{epoch}.png")
        plt.close()
        return loss_epoch / len(dataloader)

@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino_wave.yaml")
def main(cfg: DictConfig):

    # Set up device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LaunchLogger.initialize()

    # Define the 1D wave equation (with wave speed c=1.0, for example)
    wave_eq = WaveEquation(c=1.0, dim=1)
    
    # Load training and validation datasets.
    # (Assuming HDF5 datasets where the input field is the wave at t=0 and
    #  the output field is the wave at later times.)
    train_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Wave_1D/train.hdf5"), device=device
    )
    val_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Wave_1D/validation.hdf5"), device=device
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size.train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size.validation, shuffle=False)

    # Instantiate a Fourier Neural Operator (FNO) for 1D.
    model = FNO(
        in_channels=cfg.model.fno.in_channels,    # e.g., 1 for initial wave field
        out_channels=cfg.model.fno.out_channels,  # e.g., 1 for predicted wave field
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=1,                              # 1D problem
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(device)

    # Create a PhysicsInformer to enforce the 1D wave equation.
    # It will compute the PDE residual using a finite-difference method.
    phy_informer = PhysicsInformer(
        required_outputs=["wave_equation"],
        equations=wave_eq,
        grad_method="finite_difference",
        device=device,
        fd_dx=1 / cfg.dataset.spatial_resolution,  # e.g. resolution of the spatial grid
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.start_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # Training loop
    for epoch in range(cfg.max_epochs):
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_loader), epoch_alert_freq=10
        ) as log:
            for data in train_loader:
                optimizer.zero_grad()
                invar = data[0]  # input: wave at t=0, shape [batch, channels, ...]
                outvar = data[1] # ground-truth wave field at later times

                # Forward pass: predict wave field at later times.
                # Here we assume the network input is the initial condition.
                pred = model(invar[:, 0].unsqueeze(dim=1))
                
                # Compute the physics-informed residual.
                # The physics informer will use finite differences on the predicted field
                # to evaluate the wave equation residual.
                residuals = phy_informer.forward({"u": pred})
                # Extract the residual associated with the wave equation.
                pde_residual = residuals["wave_equation"]
                # For 1D, pad along the spatial dimension if necessary.
                pde_residual = F.pad(pde_residual[:, :, 2:-2], [2, 2], "constant", 0)
                loss_pde = F.l1_loss(pde_residual, torch.zeros_like(pde_residual))
                
                # Compute data loss between predicted and true wave field.
                loss_data = F.mse_loss(outvar, pred)
                
                # Total loss: data loss plus a physics loss weighted by a factor.
                loss = loss_data + (1 / cfg.dataset.spatial_resolution) * cfg.physics_weight * loss_pde
                
                # Backpropagation and optimization step.
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                log.log_minibatch({
                    "loss_data": loss_data.detach(),
                    "loss_pde": loss_pde.detach()
                })
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Run validation and log the error.
        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, val_loader, epoch)
            log.log_epoch({"Validation error": error})

        # Save a checkpoint at the end of each epoch.
        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

if __name__ == "__main__":
    main()
