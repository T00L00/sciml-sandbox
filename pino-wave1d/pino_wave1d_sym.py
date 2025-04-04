from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import physicsnemo.sym
from physicsnemo.models.layers.spectral_layers import fourier_derivatives
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter
from physicsnemo.sym.utils.io.vtk import grid_to_vtk

from utilities import download_FNO_dataset, load_FNO_dataset, preprocess_FNO_mat
from ops import dx, ddx

class WavePDE(torch.nn.Module):

    def __init__(self, c: float=1.):
        super().__init()
        self.c = c
    
    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        u_tt = input_var["u__x__x"]
        u_xx = input_var["u__y__y"]

        residual = u_tt - (self.c**2) * u_xx
        output_var = {"wave_pde": residual}
        return output_var

@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:

    # Define keys
    input_keys = [Key("a")]
    output_keys = [Key("u")]

    # Load data (if any)

    # Create datasets

    # Instantiate FNO architecture
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys
    )

    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net
    )

    derivatives = [
        Key("u", derivatives=[Key("x"), Key("x")]),
        Key("u", derivatives==[Key("y"), Key("y")])
    ]
    fno.add_pino_gradients(
        derivatives=derivatives,
        domain_length=[1., 1.]
    )

    # Define custom node for computing pde loss from wave equation

    inputs = ["u__x__x", "u__y__y"]
    wave_pde_node = Node(
        inputs=inputs,
        outputs=["wave_pde"],
        evaluate=WavePDE(c=1.), # The predefined PDE class that computes residual in forward method
        name="Wave PDE Node"
    )

    nodes = [fno.make_node(), wave_pde_node]

    # Add supervised grid constraint to ensure training on data
    supervised_constraint = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid
    )

    # Add grid validator
    validator = GridValidator(
        nodes=nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True
    )

    # Add everything to domain
    domain = Domain()
    domain.add_constraint(supervised_constraint, "supervised")
    domain.add_validator(validator, "test")

    # Create and run solver
    solver = Solver(cfg, domain)
    solver.solve()

if __name__ == "__main__":
    run()