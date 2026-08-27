import torch


def get_parameters(model):
    """
    Convert the PyTorch model state_dict into a list
    of NumPy arrays that can be exchanged through Flower.
    """
    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(model, parameters):
    """
    Load parameters received from the federated server
    into the PyTorch model.
    """

    state_dict = model.state_dict()

    new_state_dict = {}

    for (key, old_value), new_value in zip(
        state_dict.items(),
        parameters
    ):
        tensor = torch.tensor(
            new_value,
            dtype=old_value.dtype
        )

        new_state_dict[key] = tensor

    model.load_state_dict(new_state_dict, strict=True)