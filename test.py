import torch

initial_world = torch.tensor(
    [
        [0, 0],
        [1, 0],
        [1, 1]
    ]
)

new = torch.tensor([1, 1, 1]).reshape(1)

print(initial_world + new)

