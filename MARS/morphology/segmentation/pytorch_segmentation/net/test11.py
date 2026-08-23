import torch
from torch import nn

batch_size = 1
c, h, w = 3, 10, 10
nb_classes = 5

x = torch.randn(batch_size, c, h, w)
target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)

model = nn.Sequential(
    nn.Conv2d(c, 6, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(6, nb_classes, 3, 1, 1)
)

criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, target)
loss.backward()
