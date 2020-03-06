import torch

x = torch.randn(3, requires_grad=True)
y = torch.randn(3, requires_grad=True)
z = x * y
z = z.relu() + y

print(z.grad_fn.next_functions)

z.backward(torch.ones(3))

print(x.grad)
