import time
import jax
import jax.numpy as jnp

import dataset

layer_sizes = [2, 2, 1]
lr = 0.2
num_epochs = 10000


def init_random_params(layer_sizes, key, init="normal"):
    if init == "uniform":
        return [
            (jax.random.uniform(key, (m, n)), jax.random.uniform(key, (n,)))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
    elif init == "normal":
        return [
            (jax.random.normal(key, (m, n)), jax.random.normal(key, (n,)))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
    else:
        raise Exception("only uniform or normal initialization allowed")


def forward(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.sigmoid(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return jax.nn.sigmoid(logits)


def loss(params, x, y):
    preds = forward(params, x)
    return jnp.mean((preds - y) ** 2)


def accuracy(params, x, y):
    predicted_class = jnp.rint(forward(params, x))
    return jnp.mean(predicted_class == y)


@jax.jit
def update_params(params, x, y):
    grads = jax.grad(loss)(params, x, y)

    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]


dataset = dataset.XorDataSet()
# dataset = dataset.AndDataSet()

key = jax.random.PRNGKey(int(time.time()))
params = init_random_params(layer_sizes, key)
for w, b in params:
    print("w: ", w)
    print("b: ", b)

start_time = time.time()

key, _ = jax.random.split(key, 2)

# x, y = dataset.get_noisy_samples(num=4, key=key)

for epoch in range(num_epochs):
    x,y = dataset.get_samples()

    params = update_params(params, x, y)

    if epoch % 1000 == 0:
        print("predict:", forward(params, x))
        print("params", params)
        print("LOSS:", loss(params, x, y))
        epoch_time = time.time() - start_time
        print("Epoch {}, Training Time {:0.2f} sec".format(epoch, epoch_time))
        print("Accuracy {}\n".format(accuracy(params, x, y)))
