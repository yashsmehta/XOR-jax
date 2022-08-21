import time
import jax
import jax.numpy as jnp
import optax

import dataset

layer_sizes = [2, 2, 1]
max_iter = 500


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
        # activations = jax.nn.sigmoid(outputs)
        activations = jax.nn.relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return jnp.squeeze(jax.nn.relu(logits))


def loss(params, x, y):
    preds = forward(params, x)
    return jnp.mean(optax.l2_loss(preds, y))


def accuracy(params, x, y):
    predicted_class = jnp.rint(forward(params, x))
    return jnp.mean(predicted_class == y)


# dataset = dataset.XorDataSet()
dataset = dataset.AndDataSet()

key = jax.random.PRNGKey(int(time.time()))
params = init_random_params(layer_sizes, key)
for w, b in params:
    print("w: ", w)
    print("b: ", b)

start_time = time.time()
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)

for iteration in range(max_iter):
    key, _ = jax.random.split(key, 2)
    # x,y = dataset.get_samples()
    x, y = dataset.get_noisy_samples(num=4, key=key)

    grads = jax.grad(loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if iteration % 100 == 0:
        print("predict:", forward(params, x))
        print("params", params)
        print("LOSS:", loss(params, x, y))
        iteration_time = time.time() - start_time
        print("Epoch {}, Training Time {:0.2f} sec".format(iteration, iteration_time))
        print("Accuracy {}\n".format(accuracy(params, x, y)))
