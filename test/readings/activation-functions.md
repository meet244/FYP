# Activation functions (ITC604 reading)

ClassScribe test document. Upload as a `.md` material so the local index has a
plain-text modality next to the PDFs and diagrams.

## Step

Used in the original perceptron. Output is 0 or 1. Not differentiable at 0, and
the derivative is 0 everywhere else, so it cannot be used with backpropagation.

## Sigmoid

σ(z) = 1 / (1 + exp(-z)). Range (0, 1). Derivative σ(z)(1 − σ(z)). Saturates for
|z| large, which makes gradients tiny — the vanishing-gradient problem in deep
sigmoid stacks. Still used as a single output unit for binary classification
when the loss is binary cross-entropy.

## Tanh

tanh(z) = 2σ(2z) − 1. Range (−1, 1), zero-centred, otherwise the same saturation
problem as sigmoid. Common in older recurrent nets.

## ReLU

ReLU(z) = max(0, z). Derivative 1 for z > 0, 0 for z < 0. Cheap, sparse, and the
default hidden activation since AlexNet. Dying ReLU: a unit that always sees
z < 0 never updates. Leaky ReLU uses a small slope (typically 0.01) on the
negative side.

## Softmax

For a K-way output, softmax(z)_k = exp(z_k) / sum_j exp(z_j). Used with
categorical cross-entropy. Not an activation inside hidden layers.

## Rule of thumb for this course

Hidden layers: ReLU. Binary output: sigmoid + BCE. Multiclass output: softmax +
cross-entropy. Do not put a sigmoid on every hidden layer and then wonder why
training stalled.
