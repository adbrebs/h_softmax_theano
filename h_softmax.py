import theano
from theano import tensor

from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable

try:
    from theano.sandbox.cuda.blocksparse import sparse_block_dot_SS
except ImportError:
    pass
            
            
def h_softmax_cpu(W1, b1, W2, b2, x, n_outputs, n_classes,
                  n_outputs_per_class, batch_size, target=None):
    """
    CPU version of a two-layer hierarchical softmax. This function also works
    on GPU but h_softmax_gpu is more optimized.
    See hierarchical_softmax's docstring for the description of the arguments.
    """

    # First softmax which computes the probabilities of belonging to each class
    class_probs = tensor.nnet.softmax(tensor.dot(x, W1) + b1)

    if target is None:
        # Computes the probabilites of all the outputs

        def _compute_output_probs_per_class(class_id, W2, b2, x):

            # Computes the probabilities of the outputs of to a specific class
            output_prob = tensor.nnet.softmax(tensor.dot(x, W2[class_id, :, :]) +
                                         b2[class_id, :])
            output_prob = output_prob * class_probs[:, class_id][:, None]
            return output_prob

        output_probs = theano.scan(_compute_output_probs_per_class,
                                   tensor.arange(n_classes),
                                   non_sequences=[W2, b2, x],
                                   name='compute_output_probs_per_class')[0]
        output_probs = output_probs.dimshuffle((1, 0, 2))
        output_probs = output_probs.reshape((batch_size,
                                             n_classes * n_outputs_per_class))
        output_probs = output_probs[:, :n_outputs]

    else:
        # Computes the probabilities of the outputs specified by the targets

        # Flattens the targets
        target = target.flatten()

        # Classes to which belong each target
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_classes

        def _compute_output_prob_per_point(idx_point, W2, b2, x):
            # Computes the output probability of a specific datapoint
            class_point = target_classes[idx_point]
            output_prob = tensor.nnet.softmax(tensor.dot(x[idx_point], W2[class_point, :, :]) +
                                         b2[class_point, :])
            output_prob = output_prob[0, target_outputs_in_class[idx_point]] \
                          * class_probs[idx_point, class_point]
            return output_prob

        output_probs = theano.scan(_compute_output_prob_per_point,
                                   tensor.arange(batch_size),
                                   non_sequences=[W2, b2, x],
                                   name='compute_output_prob_per_point')[0]

    return output_probs


def h_softmax_gpu(W1, b1, W2, b2, x, n_outputs, n_classes,
                  n_outputs_per_class, batch_size, target=None):
    """
    GPU-only version of a two-layer hierarchical softmax.
    See hierarchical_softmax's docstring for the description of the arguments.
    """
    W1 = as_cuda_ndarray_variable(W1)
    b1 = as_cuda_ndarray_variable(b1)
    W2 = as_cuda_ndarray_variable(W2)
    b2 = as_cuda_ndarray_variable(b2)
    x = as_cuda_ndarray_variable(x)

    # First softmax which computes the probabilities of belonging to each class
    class_probs = tensor.nnet.softmax(tensor.dot(x, W1) + b1)

    if target is None:
        # Computes the probabilites of all the outputs

        class_ids = tensor.tile(tensor.arange(n_classes, dtype="int32")[None, :], (batch_size, 1))

        # Second softmax that computes the output probabilities
        activations = sparse_block_dot_SS(
            W2[None, :, :, :], x[:, None, :],
            tensor.zeros((batch_size, 1), dtype='int32'), b2, class_ids)

        output_probs = tensor.nnet.softmax(activations.reshape((-1, n_outputs_per_class)))
        output_probs = output_probs.reshape((batch_size, n_classes, -1))
        output_probs = class_probs[:, :, None] * output_probs
        output_probs = output_probs.reshape((batch_size, -1))
        output_probs = output_probs[:, :n_outputs]

    else:
        # Computes the probabilities of the outputs specified by the targets

        # Flattens the targets
        target = target.flatten()

        # Classes to which belong each target
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_classes

        # Second softmax that computes the output probabilities
        activations = sparse_block_dot_SS(
            W2[None, :, :, :], x[:, None, :],
            tensor.zeros((batch_size, 1), dtype='int32'), b2,
            target_classes[:, None])

        output_probs = tensor.nnet.softmax(activations[:, 0, :])
        target_class_probs = class_probs[tensor.arange(batch_size), target_classes]
        output_probs = output_probs[tensor.arange(batch_size),
                                    target_outputs_in_class]
        output_probs = target_class_probs * output_probs

    return output_probs
