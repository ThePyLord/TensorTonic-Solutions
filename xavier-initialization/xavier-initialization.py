def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    L = np.sqrt(6/(fan_in + fan_out))

    W = np.asarray(W) * (2 * L) - L
    return W