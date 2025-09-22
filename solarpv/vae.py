from __future__ import annotations


def build_vae_keras(
    input_dim: int,
    hidden_dim: int = 64,
    latent_dim: int = 8,
    lr: float = 1e-3,
):
    """Build a simple VAE using Keras/TensorFlow 2.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden layer size.
        latent_dim: Latent space dimension.
        lr: Learning rate for Adam optimizer.

    Returns:
        Tuple of (VAE model, encoder model that outputs z_mean).
    """
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers

    inp = layers.Input(shape=(input_dim,))
    h = layers.Dense(hidden_dim, activation="relu")(inp)
    z_mean = layers.Dense(latent_dim)(h)
    z_logvar = layers.Dense(latent_dim)(h)

    # One blank line before a nested def for PEP8 E306
    def sampling(args):
        z_m, z_lv = args
        eps = tf.random.normal(shape=(tf.shape(z_m)[0], tf.shape(z_m)[1]))
        return z_m + tf.exp(0.5 * z_lv) * eps

    z = layers.Lambda(sampling)([z_mean, z_logvar])
    dh = layers.Dense(hidden_dim, activation="relu")(z)
    out = layers.Dense(input_dim)(dh)
    vae = Model(inp, out)

    recon = tf.reduce_mean(
        tf.reduce_sum(tf.square(inp - out), axis=1)
    )
    kl = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
    )
    vae.add_loss(recon + 1e-3 * kl)
    vae.compile(optimizer=optimizers.Adam(lr))
    enc = Model(inp, z_mean)
    return vae, enc
