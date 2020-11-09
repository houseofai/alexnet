import tensorflow as tf


class PCAColorAugmentation:
    def __init__(self, std_deviation=0.1, clipping=False, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.std_deviation = std_deviation
        self.clipping = clipping
        self.scale = scale

    def augmentation(self, inputs):
        # assume channels-last
        input_shape = tf.keras.backend.int_shape(inputs)
        ranks = len(input_shape)
        assert ranks == 3
        chs = input_shape[-1]

        x = inputs

        # scaling-factor
        calculate_axis, reduce_axis = 1, 2
        C = tf.keras.backend.int_shape(x)[reduce_axis]
        var = tf.keras.backend.var(x, axis=calculate_axis, keepdims=True)
        scaling_factors = tf.sqrt(C / tf.reduce_sum(var, axis=reduce_axis, keepdims=True))
        # scaling
        x = x * scaling_factors

        # subtract mean for cov matrix
        mean = tf.reduce_mean(x, axis=calculate_axis, keepdims=True)
        x -= mean

        # covariance matrix
        cov_n = max(tf.keras.backend.int_shape(x)[calculate_axis] - 1, 1)
        # cov (since x was normalized)
        cov = tf.matmul(x, x, transpose_a=True) / cov_n

        # eigen value(S), eigen vector(U)
        S, U, V = tf.linalg.svd(cov)
        # eigen_values vs eigen_vectors

        # random values
        rand = tf.random.normal(tf.shape(S), mean=0.0, stddev=self.std_deviation)
        delta_original = tf.squeeze(tf.matmul(U, tf.expand_dims(rand * S, axis=-1)), axis=-1)

        # adjust delta shape
        delta = tf.expand_dims(delta_original, axis=ranks - 2)

        # delta scaling
        delta = delta * self.scale

        # clipping (if clipping=True)
        result = inputs + delta
        if self.clipping:
            result = tf.clip_by_value(result, 0.0, self.scale)

        return result
