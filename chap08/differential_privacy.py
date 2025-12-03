import numpy as np
from sklearn.linear_model import LogisticRegression


class DifferentiallyPrivateTrainer:
    """
    Differentially Private Trainer for Logistic Regression.

    Implements DP-SGD with noise addition for differential privacy.
    """

    def __init__(self, epsilon=1.0, delta=1e-5):
        # Initialize privacy parameters and logistic regression model
        self.epsilon = epsilon  # Privacy budget: lower means more privacy
        self.delta = delta  # Probability of privacy violation
        self.model = LogisticRegression()  # Underlying logistic regression model

    def _add_noise(self, gradients):
        """
        Add calibrated Gaussian noise to gradients.

        Uses L2 sensitivity to compute the noise scale and applies noise.
        """
        sensitivity = 1.0  # L2 sensitivity for normalized gradients
        # Calculate noise scale based on epsilon and delta
        noise_scale = (
            sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        )
        # Generate Gaussian noise with computed scale
        noise = np.random.normal(0, noise_scale, size=gradients.shape)
        return gradients + noise

    def train(self, X, y, batch_size=64, epochs=10):
        """
        Train the model using differentially private SGD.

        Shuffles data, computes and clips gradients, adds noise, and updates the model.
        """
        n_samples, n_features = X.shape
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            # Shuffle dataset at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Compute gradients for the current batch
                gradients = self._compute_gradients(X_batch, y_batch)

                # Clip gradients to ensure bounded sensitivity
                gradients = self._clip_gradients(gradients, clip_norm=1.0)

                # Add calibrated noise for differential privacy
                private_gradients = self._add_noise(gradients)

                # Update the model parameters with noisy gradients
                self._update_model(private_gradients)

        return self.model
