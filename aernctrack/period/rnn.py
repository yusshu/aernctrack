import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_period_rnn(xs, ys, sequence_length=50, epochs=100):
    # Ensure xs and ys have the same length
    if len(xs) != len(ys) or len(xs) < sequence_length:
        return None

    # Prepare the data for the LSTM
    X_sequences = []
    Y_sequences = []
    for i in range(len(ys) - sequence_length):
        X_sequences.append(ys[i:i + sequence_length])
        Y_sequences.append(ys[i + sequence_length])

    X_sequences = np.array(X_sequences)
    Y_sequences = np.array(Y_sequences)

    # Reshape data for LSTM [samples, time steps, features]
    X_sequences = X_sequences.reshape((X_sequences.shape[0], sequence_length, 1))

    # Build the LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_sequences, Y_sequences, epochs=epochs, verbose=0)

    # Generate predictions for the signal
    predictions = model.predict(X_sequences, verbose=0).flatten()

    # Compute the error between actual and predicted to estimate periodicity
    diffs = np.abs(predictions - Y_sequences)
    min_diff_idx = np.argmin(diffs)

    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(xs[sequence_length:], ys[sequence_length:], label='Actual Data', color='blue')
    # plt.plot(xs[sequence_length:], predictions, label='Predicted Data', color='red', linestyle='dashed')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Actual vs Predicted Data')
    # plt.legend()
    # plt.show()

    # Estimate the period by looking at the index where the model best fits the repeating pattern
    estimated_period_idx = np.where(diffs < np.mean(diffs) * 0.5)[0]
    if len(estimated_period_idx) < 2:
        return None

    # Calculate the estimated period in terms of the index difference and convert to time
    period_indices = estimated_period_idx[1] - estimated_period_idx[0]
    sample_spacing = xs[1] - xs[0]
    period = period_indices * sample_spacing

    return period
