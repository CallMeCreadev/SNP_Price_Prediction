from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from AI_data_formatter import DataFormatter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


sequence_length = 20  # or any other length you choose

def get_feature_by_loc(callback, sequence_length):
    """
    Construct and reshape the feature data for LSTM input.

    :param sequence_length: The number of time steps in each input sequence for the LSTM.
    :return: The reshaped data ready for LSTM input.
    """
    current_features = DataFormatter.construct_complete_feature_dictionary(callback, sequence_length)
    unpacked_current_feature_array = np.array(list(current_features.values()))
    current_feature_array = np.expand_dims(unpacked_current_feature_array, axis=0)
    current_data_reshaped = current_feature_array.reshape(1, sequence_length, 285)  # Explicitly add batch dimension

    return current_data_reshaped


def generate_denormalized_predictions(feature_index, sequence_length, model, target_feature_name):
    """
    Generate denormalized predictions for a given feature index and sequence length using a provided model.

    :param feature_index: The index of the feature to predict.
    :param sequence_length: The sequence length for the model input.
    :param model: The trained model used for prediction.
    :param target_feature_name: The name of the target feature for denormalization.
    :return: The denormalized predictions.
    """
    # Fetching the feature data
    feature_data = get_feature_by_loc(feature_index, sequence_length)  # Replace or define this function as needed

    # Generating predictions
    predictions = model.predict(feature_data)

    # Denormalizing predictions
    denormalized_predictions = DataFormatter.denormalize_data(predictions, 'closing_prices', target_feature_name)

    # Converting predictions into prices
    prices = DataFormatter.convert_prediction_into_prices(denormalized_predictions, days_ago= (feature_index+1))

    # Outputting the results
    print(f"Predictions {feature_index} days ago: {predictions}")
    print("Denormalized predictions:", denormalized_predictions)
    print("future 7 day moving average", prices[0])

    return prices


def calculate_if_correct(feature_index, sequence_length, model, target_feature_name):
    # Fetching the feature data
    feature_data = get_feature_by_loc(feature_index, sequence_length)  # Replace or define this function as needed

    # Generating predictions
    predictions = model.predict(feature_data)

    denormalized_predictions = DataFormatter.denormalize_data(predictions, 'closing_prices', target_feature_name)

    # Denormalizing predictions
    check_correct_pred = DataFormatter.validate_predictions(denormalized_predictions, days_ago=feature_index+1)

    return check_correct_pred


targets = DataFormatter.construct_complete_targets()
#eekly_only_targets = [item[:3] for item in targets]
#target_array = np.array(weekly_only_targets)
target_array = np.array(targets)
print("Shape of target array:", target_array.shape)
print(target_array[-1])
adjusted_target_array = target_array[sequence_length - 1:]


features = DataFormatter.construct_complete_feature_dictionary(callback=-1, sequence_size=sequence_length)
features_lengths = {key: len(value) for key, value in features.items()}
print(features_lengths)
unpacked_feature_array = np.array(list(features.values()))
print(unpacked_feature_array.shape)
feature_array = np.transpose(unpacked_feature_array, (1, 0, 2))
print(feature_array.shape)
data_reshaped = feature_array.reshape(3983, 285)
print("Shape of flattened array:", data_reshaped.shape)







number_of_samples = data_reshaped.shape[0] - sequence_length + 1
number_of_features = 285  # From your reshaped data

# Initialize an empty array for the reshaped data
lstm_features = np.zeros((number_of_samples, sequence_length, number_of_features))

# Populate the array with sequences of your data
for i in range(number_of_samples):
    lstm_features[i] = data_reshaped[i:i+sequence_length]






# Data splitting
X_train, X_test, y_train, y_test = train_test_split(lstm_features, adjusted_target_array, test_size=0.2, random_state=42)

# Model definition
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(sequence_length, 285)))
model.add(Dropout(0.2))  # Dropout layer after the first LSTM
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))  # Dropout layer after the second LSTM
model.add(LSTM(100))  # Last LSTM layer should not return sequences
model.add(Dropout(0.2))  # Dropout layer after the last LSTM
model.add(Dense(5, activation='linear'))

# Compiling the model with a slightly adjusted learning rate
optimizer = Adam(learning_rate=0.001)  # Reduced learning rate
model.compile(optimizer=optimizer, loss='huber_loss')

# Early stopping callback with increased patience
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Model training with validation split and early stopping
model.fit(X_train, y_train, epochs=300, batch_size=20, validation_split=0.2, callbacks=[early_stopping])

# Model evaluation

most_recent = generate_denormalized_predictions(0, sequence_length, model, 'TGT_SPY')
yesterday = generate_denormalized_predictions(1, sequence_length, model, 'TGT_SPY')
two_days_ago = generate_denormalized_predictions(2, sequence_length, model, 'TGT_SPY')
three_days_ago = generate_denormalized_predictions(3, sequence_length, model, 'TGT_SPY')
four_days_ago = generate_denormalized_predictions(4, sequence_length, model, 'TGT_SPY')
last_week = generate_denormalized_predictions(5, sequence_length, model, 'TGT_SPY')
most_recent1 = generate_denormalized_predictions(6, sequence_length, model, 'TGT_SPY')
yesterday2 = generate_denormalized_predictions(7, sequence_length, model, 'TGT_SPY')
two_days_ago3 = generate_denormalized_predictions(8, sequence_length, model, 'TGT_SPY')
three_days_ago4 = generate_denormalized_predictions(9, sequence_length, model, 'TGT_SPY')
four_days_ago5 = generate_denormalized_predictions(10, sequence_length, model, 'TGT_SPY')
last_week6 = generate_denormalized_predictions(11, sequence_length, model, 'TGT_SPY')


count = 0
for i in range(1, 250):
    count += calculate_if_correct(i, sequence_length, model, 'TGT_SPY')
print(count)





