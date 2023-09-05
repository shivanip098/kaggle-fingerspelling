import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, GRU, Dropout, Bidirectional, Concatenate
from keras.utils import to_categorical, pad_sequences
from keras.optimizers import Adam
from joblib import Parallel, delayed
from keras.wrappers.scikit_learn import KerasClassifier
import zipfile
import json
import tensorflow as tf

class SequenceLabelingModel:
    def __init__(self, model_filepath=None, label_encoder_filepath=None):
        self.model = None
        self.label_encoder = None
        if model_filepath and label_encoder_filepath:
            self.load_model(model_filepath, label_encoder_filepath)

    def load_model(self, model_filepath, label_encoder_filepath):
        # Load the trained model
        self.model = load_model(model_filepath)

        # Load the label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_encoder_filepath)

    def preprocess_sequence(self, sequence_frames):
        # Pad the sequence frames
        sequences_padded = pad_sequences([sequence_frames], padding='post')

        return sequences_padded

    def predict_sequence(self, sequence_frames):
        # Preprocess the sequence
        sequences_padded = self.preprocess_sequence(sequence_frames)

        # Make predictions
        predictions = self.model.predict(sequences_padded)

        return predictions[0]  # Return the predicted probabilities

    def predict_sequences(self, sequences_frames):
        
        num_cores = 8
        # Use parallel processing to make predictions on multiple sequences simultaneously
        predictions = Parallel(n_jobs=num_cores)(
            delayed(self.predict_sequence)(sequence_frames)
            for sequence_frames in sequences_frames
        )
        return predictions
    

    def preprocess_data(self):
        # Read the CSV file and select desired columns
        train_data = pd.read_csv('train.csv')

        # Convert sequence_id column to numeric type
        train_data['sequence_id'] = pd.to_numeric(train_data['sequence_id'], errors='coerce')

        # Initialize lists to store data
        sequences = []
        labels = []
        column_order = []

        # Iterate over each file in the zip
        with zipfile.ZipFile('train_landmarks.zip', 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.parquet'):
                    with zip_ref.open(file_info) as file:
                        landmark_data = pd.read_parquet(file)

                        # Identify the relevant columns based on left and right hand landmarks
                        left_hand_columns = landmark_data.filter(regex=r'[xy]_left_hand_.*').columns
                        right_hand_columns = landmark_data.filter(regex=r'[xy]_right_hand_.*').columns

                        # Define the specific face datapoints indices
                        specific_face_datapoints = [57, 61, 185, 76, 146, 62, 184, 183, 78, 77, 96, 95, 191, 40, 74, 42, 80, 88, 89, 90, 91, 39, 73, 41, 81, 178, 179, 180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315, 314, 269, 303, 271, 311, 402, 403, 404, 405, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324, 325, 307, 375, 292, 291, 306]

                        # Filter the relevant face columns from the landmark data
                        relevant_face_columns = [col for col in landmark_data.columns if any(f"_face_{index}" in col for index in specific_face_datapoints) and ('x' in col or 'y' in col)]
                        #facial_columns = landmark_data[relevant_face_columns]

                        # Filter the relevant columns for left and right hand landmarks
                        relevant_columns = left_hand_columns.union(right_hand_columns)
                        relevant_columns = relevant_columns.union(relevant_face_columns)

                        # Append the relevant columns to the desired column order
                        column_order.extend(relevant_columns)

                        # Create a dictionary to hold the selected columns
                        selected_columns = {
                            "selected_columns": relevant_columns.tolist()
                        }

                        # Write the dictionary to a JSON file
                        with open("selected_columns.json", "w") as f:
                            json.dump(selected_columns, f)

                        # Include the 'frame' column in the relevant columns
                        relevant_columns = relevant_columns.union(['frame'])

                        # Extract the subset of columns from the landmark data
                        landmark_data = landmark_data[relevant_columns]

                        # Replace missing values with -1 using fillna
                        landmark_data.fillna(value=-1, inplace=True)

                        # Create indicator variables for missing values in relevant columns
                        indicator_columns = []
                        indicator_data = []
                        for column in relevant_columns:
                            indicator_column = f'{column}_missing'
                            indicator_columns.append(indicator_column)
                            indicator_data.append(landmark_data[column].isnull().astype(int))

                        indicator_data = pd.concat(indicator_data, axis=1, keys=indicator_columns)
                        landmark_data = pd.concat([landmark_data, indicator_data], axis=1)

                        # Filter every 3rd frame
                        landmark_data = landmark_data.loc[landmark_data['frame'] % 3 == 0]

                        # Merge landmark data with train data
                        merged_data = landmark_data.merge(train_data, on='sequence_id', how='left')

                        # Iterate over unique sequence IDs
                        for sequence_id in merged_data['sequence_id'].unique():
                            sequence_data = merged_data.loc[merged_data['sequence_id'] == sequence_id]

                            # Get all frames and corresponding labels for the sequence ID
                            sequence_frames = sequence_data.drop(['sequence_id', 'frame', 'phrase', 'path', 'participant_id', 'file_id'], axis=1).values
                            sequence_label = sequence_data['phrase'].iloc[0]  # Assuming the phrase is the same for all frames in a sequence

                            # Split the phrase into individual characters
                            characters = list(sequence_label)

                            # Compute the number of frames per character
                            frames_per_character = len(sequence_frames) // len(characters)

                            # Append frames and labels to the lists
                            for i, character in enumerate(characters):
                                start_index = i * frames_per_character
                                end_index = start_index + frames_per_character

                                character_frames = sequence_frames[start_index:end_index]

                                # Check if character_frames is not empty
                                if len(character_frames) > 0:
                                    sequences.append(character_frames)
                                    labels.append(character)

        # Convert sequences and labels to numpy arrays
        sequences = np.array(sequences)
        labels = np.array(labels)

        # Perform label encoding
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Normalize and pad the sequences
        normalized_sequences = []
        for sequence in sequences:
            # Check for empty sequences
            if len(sequence) == 0:
                continue

            # Normalize the sequence
            mean = np.mean(sequence, axis=0)
            std = np.std(sequence, axis=0)

            # Check for invalid values
            if np.any(np.isnan(mean)) or np.any(np.isnan(std)):
                continue

            normalized_sequence = (sequence - mean) / std
            normalized_sequences.append(normalized_sequence)

        # Check if any valid sequences were found
        if len(normalized_sequences) == 0:
            raise ValueError("No valid sequences found for normalization.")

        # Pad sequences to a fixed length
        self.sequences_padded = pad_sequences(normalized_sequences, padding='post')

        # Verify consistent sample sizes
        print(f"Number of sequences: {len(self.sequences_padded)}")
        print(f"Number of labels: {len(encoded_labels)}")

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.sequences_padded, encoded_labels, test_size=0.2, random_state=42)

        # Convert labels to categorical format
        num_classes = len(self.label_encoder.classes_)
        self.y_train_categorical = to_categorical(self.y_train, num_classes=num_classes)
        self.y_test_categorical = to_categorical(self.y_test, num_classes=num_classes)


    def build_model(self):
        # Determine the input shape
        input_shape = self.X_train.shape[1:]

        # Get the number of classes from the label encoder
        num_classes = len(self.label_encoder.classes_)

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=6, batch_size=15, validation_data=(self.X_test, self.y_test))

    def evaluate_model(self):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

    def save_model(self, model_file, label_encoder_file):
        # Save the model and label encoder
        self.model.save(model_file)
        np.save(label_encoder_file, self.label_encoder.classes_)

# Create an instance of the SequenceLabelingModel class
sequence_labeling_model = SequenceLabelingModel()

# Create OneDeviceStrategy for single-device training
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

# Inside the strategy.scope():
with strategy.scope():
    # Preprocess the data
    sequence_labeling_model.preprocess_data()

    # Build the model
    sequence_labeling_model.build_model()
    
    # Train the model
    sequence_labeling_model.train_model()

    # Evaluate the model
    sequence_labeling_model.evaluate_model()

    # Save the model and label encoder
    sequence_labeling_model.save_model('model.h5', 'label_encoder.npy')