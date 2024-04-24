from lxml import etree
import numpy as np
import os
import re
def preprocess_inkml(content):
    used_ids = set()  # To track used IDs to avoid duplicates

    def replace_invalid_chars(match):
        original_id = match.group(1)
        # Normalize the ID by replacing invalid characters with underscores
        valid_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_id)
        # Ensure it starts with a valid character
        if not valid_id[0].isalpha():
            valid_id = f"id_{valid_id}"

        # Ensure uniqueness of the ID
        original_valid_id = valid_id
        count = 1
        while valid_id in used_ids:
            valid_id = f"{original_valid_id}_{count}"
            count += 1
        used_ids.add(valid_id)
        
        return f'xml:id="{valid_id}"'

    pattern = re.compile(r'xml:id="([^"]+)"')
    corrected_content = pattern.sub(replace_invalid_chars, content)
    return corrected_content
# Example usage:
# Suppose 'inkml_content' is your loaded InkML content as a string
#inkml_content = """<ink xmlns="http://www.w3.org/2003/InkML">... your XML data here ...</ink>"""
#corrected_content = preprocess_inkml(inkml_content)


print("------------------ READ FOLDER -------------------------")
def parse_inkml(inkml_path):
    ns = {'inkml': 'http://www.w3.org/2003/InkML'}
    try:
        with open(inkml_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Pre-process the content to fix xml:id attributes
        corrected_content = preprocess_inkml(content)

        # Parse the corrected XML content
        root = etree.fromstring(corrected_content.encode('utf-8'))
    
        # Extracting strokes
        traces = []
        for trace in root.xpath('//inkml:trace', namespaces=ns):
            points = []
            for point in trace.text.strip().split(','):
                x, y = map(float, point.strip().split(' '))
                points.append([x, y])
            traces.append(np.array(points))

        # Extracting label
        annotation_xml = root.xpath('//inkml:annotation[@type="truth"]', namespaces=ns)
        label = annotation_xml[0].text if annotation_xml else "Label not found"
        if not annotation_xml:
            print("No label found in the file, check annotation type or file content.")

        return traces, label
    except Exception as e:
        print(f"Error processing file {inkml_path}: {e}")
        return [], "Error"
    

def read_folder_of_inkml(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.inkml'):
            file_path = os.path.join(folder_path, filename)
            strokes, label = parse_inkml(file_path)
            data.append((strokes, label))
    return data

# Specify the path to the folder containing the InkML files
folder_path = 'TestINKMLGT'
all_data = read_folder_of_inkml(folder_path)

# Example of how to access the data
for index, (strokes, label) in enumerate(all_data):
    print(f"Data from file {index}: Label = {label}, Number of strokes = {len(strokes)}")


import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def normalize_and_pad_traces(all_data):
    # Normalize each stroke sequence and pad data to the maximum sequence length in the batch
    all_traces = [item for sublist in [data[0] for data in all_data] for item in sublist]  # flatten all traces
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Determine the maximum length of any trace sequence for uniform padding
    max_len = max(len(trace) for traces in all_data for trace in traces[0])

    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))
    return padded_data


def encode_labels(padded_data):
    labels = [label for _, label in padded_data]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Flatten the nested sequences
    inputs = [trace for traces, _ in padded_data for trace in traces]
    flattened_labels = [encoded_labels[i] for i, (traces, _) in enumerate(padded_data) for _ in range(len(traces))]

    categorical_labels = to_categorical(flattened_labels)
    
    return np.array(inputs), np.array(categorical_labels), encoder

print("-------------- NOMALIZE  -----------------------------")

padded_data = normalize_and_pad_traces(all_data)
print("-------------- ENCODE -----------------------------")

inputs, labels, label_encoder = encode_labels(padded_data)
print("-------------- TRAIN -----------------------------")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_input_shape(inputs):
    max_len = max(len(seq) for seq in inputs)
    max_features = max(len(seq[0]) for seq in inputs)
    return (max_len, max_features)

print("-------------- BUILD MODEL -----------------------------")

# Get the input shape from the first input sample and the number of classes from the labels
input_shape = get_input_shape(inputs)
num_classes = labels.shape[1]

model = build_model(input_shape, num_classes)
model.summary()

print("-------------- FIT -----------------------------")

history = model.fit(inputs, labels, epochs=10, batch_size=64, validation_split=0.2)
model.save("model.h5")



import matplotlib.pyplot as plt
print("-------------- PLOT HISTORY -----------------------------")

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model accuracy by Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('model_accuracy.png')  # Save the figure
plt.show()


# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss by Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('model_loss.png')  # Save the figure
plt.show()

test_folder_path = 'test_inputs'
#all_test_data = read_folder_of_inkml(test_folder_path)
print("-------------- TEST -----------------------------")

    