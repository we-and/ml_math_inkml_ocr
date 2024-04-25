from lxml import etree
import numpy as np
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical



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


#Output shape: (num_samples, sequence_length, num_features)
def normalize_and_pad_traces9(all_data, max_len):
    all_traces = [trace for data in all_data for trace in data[0]]
    if not all_traces:
        return np.empty((0, max_len, 2))  # Ensure 3D array with correct dims even if empty

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Normalize and pad each sequence to max_len
    padded_traces = []
    for traces in all_traces:
        # Normalize traces
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        # Pad each normalized trace to the max_len
        padded_traces.append(pad_sequences([normalized_traces], maxlen=max_len, padding='post', dtype='float32')[0])  # pad_sequences returns a list

    return np.stack(padded_traces) 
    # Use the function to prepare test data

import json
with open('model_config.json', 'r') as f:
    config = json.load(f)
max_len = config['max_len']

# Load the model
model = load_model('model.h5')

# Load and preprocess test data
folder_path_test = 'test_inputs'  # Adjust this path to your test data
test_data = read_folder_of_inkml(folder_path_test)
#padded_test_data = normalize_and_pad_traces(test_data)
print("N inputs:",len(test_data))
print("N input parts",len(test_data[0]))
print("N strokes",len(test_data[0][0]))
print("STROKES",test_data[0][0])
print("LABEL",test_data[0][1])


#test_inputs = normalize_and_pad_tracesb(padded_test_data)
# Use the function to prepare test data
test_inputs = normalize_and_pad_traces9(test_data, max_len)
print("NORMALIZED",len(test_inputs))


# Prepare test inputs
#test_inputs = np.array([traces for traces, _ in padded_test_data])


# Assuming padded_test_data is prepared correctly and includes labels:
#test_inputs = np.array([traces for traces, _ in padded_test_data])

# If labels are not needed or not included in padded_test_data:
#test_inputs = np.array([data[0] for data in padded_test_data])  # use data[0] to get traces only


# Check the shape and dtype of the inputs
print("Test data shape:", test_inputs.shape)
print("Test data dtype:", test_inputs.dtype)


# Predict using the model
print("Data shape:", test_inputs.shape)
print("Data dtype:", test_inputs.dtype)
print("Data ndim:", test_inputs.ndim)
if test_inputs.ndim != 3 or test_inputs.dtype != np.float32:
    raise ValueError("Input data is not properly formatted for prediction.")

predictions = model.predict(test_inputs)

# Assuming labels are categorical and need to be encoded
def encode_labels(padded_data):
    labels = [label for _, label in padded_data]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    return np.array([data[0] for data in padded_data]), categorical_labels, encoder



# Optional: Decode predictions if needed
predicted_classes = np.argmax(predictions, axis=1)

import pickle
# Loading the encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

predicted_labels = label_encoder.inverse_transform(predicted_classes)


# Print or process your predictions
print(predicted_classes)
print(predicted_labels)


# Assume you have an array that maps each of the 130 predictions back to one of 15 files
# For simplicity, let's create a mock mapping (ensure to replace this with your actual mapping based on how traces are grouped into files)
file_mapping = np.repeat(np.arange(15), 130 // 15)[:130]

# Aggregate predictions by files
file_predictions = []
for i in range(15):
    # Get predictions for the current file
    file_traces_predictions = predicted_classes[file_mapping == i]
    # Find the most common prediction (majority voting)
    most_common_prediction = Counter(file_traces_predictions).most_common(1)[0][0]
    file_predictions.append(most_common_prediction)

# Now `file_predictions` contains one prediction per file
print("Aggregated File Predictions:", file_predictions)

