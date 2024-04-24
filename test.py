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


def normalize_and_pad_traces_old(all_data):
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
    

def normalize_and_pad_traces(all_data):
    all_traces = [item for sublist in [data[0] for data in all_data] for item in sublist]
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_len = max(len(trace) for trace in all_traces)

    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))
    return padded_data


def normalize_and_pad_traces33(all_data):
    # Normalize each stroke sequence and determine the global max length
    all_traces = [item for sublist in [data[0] for data in all_data] for item in sublist]
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_len = max(len(trace) for trace in all_traces)

    # Pad each trace and stack them into a single NumPy array
    padded_data = []
    for traces, _ in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.extend(padded_traces)  # Extend, not append, to flatten the list

    return np.stack(padded_data)  
    
def normalize_and_pad_traces4(all_data):
    # Flatten all traces to find the global minimum and maximum for normalization
    all_traces = [trace for data in all_data for trace in data[0] if trace.size > 0]
    if not all_traces:
        return np.array([], dtype=float)

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Determine the maximum length of any trace sequence for uniform padding
    max_len = max(len(trace) for trace in all_traces)

    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        # Pad sequences to the maximum length
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))

    return padded_data

def normalize_and_pad_traces2(all_data):
    if not all_data:
        return np.array([])  # Return an empty array if no data

    all_traces = [trace for data in all_data for trace in data[0] if trace.size > 0]
    if not all_traces:
        return np.array([])  # Return an empty array if no valid traces

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    max_len = max(len(trace) for trace in all_traces)

    padded_data = []
    for traces, _ in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.extend(padded_traces)  # Flatten the batch for uniform shape

    return np.array(padded_data, dtype='float32')

def read_folder_of_inkml(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.inkml'):
            file_path = os.path.join(folder_path, filename)
            strokes, label = parse_inkml(file_path)
            data.append((strokes, label))
    return data


def normalize_and_pad_traces8(all_data, max_len):
    all_traces = [trace for data in all_data for trace in data[0]]
    if not all_traces:
        return np.empty((0, max_len, 2))  # Adjust dimensions as needed

    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    padded_traces = []
    for traces in all_traces:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces.append(pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32'))

    return np.stack(padded_traces)



def normalize_and_pad_tracesb(all_data):
    # Normalize each stroke sequence and determine the global max length
    all_traces = []
    for data in all_data:
        traces = data[0]
        for trace in traces:
            all_traces.append(trace)
    
    if not all_traces:
        return np.empty((0, 0, 0))  # Return an empty 3D array if no traces are present
    
    all_points = np.concatenate(all_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    max_len = max(len(trace) for trace in all_traces)

    # Pad each trace and normalize
    padded_data = []
    for traces, label in all_data:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        padded_data.append((padded_traces, label))

    # Combine all traces into a single NumPy array
    traces_array = np.vstack([traces for traces, _ in padded_data])
    return traces_array

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
padded_test_data = normalize_and_pad_traces(test_data)


#test_inputs = normalize_and_pad_tracesb(padded_test_data)
# Use the function to prepare test data
test_inputs = normalize_and_pad_traces9(test_data, max_len)



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
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Print or process your predictions
print(predicted_labels)