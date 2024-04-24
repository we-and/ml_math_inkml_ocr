from lxml import etree
import numpy as np
import os
import re
def preprocess_inkml(content):
    # This pattern matches xml:id attributes that contain any non-NCName-valid characters
    def replace_invalid_chars(match):
        original_id = match.group(1)
        # Replace invalid characters with underscores and ensure it starts with a letter or underscore
        valid_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_id)
        if not valid_id[0].isalpha():
            valid_id = f"_{valid_id}"
        return f'xml:id="{valid_id}"'

    pattern = re.compile(r'xml:id="([^"]+)"')
    corrected_content = pattern.sub(replace_invalid_chars, content)
    return corrected_content

# Example usage:
# Suppose 'inkml_content' is your loaded InkML content as a string
#inkml_content = """<ink xmlns="http://www.w3.org/2003/InkML">... your XML data here ...</ink>"""
#corrected_content = preprocess_inkml(inkml_content)


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

print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")

from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_strokes(strokes):
    # Normalize strokes to [0, 1] range
    flat_strokes = np.concatenate(strokes)
    min_vals = flat_strokes.min(0)
    max_vals = flat_strokes.max(0)
    normalized_strokes = [(stroke - min_vals) / (max_vals - min_vals) for stroke in strokes]

    # Pad sequences to the same length
    padded_strokes = pad_sequences(normalized_strokes, padding='post', dtype='float32')
    return padded_strokes

padded_strokes = preprocess_strokes(strokes)


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

# Assuming num_classes is the number of different characters/symbols in the dataset
num_classes = 100  # This needs to be set to the actual number of classes
model = build_model((None, 2), num_classes)


from tensorflow.keras.utils import to_categorical

# Assuming `labels` are integer-encoded
labels_categorical = to_categorical(labels, num_classes=num_classes)
model.fit(padded_strokes, labels_categorical, epochs=10, validation_split=0.2)

