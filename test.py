from lxml import etree
import numpy as np
import os
import re
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
    


def load_test_data(test_folder):
    test_data = []
    for filename in os.listdir(test_folder):
        if filename.endswith('.inkml'):
            file_path = os.path.join(test_folder, filename)
            traces = parse_inkml(file_path)
            test_data.append(traces)
    return test_data


def preprocess_test_data(test_data):
    # Ensure that the traces are arrays and filter out any empty traces
    filtered_traces = []
    for traces in test_data:
        valid_traces = [np.array(trace) for trace in traces if trace]  # Convert to array and check if not empty
        if valid_traces:  # Ensure there are valid traces before appending
            concatenated = np.concatenate(valid_traces, axis=0)  # Concatenate valid traces
            if concatenated.size > 0:  # Make sure concatenated results are not empty
                filtered_traces.append(concatenated)

    if not filtered_traces:
        raise ValueError("No valid traces found in the test data.")

    # Calculate the global min and max for normalization
    all_points = np.concatenate(filtered_traces, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Normalize and pad each trace
    normalized_padded_traces = []
    for traces in filtered_traces:
        normalized_traces = [(trace - min_vals) / (max_vals - min_vals) for trace in traces]
        max_len = max(len(trace) for trace in normalized_traces)
        padded_traces = pad_sequences(normalized_traces, maxlen=max_len, padding='post', dtype='float32')
        normalized_padded_traces.append(padded_traces)

    return np.array(normalized_padded_traces, dtype=object)


def predict(model, test_data):
    predictions = []
    for data in test_data:
        pred = model.predict(np.array([data]))
        predictions.append(pred)
    return predictions


# Call these functions
test_folder = 'test_inputs'
print("----------------LOAD TEST DATA --------------------")
raw_test_data = load_test_data(test_folder)
# Example usage:
try:

    print("----------------PREPROCESS TEST --------------------")
    preprocessed_test_data = preprocess_test_data(raw_test_data)


    print("----------------LOAD MODEL --------------------")
    # Load your trained model
    from tensorflow.keras.models import load_model
    model = load_model('model.h5')

    print("----------------PREDICT --------------------")

    # Make predictions
    test_predictions = predict(model, preprocessed_test_data)

    # Optionally, print or process predictions
    print(test_predictions)

    # Evaluate the model on the test inputs
    #test_loss, test_accuracy = model.evaluate(test_inputs)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Evaluate the model on the test set if you have separated some data for testing
    #test_loss, test_accuracy = model.evaluate(test_inputs, test_labels)
    #print(f"Test Accuracy: {test_accuracy}")


except ValueError as e:
    print(e)
