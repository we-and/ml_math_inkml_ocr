import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


from lxml import etree
import numpy as np

def parse_inkml(inkml_path):
    tree = etree.parse(inkml_path)
    namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
    
    traces = []
    for trace in tree.xpath('//inkml:trace', namespaces=namespaces):
        points = []
        for point in trace.text.strip().split(','):
            x, y = point.strip().split(' ')
            points.append((float(x), float(y)))
        traces.append(points)
    
    return traces

# Example usage
inkml_path = 'path_to_your_inkml_file.inkml'
strokes = parse_inkml(inkml_path)

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Normalize and pad stroke data
def normalize_and_pad(strokes):
    # Flatten the list of strokes and normalize
    all_points = np.concatenate(strokes, axis=0)
    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)
    normalized_strokes = [(np.array(stroke) - min_vals) / (max_vals - min_vals) for stroke in strokes]
    
    # Pad sequences for uniform input size
    padded_strokes = pad_sequences(normalized_strokes, padding='post', dtype='float32')
    return padded_strokes

padded_strokes = normalize_and_pad(strokes)



# Mock function to load your data
def load_data():
    # This function should return your stroke data and labels
    # For example:
    # X = [array of [array of [x1, y1], [x2, y2], ..., [xn, yn]] for each sample]
    # y = [corresponding label for each sample]
    return np.random.rand(1000, 10, 2), np.random.randint(0, 2, (1000,))

# Load your preprocessed stroke data and labels
X, y = load_data()

# Pad sequences for uniform input size
X_padded = pad_sequences(X, padding='post', dtype='float32')

# Convert labels to categorical
y_categorical = tf.keras.utils.to_categorical(y)

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

model = build_model((None, 2), y_categorical.shape[1])
model.summary()

model.fit(X_padded, y_categorical, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_padded, y_categorical)
print(f"Model accuracy: {accuracy}")

# Use the model to predict new data
def predict_new_data(new_strokes):
    new_strokes_padded = pad_sequences([new_strokes], maxlen=X_padded.shape[1], padding='post', dtype='float32')
    prediction = model.predict(new_strokes_padded)
    return np.argmax(prediction)

# Example usage
new_strokes = np.array([[1, 2], [3, 4], [5, 6]])  # Example new stroke data
prediction = predict_new_data(new_strokes)
print(f"Predicted class: {prediction}")