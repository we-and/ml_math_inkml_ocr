# Handwritten Math Expression Recognition
This project focuses on recognizing handwritten mathematical expressions using deep learning techniques. It utilizes the CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions) dataset, which consists of handwritten mathematical expressions in InkML format.
## Dataset
The CROHME dataset is used for training and evaluating the model. The dataset contains handwritten mathematical expressions in InkML format, along with their corresponding LaTeX representations. The InkML files are parsed to extract the stroke information and labels for each expression.

### PNG Input example (pre-preprecessing)
![alt text](https://github.com/we-and/ml_math_inkml_ocr/blob/main/test_inputs_png/129_em_542.png?raw=true)
### Reading labels from inkml
![alt text](https://github.com/we-and/ml_math_inkml_ocr/blob/main/screenshot.png?raw=true)


## Preprocessing
The preprocessing steps include:

Parsing InkML files: The InkML files are parsed to extract the stroke information and labels for each expression. The parse_inkml function is used to read the InkML files and extract the relevant data.
Normalizing and padding traces: The stroke sequences are normalized to a fixed range and padded to a uniform length using the normalize_and_pad_traces function. This step ensures consistent input dimensions for the model.
Encoding labels: The labels are encoded using the LabelEncoder from scikit-learn. The encode_labels function performs label encoding and converts the labels to categorical format.

## Model Architecture
The project uses a deep learning model based on Long Short-Term Memory (LSTM) networks. The model architecture consists of the following layers:

LSTM layer with 128 units and return sequences set to True
Dropout layer with a rate of 0.2
LSTM layer with 64 units
Dropout layer with a rate of 0.2
Dense layer with 64 units and ReLU activation
Dense output layer with softmax activation

The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
## Training
The model is trained using the preprocessed data. The training process involves:

Building the model using the build_model function, which takes the input shape and the number of classes as parameters.
Fitting the model using the model.fit function, specifying the number of epochs, batch size, and validation split.
Evaluating the trained model on a separate test set using the model.evaluate function.

## Usage
To use this project:

Clone the repository: git clone https://github.com/your-username/handwritten-math-expression-recognition.git
Install the required dependencies: pip install -r requirements.txt
Prepare your dataset in InkML format and place it in the appropriate directory.
Run the crohme.py script to train the model: python crohme.py
Evaluate the trained model on a test set and analyze the results.

## Future Improvements
Some potential areas for future improvements include:

Experimenting with different model architectures and hyperparameters to improve recognition accuracy.
Implementing data augmentation techniques to enhance the model's robustness.
Exploring additional preprocessing techniques to handle variations in handwriting styles.
Integrating the trained model into a user-friendly application for real-time handwritten math expression recognition.

