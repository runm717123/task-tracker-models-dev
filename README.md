# Task Tracker Title Classifier

A machine learning model to classify task titles into different categories using Universal Sentence Encoder and TensorFlow. This classifier helps distinguish between valid project titles, meetings, background tasks, general activities, and project-specific tasks.

## Model Categories

The classifier categorizes text into 6 classes:
- **valid_title**: Project names and sprint titles
- **background_task**: Infrastructure and setup tasks
- **meetings**: Meeting-related activities
- **general_tasks**: General development tasks
- **general_activities**: Non-work activities
- **project_tasks**: Specific coding tasks

## Setup and Prerequisites

### 1. Download Universal Sentence Encoder Model

Before running the notebook, you need to download the Universal Sentence Encoder model:

1. Go to [Kaggle Universal Sentence Encoder](https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow1/lite/2)
2. Download the **TensorFlow1 Lite v2** model
3. Extract the downloaded files
4. Upload the entire model folder to your Google Drive at: `ndev-task-tracker/universal-sentence-encoder-tensorflow1-lite-v2`

The notebook uses the following Python packages (automatically installed in Colab):
- TensorFlow
- TensorFlow Hub
- NumPy
- SentencePiece
- Scikit-learn

## Running the Notebook in Google Colab

### Step 1: Mount Google Drive
Run the first cell to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Environment Setup
Execute the environment configuration cell to set up TensorFlow for legacy Keras compatibility.
this is required only if you want to convert this model for later use in web app using tensorflowjs[wizard]

### Step 3: Load Models and Dependencies
Run the "Import & Model load" section to:
- Load the Universal Sentence Encoder model from your Google Drive
- Initialize the SentencePiece tokenizer
- Import required libraries

### Step 4: Prepare Data
The notebook includes predefined datasets for each category. You can:
- Review the existing data in the "Data Preparation" section
- **Adjust the datasets** by modifying the lists in the data preparation cell if needed
- Add or remove examples from any category to improve model performance
- adjust `random_state` and check the y_val, its better if y_val distributed evenly 

### Step 5: Train the Model
Execute the "Training" section to:
- Preprocess the text data
- Generate embeddings using Universal Sentence Encoder
- Split data into training and validation sets
- Train the neural network classifier

### Step 6: Evaluate Performance
Run the "evaluation" section to:
- Test the model on validation data
- See accuracy metrics
- Test predictions on sample sentences

### Step 7: Save the Model
Execute the "Save Model" section to save the trained model as `saved_model.h5`.

### Step 8: Download the Model
After training, download the saved model file:
1. In Colab, go to the Files panel (folder icon on the left)
2. Locate `saved_model.h5`
3. Right-click and select "Download"

## Model Conversion for Web Deployment

After downloading the model, you can convert it to TensorFlow.js format for web deployment
for more read the official docs [here](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#regular-conversion-script-tensorflowjs_converter)

### Convert the Model
Run the following command to convert the Keras model to TensorFlow.js format:
```bash
tensorflowjs_converter --input_format=keras saved_model.h5 classifier
```

This will create a `classifier` directory containing:
- `model.json`: Model architecture
- `*.bin` files: Model weights

## Model Architecture

- **Input**: 512-dimensional sentence embeddings from Universal Sentence Encoder Lite
- **Hidden Layer**: 50 neurons with ReLU activation
- **Dropout**: 0.2 for regularization
- **Output**: 6 classes with softmax activation
- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy

## Tips for Better Performance

1. **Data Quality**: Ensure your training data represents real-world usage patterns
2. **Data Balance**: Try to have roughly equal examples for each category
3. **Validation**: Test with diverse examples to ensure good generalization
4. **Hyperparameter Tuning**: Experiment with different architectures, learning rates, and epochs

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
See the [LICENSE](./LICENSE) file for details.
