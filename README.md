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
4. Upload the entire model folder to your Google Drive at: `/MyDrive/ndev-task-tracker/universal-sentence-encoder-tensorflow1-lite-v2`

The model folder structure should look like:
```
/content/drive/MyDrive/ndev-task-tracker/universal-sentence-encoder-tensorflow1-lite-v2/
├── saved_model.pb
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/
    └── universal_encoder_8k_spm.model
```

### 2. Required Dependencies

The notebook uses the following Python packages (automatically installed in Colab):
- TensorFlow
- TensorFlow Hub
- TensorFlow Text
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

After downloading the model, you can convert it to TensorFlow.js format for web deployment:

### Prerequisites for Conversion
Install TensorFlow.js converter on your local machine:
```bash
pip install tensorflowjs
```

### Convert the Model
Run the following command to convert the Keras model to TensorFlow.js format:
```bash
tensorflowjs_converter --input_format=keras saved_model.h5 classifier
```

This will create a `classifier` directory containing:
- `model.json`: Model architecture
- `*.bin` files: Model weights

## Usage Example

After training, you can classify new text:

```python
new_sentences = [
    "writing code", 
    "eating a little pizza for a minute and wrote a code", 
    "catchup with mr x"
]

# Get predictions
new_input = to_sparse(new_sentences)
new_embeddings = embed_fn(**new_input)['default']
predictions = model.predict(new_embeddings)

# Get predicted classes
predicted_labels = predictions.argmax(axis=1)
predicted_class_names = [class_names[label] for label in predicted_labels]
confidences = predictions.max(axis=1)

print("Predictions:", predicted_class_names)
print("Confidences:", confidences)
```

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

## Troubleshooting

### Common Issues:

1. **Model Path Error**: Ensure the Universal Sentence Encoder model is uploaded to the correct Google Drive path
2. **Memory Issues**: If you encounter memory problems, try reducing the dataset size or using a smaller batch size
3. **Poor Accuracy**: Consider adding more training examples or adjusting the model architecture

### File Paths:
- Model path in Colab: `/content/drive/MyDrive/ndev-task-tracker/universal-sentence-encoder-tensorflow1-lite-v2`
- Saved model output: `saved_model.h5` (in Colab's current directory)

## License

This project is intended for educational and development purposes. Please ensure you comply with the Universal Sentence Encoder model's license terms when using it.
