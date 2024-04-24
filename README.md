# Classification-of-Downs-Syndrome-in-Children
Down Syndrome Detection with CNNs
Introduction
This project aims to detect Down syndrome using image data. It employs Convolutional Neural Networks (CNNs) for classification and evaluates model performance on a dataset containing images of individuals with and without Down syndrome.

Installation Instructions
To run the code, follow these steps:

Install TensorFlow and other dependencies:
pip install tensorflow visualkeras opencv-python-headless tqdm matplotlib pandas

Dataset Description
The dataset consists of images categorized into two classes: healthy and Down syndrome. Images are stored in separate directories for each class.
https://www.kaggle.com/datasets/mervecayli/detection-of-down-syndrome-in-children

Code Overview
Data Preparation: Images are loaded, resized, and preprocessed. The dataset is split into training, validation, and test sets.
Data Augmentation: ImageDataGenerator is used for data augmentation to enhance model generalization.
Model Architecture: A CNN model is constructed with convolutional, pooling, and fully connected layers.
Training: The model is trained using the training set with early stopping and model checkpointing callbacks.
Evaluation: Model performance is evaluated on the test set using metrics like accuracy and the confusion matrix.
Transfer Learning: Transfer learning with MobileNetV2 is explored for comparison.

Results
Both CNN and MobileNetV2 models achieved promising results in Down syndrome detection. CNN achieved an accuracy of 69% on the test set, while MobileNetV2 achieved an accuracy of 82%.

Conclusion
In conclusion, CNNs show potential for detecting Down syndrome from image data. Further optimizations and exploration of advanced architectures could enhance model performance.

Usage
Run train.py to train the CNN model.
Run evaluate.py to evaluate the model on the test set.

References
TensorFlow Documentation: https://www.tensorflow.org/
Visualkeras Documentation: https://github.com/paulgavrikov/visualkeras
