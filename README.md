EMNIST Classification Project

Dataset Overview

The EMNIST "balanced" dataset consists of 112,800 training samples and 18,800 testing samples across 47 classes. Images are 28 x 28 pixels, grayscale, and centered. Each subset, split for cross-validation, is normalized with pixel values ranging from 0 to 255. This MNIST-style dataset can be loaded using PyTorch's DataLoader class.

CNN Model Architecture

The project implements a Convolutional Neural Network (CNN) for EMNIST classification. The CNN class, a subclass of nn.Module, initializes layers, including two convolutional layers, batch normalization, max pooling, and fully connected layers. ReLU activation and CrossEntropy loss functions are used. Adam optimizer, learning rate (0.001), L1 regularization (1e-5), batch size (64), and 10 epochs are set. KFold cross-validation (k=3) is employed, training the CNN on subsets and evaluating on validation sets.

MLP Model

The code defines a Multi-Layer Perceptron (MLP) using PyTorch's nn.Module class. It consists of three fully connected layers with ReLU activation and optional dropout to prevent overfitting. K-Fold cross-validation is applied to train and evaluate the MLP.

Hyperparameter Tuning

Hyperparameters tuned include activation function, optimizer, batch normalization, dropout, regularization, epochs, batch size, and learning rate. The rationale is to optimize model performance by addressing issues like overfitting and underfitting.

Regularization Techniques

To counter overfitting, L1 regularization and dropout are implemented. L1 regularization adds a penalty term to encourage smaller weights, while dropout randomly drops neurons during training. Multiple layers in the neural network address underfitting, and adaptive learning rates and activation functions (ReLU, Leaky ReLU, ELU) enhance non-linear learning.

Training and Evaluation

Training and validation accuracy, along with losses, are printed for each epoch. Best hyperparameters are identified based on performance metrics. CNN consistently outperforms MLP, demonstrating better spatial feature extraction and hierarchical representation learning.

Results and Comparison

CNN achieves higher testing accuracy and lower testing loss compared to MLP. It outperforms MLP due to its ability to capture spatial information, reduce parameters, and learn complex patterns. Precision, F1 score, accuracy, and recall scores consistently favor CNN.

In conclusion,the project successfully classifies EMNIST data using CNN and MLP models. Hyperparameter tuning and regularization techniques enhance model performance. The comparative analysis supports the effectiveness of CNN for image classification tasks.
