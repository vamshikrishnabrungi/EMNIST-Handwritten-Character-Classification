import os
import zipfile
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.losses import binary_crossentropy
from keras import backend as K

#Here we define dice co efficient which will be used in furthur 
def dice_coefficient(y_true, y_pred, smooth=1.0):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_coefficient_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)
#import the libraries

def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        #we  Calculate the binary cross-entropy
        bce = K.binary_crossentropy(y_true, y_pred)

        # here we Apply the weights
        weighted_bce = K.mean(weights * bce)

        return weighted_bce
    return loss

import os
import zipfile
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Extracting with the contents of the zip file
with zipfile.ZipFile('/content/Cam101.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')

# Defining  the train and test directories
train_dir = '/content/Cam101/train'
test_dir = '/content/Cam101/test'

# Geting a list of all the original and mask files in the train directory
train_files = os.listdir(train_dir)
train_orig_files = [filename for filename in train_files if '_L' not in filename]
train_mask_files = [filename for filename in train_files if '_L' in filename]

# Geting a list of all the original and mask files in the test directory
test_files = os.listdir(test_dir)
test_orig_files = [filename for filename in test_files if '_L' not in filename]
test_mask_files = [filename for filename in test_files if '_L' in filename]

# Counting the number of training and testing samples
num_train_samples = len(train_orig_files)
num_test_samples = len(test_orig_files)

# Printing the number of training and testing samples
print(f"Number of training samples: {num_train_samples}")
print(f"Number of testing samples: {num_test_samples}")


# Visualizing some random sample images from the training set
num_samples = 3
selected_train_images = random.sample(train_orig_files, num_samples)

fig = plt.figure(figsize=(20, 10))
for i in range(num_samples):
    # Geting the paths of the original and mask images
    orig_path = os.path.join(train_dir, selected_train_images[i])
    mask_path = os.path.join(train_dir, selected_train_images[i].replace('.png', '_L.png'))

    # Loading the images
    orig_img = plt.imread(orig_path)
    mask_img = plt.imread(mask_path)

    # Creating a subplot for the original image and another subplot for the mask image
    orig_ax = fig.add_subplot(2, num_samples, i+1)
    mask_ax = fig.add_subplot(2, num_samples, i+num_samples+1)

    # Displaying the original and mask images
    orig_ax.imshow(orig_img)
    mask_ax.imshow(mask_img)

    # Set the title of the subplots to the filename
    orig_ax.set_title(selected_train_images[i])
    mask_ax.set_title(selected_train_images[i].replace('.png', '_L.png'))

plt.show()


# Visualizing some random sample images from the testing set
num_samples = 3
selected_test_images = random.sample(test_orig_files, num_samples)

fig = plt.figure(figsize=(20, 10))
for i in range(num_samples):
    # Geting the paths of the original and mask images
    orig_path = os.path.join(test_dir, selected_test_images[i])
    mask_path = os.path.join(test_dir, selected_test_images[i].replace('.png', '_L.png'))

    # Loading the images
    orig_img = plt.imread(orig_path)
    mask_img = plt.imread(mask_path)

    # Creating a subplot for the original image and another subplot for the mask image
    orig_ax = fig.add_subplot(2, num_samples, i+1)
    mask_ax = fig.add_subplot(2, num_samples, i+num_samples+1)

    # Display the original and mask images
    orig_ax.imshow(orig_img)
    mask_ax.imshow(mask_img)


    # Seting the title of the subplots to the filename
    orig_ax.set_title(selected_test_images[i])
    mask_ax.set_title(selected_test_images[i].replace('.png', '_L.png'))


plt.show()

import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import shutil

# Defining the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.CenterCrop(height=256, width=256, p=0.5),
    A.Resize(height=128, width=128, p=1),
])

# Defining the path to the directory containing the images and masks
train_dir = '/content/Cam101/train'

# Defining the path to the directories where the augmented images and masks will be stored
aug_train_dir = '/content/Cam101/augmented_train'
aug_val_dir = '/content/Cam101/augmented_val'

# Creating the directories for the augmented images and masks
os.makedirs(aug_train_dir, exist_ok=True)
os.makedirs(aug_val_dir, exist_ok=True)

# Looping through all images in the training set and apply the augmentation pipeline
for filename in os.listdir(train_dir):
    if filename.endswith('.png') and not filename.endswith('_L.png'):
        image_path = os.path.join(train_dir, filename)
        mask_path = os.path.join(train_dir, filename[:-4] + '_L.png')
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path).astype('float32')
        
        # Applying the same augmentation pipeline to the image and mask
        augmented = transform(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']
        
        # Saving the augmented image and mask
        if np.random.rand() < 0.9:
            # Saving to the training set
            shutil.copy(image_path, os.path.join(aug_train_dir, filename))
            cv2.imwrite(os.path.join(aug_train_dir, filename[:-4] + '_L.png'), augmented_mask)
        else:
            # Saving to the validation set
            shutil.copy(image_path, os.path.join(aug_val_dir, filename))
            cv2.imwrite(os.path.join(aug_val_dir, filename[:-4] + '_L.png'), augmented_mask)
        
        # Display the original and augmented images side by side
        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Augmented Image')
        ax[2].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        ax[2].set_title('Original Mask')
        ax[3].imshow(cv2.cvtColor(augmented_mask, cv2.COLOR_BGR2RGB))
        ax[3].set_title('Augmented Mask')
        plt.show()
        
        # Printing the original image and mask names
        print('Original Image:', image_path)
        print('Original Mask:', mask_path)



import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import shutil

import skimage.transform
aug_train = '/content/Cam101/augmented_train/'

# Geting list of augmented training images
train_files = os.listdir(aug_train)
augmented_train = []
train_labels = []
for file in train_files:
    if file.endswith('.png') and not file.endswith('_L.png'):  # Only load original images (not masks)
        img = tf.keras.preprocessing.image.load_img(aug_train + file, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        augmented_train.append(img)
        
        # Geting corresponding label and append to train_labels list
        label_file = file.split('.')[0] + '_L.png'
        label = tf.keras.preprocessing.image.load_img(aug_train + label_file, target_size=(224, 224), color_mode='grayscale')
        label = tf.keras.preprocessing.image.img_to_array(label)
        label = label / 255.0  # Normalize pixel values
        train_labels.append(label)
        
augmented_train = tf.stack(augmented_train)
train_labels = tf.stack(train_labels)


aug_val = '/content/Cam101/augmented_val/'
# Geting list of augmented validation images
val_files = os.listdir(aug_val)

augmented_val = []
val_labels = []
for file in val_files:
    if not file.endswith('_L.png'):  # Exclude files ending with '_L.png'
        img = tf.keras.preprocessing.image.load_img(aug_val + file, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        augmented_val.append(img)
        
        # Geting corresponding label and append to val_labels list
        label_file = file.split('.')[0] + '_L.png'
        label = tf.keras.preprocessing.image.load_img(aug_val + label_file, target_size=(224, 224), color_mode='grayscale')
        label = tf.keras.preprocessing.image.img_to_array(label)
        label = label / 255.0  # Normalize pixel values
        val_labels.append(label)
        
augmented_val = tf.stack(augmented_val)
val_labels = tf.stack(val_labels)


import torch

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")


import time
start_time = time.time()

import tensorflow as tf

# defining the convolution block
def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    # first convolution
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    # secondconvolution
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

# defining the Unet model
def Unet(inputImage, numFilters=16, dropout=0.1, doBatchNorm=True, numClasses=1):
    # defining the encoder path
    a1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(a1)
    p1 = tf.keras.layers.Dropout(dropout)(p1)
    
    a2 = Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(a2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)
    
    a3 = Conv2dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(a3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)
    
    a4 = Conv2dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(a4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)
    
    a5 = Conv2dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)
    
    # defining the decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides=(2, 2), padding='same')(a5)
    u6 = tf.keras.layers.concatenate([u6, a4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    a6 = Conv2dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides=(2, 2), padding='same')(a6)
    
    u7 = tf.keras.layers.concatenate([u7, a3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    a7 = Conv2dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides=(2, 2), padding='same')(a7)
    u8 = tf.keras.layers.concatenate([u8, a2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    a8 = Conv2dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    # Decoder Path
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides=(2, 2), padding='same')(a8)
    u9 = tf.keras.layers.concatenate([u9, a1])
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    a9 = Conv2dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    # Output Layer
    output = tf.keras.layers.Conv2D(numClasses, (1, 1), activation='softmax')(a9)

    # Create and compile model
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model


from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from keras.metrics import Accuracy, Precision, Recall, AUC, MeanSquaredError
from keras.utils import plot_model
# Defining the optimizer and loss function
lr = 0.001
optimizer = RMSprop(learning_rate=lr)
loss_fn = dice_coefficient_loss
input_shape = (224, 224, 3) # Define the shape of the input image
num_classes = 1 # Define the number of classes
input_image = tf.keras.layers.Input(shape=input_shape) # Create an input layer for the model
model = Unet(input_image, numFilters=16, dropout=0.1, doBatchNorm=True, numClasses=num_classes) # Create the model using the input layer
# Print the hyperparameters
print(f"Learning rate: {lr}, Optimizer: {optimizer}, Loss function: {loss_fn}")
numClasses = 1
# Compiling the model with the optimizer and loss function, and add evaluation metrics
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[MeanIoU(numClasses), 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), tf.keras.metrics.MeanSquaredError()])

# Training the model
history = model.fit(augmented_train, train_labels, epochs=50, batch_size=16, validation_data=(augmented_val, val_labels))
# Save the model
model.save('my_unet_model.h5')
# Printing training and validation loss, and evaluation metrics
train_loss, train_mean_iou, train_acc, train_precision, train_recall, train_auc, train_mse = model.evaluate(augmented_train, train_labels, verbose=0)
val_loss, val_mean_iou, val_acc, val_precision, val_recall, val_auc, val_mse = model.evaluate(augmented_val, val_labels, verbose=0)

print(f"Training loss: {train_loss:.4f}, mean IoU: {train_mean_iou:.4f}, accuracy: {train_acc:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}, AUC: {train_auc:.4f}, MSE: {train_mse:.4f}")
print(f"Validation loss: {val_loss:.4f}, mean IoU: {val_mean_iou:.4f}, accuracy: {val_acc:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}, AUC: {val_auc:.4f}, MSE: {val_mse:.4f}")


import matplotlib.pyplot as plt

# Ploting the training and validation loss with respect to epoch
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Defining the optimizer and loss function for the second part of the code
lr2 = 0.01
optimizer2 = SGD(learning_rate=lr2)
loss_fn2 = weighted_binary_crossentropy(np.array([1, 5]))
input_shape2 = (224, 224, 3) # Define the shape of the input image
num_classes2 = 1 # Define the number of classes
input_image2 = tf.keras.layers.Input(shape=input_shape2) # Create an input layer for the model
model2 = Unet(input_image2, numFilters=16, dropout=0.1, doBatchNorm=True, numClasses=num_classes2) # Create the model using the input layer
# Printing the hyperparameters
print(f"Learning rate: {lr2}, Optimizer: {optimizer2}, Loss function: {loss_fn2}")
# Compile the model with the optimizer and loss function, and add evaluation metrics
model2.compile(optimizer=optimizer2, loss=loss_fn2, metrics=[MeanIoU(num_classes2), 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), tf.keras.metrics.MeanSquaredError()])

# Training the model
history2 = model2.fit(augmented_train, train_labels, epochs=50, batch_size=16, validation_data=(augmented_val, val_labels))

# Printing training and validation loss, and evaluation metrics
train_loss2, train_mean_iou2, train_acc2, train_precision2, train_recall2, train_auc2, train_mse2 = model2.evaluate(augmented_train, train_labels, verbose=0)
val_loss2, val_mean_iou2, val_acc2, val_precision2, val_recall2, val_auc2, val_mse2 = model2.evaluate(augmented_val, val_labels, verbose=0)

print(f"Training loss: {train_loss2:.4f}, mean IoU: {train_mean_iou2:.4f}, accuracy: {train_acc2:.4f}, precision: {train_precision2:.4f}, recall: {train_recall2:.4f}, AUC: {train_auc2:.4f}, MSE: {train_mse2:.4f}")
print(f"Validation loss: {val_loss2:.4f}, mean IoU: {val_mean_iou2:.4f}, accuracy: {val_acc2:.4f}, precision: {val_precision2:.4f}, recall: {val_recall2:.4f}, AUC: {val_auc2:.4f}, MSE: {val_mse2:.4f}")




# Defining the optimizer and loss function for the second part of the code
lr3 = 0.1
optimizer3 = Adam(learning_rate=lr3)
loss_fn3 = BinaryCrossentropy()
input_shape3 = (224, 224, 3) # Define the shape of the input image
num_classes3 = 1 # Define the number of classes
input_image3 = tf.keras.layers.Input(shape=input_shape3) # Create an input layer for the model
model3 = Unet(input_image3, numFilters=16, dropout=0.1, doBatchNorm=True, numClasses=num_classes3) # Create the model using the input layer
# Printing the hyperparameters
print(f"Learning rate: {lr3}, Optimizer: {optimizer3}, Loss function: {loss_fn3}")
# Compile the model with the optimizer and loss function, and add evaluation metrics
model3.compile(optimizer=optimizer3, loss=loss_fn3, metrics=[MeanIoU(num_classes3), 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), tf.keras.metrics.MeanSquaredError()])

# Training the model
history3 = model3.fit(augmented_train, train_labels, epochs=50, batch_size=16, validation_data=(augmented_val, val_labels))

# Printing training and validation loss, and evaluation metrics
train_loss3, train_mean_iou3, train_acc3, train_precision3, train_recall3, train_auc3, train_mse3 = model3.evaluate(augmented_train, train_labels, verbose=0)
val_loss3, val_mean_iou3, val_acc3, val_precision3, val_recall3, val_auc3, val_mse3 = model3.evaluate(augmented_val, val_labels, verbose=0)

print(f"Training loss: {train_loss3:.4f}, mean IoU: {train_mean_iou3:.4f}, accuracy: {train_acc3:.4f}, precision: {train_precision3:.4f}, recall: {train_recall3:.4f}, AUC: {train_auc3:.4f}, MSE: {train_mse3:.4f}")
print(f"Validation loss: {val_loss3:.4f}, mean IoU: {val_mean_iou3:.4f}, accuracy: {val_acc3:.4f}, precision: {val_precision3:.4f}, recall: {val_recall3:.4f}, AUC: {val_auc3:.4f}, MSE: {val_mse3:.4f}")

end_time = time.time()
training_time = end_time - start_time
print("Training time: " + str(training_time) + " seconds")

start_time2 = time.time()

import tensorflow as tf

def build_pspnet(input_image, num_classes):
    # Defining the input layer.
    input_layer = input_image
    # Creating the encoder.
    encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    ])(input_layer)

    # Creating the decoder.
    decoder = tf.keras.Sequential([
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(encoder),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
        tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same'),
    ])

    # Defining the output layer.
    output_layer = decoder(encoder)

    # Defining the model.
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model



learning_rates = [0.001, 0.01, 0.1]
optimizers = [Adam, SGD, RMSprop]
loss_functions = [binary_crossentropy, weighted_binary_crossentropy(np.array([1, 5])), dice_coefficient_loss]

for lr in learning_rates:
    for optimizer in optimizers:
        for loss_fn in loss_functions:
            # Defining the model and compile it with the current hyperparameters
            optimizer_instance = optimizer(learning_rate=lr)



            input_shape = (224, 224, 3) # Define the shape of the input image
            num_classes = 1 # Define the number of classes
            input_image = tf.keras.layers.Input(shape=input_shape) # Create an input layer for the model
            model = Unet(input_image, numFilters=16, dropout=0.1, doBatchNorm=True, numClasses=num_classes)
            model.compile(optimizer=optimizer_instance, loss=loss_fn, metrics=[MeanIoU(num_classes), 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), tf.keras.metrics.MeanSquaredError()])
            
            # Training and evaluate the model on the training and validation sets
            history = model.fit(augmented_train, train_labels, epochs=3, batch_size=16, validation_data=(augmented_val, val_labels), verbose=0)
            train_loss, train_mean_iou, train_acc, train_precision, train_recall, train_auc, train_mse = model.evaluate(augmented_train, train_labels, verbose=0)
            val_loss, val_mean_iou, val_acc, val_precision, val_recall, val_auc, val_mse = model.evaluate(augmented_val, val_labels, verbose=0)
            
            # Printing the hyperparameters and evaluation metrics
            print(f"Learning rate: {lr}, Optimizer: {optimizer}, Loss function: {loss_fn}")
            print(f"Training loss: {train_loss:.4f}, mean IoU: {train_mean_iou:.4f}, accuracy: {train_acc:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}, AUC: {train_auc:.4f}, MSE: {train_mse:.4f}")
            print(f"Validation loss: {val_loss:.4f}, mean IoU: {val_mean_iou:.4f}, accuracy: {val_acc:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}, AUC: {val_auc:.4f}, MSE: {val_mse:.4f}")
end_time2 = time.time()
training_time2 = end_time - start_time
print("Training time for 2nd moel : " + str(training_time2) + " seconds")





start_time3 = time.time()
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from keras.losses import binary_crossentropy


from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential


import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import shutil


# Defining the FCN architecture

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=1, activation='sigmoid'))

    return model


from skimage.transform import resize
import time
start_time = time.time()
learning_rate = 0.1
optimizer1 = Adam
loss_function = binary_crossentropy

# Resizing the labels to match the input data dimensions
train_labels_resized = np.array([resize(label, (208, 208), anti_aliasing=True, mode='reflect') for label in train_labels])
val_labels_resized = np.array([resize(label, (208, 208), anti_aliasing=True, mode='reflect') for label in val_labels])

# Initializing and compile the model
model11 = create_model()
model11.compile(optimizer=optimizer1(lr=learning_rate), loss=loss_function, metrics=[MeanIoU(num_classes=2), Accuracy()])


print(f"Training model with LR={learning_rate}, Optimizer={optimizer1._name}, Loss function={loss_function.name_}")
# Training the model on 90% of the data
history11 = model11.fit(augmented_train, train_labels_resized, batch_size=32, epochs=50, verbose=1, validation_split=0.1)


# Evaluating the model on the remaining 10% of the data
val_loss, val_mean_iou, val_pixel_acc = model11.evaluate(augmented_val, val_labels_resized, verbose=0)
train_loss, train_mean_iou, train_pixel_acc = model11.evaluate(augmented_train, train_labels_resized, verbose=0)
end_time = time.time()  # end time
training_time = end_time - start_time  # total training time
# Printing the results for this hyperparameter combination
print(f'LR={learning_rate}, Optimizer={optimizer1._name}, Loss function={loss_function.name_}, '
      f'Training Loss={train_loss}, Training Mean IoU={train_mean_iou}, Training Pixel accuracy={train_pixel_acc}, '
      f'Validation Loss={val_loss}, Validation Mean IoU={val_mean_iou}, Validation Pixel accuracy={val_pixel_acc}, ' 
      f'Training time: {training_time} seconds')



learning_rate2 = 0.01
optimizer12 = SGD
loss_function2 = binary_crossentropy

# Resizing the labels to match the input data dimensions
train_labels_resized = np.array([resize(label, (208, 208), anti_aliasing=True, mode='reflect') for label in train_labels])
val_labels_resized = np.array([resize(label, (208, 208), anti_aliasing=True, mode='reflect') for label in val_labels])

# Initialize and compile the model
model12 = create_model()
model12.compile(optimizer=optimizer12(lr=learning_rate2), loss=loss_function2, metrics=[MeanIoU(num_classes=2), Accuracy()])


print(f"Training model with LR={learning_rate2}, Optimizer={optimizer12._name}, Loss function={loss_function2.name_}")
# Training the model on 90% of the data
history12 = model12.fit(augmented_train, train_labels_resized, batch_size=32, epochs=50, verbose=1, validation_split=0.1)


# Evaluating the model on the remaining 10% of the data
val_loss, val_mean_iou, val_pixel_acc = model12.evaluate(augmented_val, val_labels_resized, verbose=0)
train_loss, train_mean_iou, train_pixel_acc = model12.evaluate(augmented_train, train_labels_resized, verbose=0)
end_time = time.time()  # end time
training_time = end_time - start_time  # total training time
# Printing the results for this hyperparameter combination
print(f'LR={learning_rate2}, Optimizer={optimizer12._name}, Loss function={loss_function2.name_}, '
      f'Training Loss={train_loss}, Training Mean IoU={train_mean_iou}, Training Pixel accuracy={train_pixel_acc}, '
      f'Validation Loss={val_loss}, Validation Mean IoU={val_mean_iou}, Validation Pixel accuracy={val_pixel_acc}, ' 
      f'Training time: {training_time} seconds')

learning_rate3 = 0.001
optimizer13 = RMSprop
loss_function3 = dice_coefficient_loss

# Resizing the labels to match the input data dimensions
train_labels_resized = np.array([resize(label, (208, 208), anti_aliasing=True, mode='reflect') for label in train_labels])
val_labels_resized = np.array([resize(label, (208, 208), anti_aliasing=True, mode='reflect') for label in val_labels])

# Initialize and compile the model
model13 = create_model()
model13.compile(optimizer=optimizer13(lr=learning_rate3), loss=loss_function3, metrics=[MeanIoU(num_classes=2), Accuracy()])


print(f"Training model with LR={learning_rate3}, Optimizer={optimizer13._name}, Loss function={loss_function3.name_}")
# Training the model on 90% of the data
history13 = model13.fit(augmented_train, train_labels_resized, batch_size=32, epochs=50, verbose=1, validation_split=0.1)


# Evaluating the model on the remaining 10% of the data
val_loss, val_mean_iou, val_pixel_acc = model13.evaluate(augmented_val, val_labels_resized, verbose=0)
train_loss, train_mean_iou, train_pixel_acc = model13.evaluate(augmented_train, train_labels_resized, verbose=0)
end_time = time.time()  # end time
training_time = end_time - start_time  # total training time
# Printing the results for this hyperparameter combination
print(f'LR={learning_rate3}, Optimizer={optimizer13._name}, Loss function={loss_function3.name_}, '
      f'Training Loss={train_loss}, Training Mean IoU={train_mean_iou}, Training Pixel accuracy={train_pixel_acc}, '
      f'Validation Loss={val_loss}, Validation Mean IoU={val_mean_iou}, Validation Pixel accuracy={val_pixel_acc}, ' 
      f'Training time: {training_time} seconds')


import matplotlib.pyplot as plt

# Ploting the training and validation loss over epochs
plt.plot(history13.history['loss'], label='Training Loss')
plt.plot(history13.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
import matplotlib.pyplot as plt


end_time3 = time.time()
training_time3 = end_time3 - start_time3
print("Training time for model 3 : " + str(training_time3) + " seconds")


#Here we test the models 

print("the ploting after testing")




import os
import matplotlib.pyplot as plt

# Visualizing some random sample images from the testing set
num_samples = 6
selected_test_images = random.sample(test_orig_files, num_samples)

fig = plt.figure(figsize=(50, 30))
for i in range(num_samples):
    # Geting the paths of the original and mask images
    orig_path = os.path.join(test_dir, selected_test_images[i])
    mask_path = os.path.join(test_dir, selected_test_images[i].replace('.png', '_L.png'))

    # Loading the images
    orig_img = plt.imread(orig_path)
    mask_img = plt.imread(mask_path)

    # Creating a subplot for the original image and another subplot for the mask image
    orig_ax = fig.add_subplot(2, num_samples, i+1)
    mask_ax = fig.add_subplot(2, num_samples, i+num_samples+1)

    # Display the original and mask images
    orig_ax.imshow(orig_img)
    mask_ax.imshow(mask_img)


    # Seting the title of the subplots to the filename
    orig_ax.set_title("orginal image")
    mask_ax.set_title("orginal mask")


plt.show()



import os
import matplotlib.pyplot as plt
import numpy as np

# Define the brightness and gamma values for adjusting the contrast
brightness = -0.8
gamma = 1.4

# Visualizing some random sample images from the testing set
num_samples = 6
selected_test_images = random.sample(test_orig_files, num_samples)

fig = plt.figure(figsize=(50, 30))
for i in range(num_samples):
    # Geting the paths of the original and mask images
    orig_path = os.path.join(test_dir, selected_test_images[i])
    mask_path = os.path.join(test_dir, selected_test_images[i].replace('.png', '_L.png'))

    # Loading the images
    orig_img = plt.imread(orig_path)
    mask_img = plt.imread(mask_path)

    # Adjusting the contrast of the images
    orig_img = np.power(orig_img, gamma)
    orig_img = np.clip(orig_img + brightness, 0, 1)
    mask_img = np.power(mask_img, gamma)
    mask_img = np.clip(mask_img + brightness, 0, 1)

    # Creating a subplot for the original image and another subplot for the mask image
    orig_ax = fig.add_subplot(2, num_samples, i+1)
    mask_ax = fig.add_subplot(2, num_samples, i+num_samples+1)

    # Display the original and mask images
    orig_ax.imshow(orig_img)
    mask_ax.imshow(mask_img)

    # Seting the title of the subplots to the filename
    orig_ax.set_title("predicted image")
    mask_ax.set_title("image mask")

plt.show()