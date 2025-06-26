🧠 CNN-Based Image Classifier for Fruits and Vegetables

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images of fruits and vegetables into multiple categories, including detecting rotten vs. fresh variants. The dataset is prepared, split, and trained in a reproducible and modular pipeline.

📁 Project Structure

├── finalocrmodel.ipynb         
├── model.h5                     
├── README.md     
├── Output Samples

🛠️ Requirements
Install the necessary Python packages before running the notebook:

pip install tensorflow matplotlib opencv-python pillow

📥 Dataset
The dataset is not included in this repository due to size limitations. You can download it from Kaggle:

https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset

After downloading, unzip the archive and organize the dataset folder as follows:

archive(2)/archive/
├── Apple/
├── RottenApple/
├── Mango/
├── RottenMango/
├── ...

You can then use the provided helper function split_dataset() to divide the data into train/, val/, and test/ folders.

📊 Dataset Preparation
The dataset is split into:

80% Training

10% Validation

10% Testing

The split_dataset() function handles the splitting and directory structuring.

🧹 Data Cleaning
To ensure data quality, invalid or corrupted images are filtered out using:

OpenCV image reading validation

imghdr for verifying image file types (jpeg, png, jpg, etc.)

🧠 Model Architecture
The CNN model consists of:

Input normalization

3 convolutional layers with ReLU activation and max pooling

Flattening and dropout layers

Dense layers leading to output neurons for each class

model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(num_classes)
])

Loss Function: SparseCategoricalCrossentropy(from_logits=True)
Optimizer: Adam
Metrics: Accuracy

🏋️ Model Training
Train the model for 25 epochs with:

Datasets loaded using image_dataset_from_directory()

Real-time validation on the validation set

Visualizing training and validation accuracy and loss with Matplotlib

📈 Sample Visualization

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.show()

🔍 Prediction on New Images
To predict the class of a new image:

image_path = 'test/RottenMango/rottenMango (117).jpg'
image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
img_array = tf.keras.utils.img_to_array(image)
img_batch = tf.expand_dims(img_array, 0)

predictions = model.predict(img_batch)
score = tf.nn.softmax(predictions)

print("Image is {} with {:0.2f}% confidence.".format(
    data_cat[np.argmax(score)],
    100 * np.max(score)
))

Use Case

This CNN-based fruits and vegetables classifier can be applied in multiple practical scenarios. For instance, grocery stores and packaging facilities can use it to automatically detect and separate fresh produce from rotten items, improving quality control and reducing food waste. Additionally, the model can be integrated into mobile applications, allowing consumers to quickly check the freshness of fruits and vegetables before purchase, promoting healthier choices. Overall, this AI-driven solution helps streamline supply chains, minimize spoilage, and empower end-users with instant, reliable freshness detection.

"# CNN-model-for-fruit-and-veg-detection-" 
