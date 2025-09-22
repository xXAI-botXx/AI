# TensorFlow / Keras

[<img align="right" width=150px src='../res/rackete_2.png'></img>](../README.md)

Table of Contents:
- [Fundamentals](#fundamentals)
- [Workflow](#workflow)
- [Classification](#classification)
- [Computer Vision](#computer-vision)
- [Notebooks and Python Scripts](#notebooks-and-python-scripts)
- [Transfer Learning](#transfer-learning)
- [Experiment Tracking](#experiment-tracking)
- [Paper Replicating](#paper-replicating)
- [Deployment](#deployment)
- [Tips](#tips)

<br>

> First base implementation. Add more content/details in future.


<br><br>

---
### Fundamentals

TensorFlow is a **high-level deep learning library** with support for both **eager execution** and **graph mode**. Keras is its high-level API for building neural networks.

<br><br>

**Installation**

```bash
pip install tensorflow
```

<br><br>

**Tensors**

```python
import tensorflow as tf

# Create a tensor
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

# Random tensor
y = tf.random.uniform((2, 2))

# Operations
z = x + y
z = tf.matmul(x, y, transpose_b=True)

# Convert to NumPy
np_array = z.numpy()
```

<br><br>

**Automatic Differentiation**

```python
x = tf.Variable([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    y = x**2 + 2*x
grad = tape.gradient(y, x)
print(grad)  # dy/dx
```

<br><br>

---
### Workflow

Typical TensorFlow workflow:
1. **Data Preparation** – Use `tf.data.Dataset` to load and preprocess data.
2. **Model Definition** – Build a model with `tf.keras.Model` or `Sequential`.
3. **Loss & Optimizer** – Choose a loss function and optimizer.
4. **Training Loop** – Either `model.fit()` or custom `GradientTape` loops.
5. **Evaluation** – Test your model and compute metrics.

```python
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=100).batch(16)
```



<br><br>

---
### Classification

<br><br>

**Simple Feedforward Neural Network**

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16)
```


<br><br>

---
### Computer Vision

TensorFlow provides `tf.keras.preprocessing.image` and `tf.data pipelines` for images.

<br><br>

**Dataset & Preprocessing**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    "data/train", target_size=(128, 128), batch_size=32, subset='training'
)
val_gen = datagen.flow_from_directory(
    "data/train", target_size=(128, 128), batch_size=32, subset='validation'
)
```

<br><br>

**Convolutional Neural Network**

```python
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_gen, validation_data=val_gen, epochs=10)
```


<br><br>

---
### Notebooks and Python Scripts


- **Jupyter Notebooks:** Great for experimentation and visualizations.
- **Python scripts:** For reproducible pipelines.

```python
# Save and load model
cnn_model.save("cnn_model.h5")
loaded_model = models.load_model("cnn_model.h5")
```


<br><br>

---
### Transfer Learning

Leverage pretrained models like `VGG16`, `ResNet50`, etc.

```python
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(10, activation='softmax')(x)

model = models.Model(base_model.input, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```



<br><br>

---
### Experiment Tracking

- **TensorBoard** (built-in)
    ```python
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    model.fit(X, y, epochs=10, callbacks=[tensorboard_cb])
    ```
- **Weights & Biases (wandb)**
    ```python
    import wandb
    from wandb.keras import WandbCallback

    wandb.init(project="my-tf-project")
    model.fit(X, y, epochs=10, callbacks=[WandbCallback()])
    ```


<br><br>

---
### Paper Replicating

Steps to replicate research:
1. Understand the architecture and preprocessing.
2. Implement the network in TensorFlow.
3. Match training and evaluation metrics.
4. Compare results and debug differences.

```python
# Example: Custom loss and optimizer as described in a paper
custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```





<br><br>

---
### Deployment

- **SavedModel Format** (recommended)
    ```python
    model.save("saved_model/my_model")
    loaded_model = tf.keras.models.load_model("saved_model/my_model")
    ```
- **TensorFlow Lite** (for mobile/edge devices)
    ```python
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    ```
- **TensorFlow Serving:** Production-ready serving.
- **Flask/FastAPI:** Wrap model in REST API.


<br><br>

---
### Tips

- TensorFlow 2.x uses **eager execution by default**, making it more Pythonic.
- Use `tf.data` pipelines for **efficient data loading**.
- Keras `Sequential` or `Functional` API works for most use-cases.
- TensorBoard is your friend for **visualizing metrics and training curves**.


---
