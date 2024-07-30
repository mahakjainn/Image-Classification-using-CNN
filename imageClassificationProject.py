#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10


# In[2]:


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


# In[3]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[4]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[6]:


train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


# In[7]:


train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[9]:


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# In[10]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))

# Step 5: Model Evaluation


# In[12]:


#Model Evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")


# In[13]:


from sklearn.metrics import classification_report
import numpy as np


# In[14]:


predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

print(classification_report(true_classes, predicted_classes))


# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_images)


# In[18]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) // 32, epochs=20,
                    validation_data=(val_images, val_labels))


# In[19]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[20]:


def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the class
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return class_names[class_index]


# In[24]:


model.save('saved_model.keras')


# In[ ]:import matplotlib.pyplot as plt
import numpy as np

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()




