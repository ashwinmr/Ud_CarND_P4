
# coding: utf-8

# # Imports

# In[2]:


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Dropout, Activation


# # Read csv and load images

# In[3]:


correction = 0.2 # Correction for side cameras
images = []
measurements = []

# Open the csv file
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for row in reader:
        # Append center left and right images and augment with flipped images
        for i in range(3):
            image_path = 'data/IMG/' + row[i].split('/')[-1]
            image_bgr = cv2.imread(image_path)
            image = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
            
            # Get flipped images
            image_flipped = cv2.flip(image,1)
            images.extend([image,image_flipped])
        # Append measurements for center left and right images and augment with flipped images
        measurement = float(row[3])
        measurements.extend([measurement,-measurement,measurement+correction,-measurement-correction,measurement-correction,-measurement+correction])

# Create the training set
X_train = np.array(images)
y_train = np.array(measurements)


# # Create model

# In[11]:


model = Sequential()
# Normalize the input
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
# Crop the images to only the road
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(32,(3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
# Get the single steering output
model.add(Dense(1))

# Compile the model
model.compile(loss='mse',optimizer='adam')
# Train the model
model.fit(X_train,y_train,epochs = 3,validation_split=0.2, shuffle = True)
# Save the model
model.save('model.h5')

