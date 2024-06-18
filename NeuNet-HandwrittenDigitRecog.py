import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf #what we use in neural networks

#load the data set of handwritten digits
mnist = tf.keras.datasets.mnist
    #we know the classifications
(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #.load_data() already splits the data via the variables in the tuple

#normalize the data into 0-1 (make the data easier to compute)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
    #x data are the labels (pictures) while the y data is the classifications (no need to normalize)

#Define Model
model = tf.keras.models.Sequential() #creating of basic NeNet
#adding layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #784 input layers
model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu)) #new hidden layer with 128 neurons and activation fnc relu
model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 10, activation=tf.nn.softmax)) #output layer of 10 neurons and the softmax function which sums all probabilities

#complie the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit the model
model.fit(x_train, y_train, epochs=3) #epochs specifies how many times the model will go through the same process

#evaluate model
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)
#ACCURACY = 97%
#LOSS = 9%

'''
#TO TEST OUR OWN VALUES
    #go to paint, resize to 28x28 and save written digits
    #move the images to the folder the program is in
for x in range (1,6): #we have images with names 1-5
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}') #gives us the index of hightest value (classification)
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
'''


