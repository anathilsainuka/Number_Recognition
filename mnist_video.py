import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Get mnist data and cache it 
def get_mnist_data():
    # Get mnist data 
    path = 'mnist.npz'

    # Get data - this will be cached 
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return (x_train, y_train, x_test, y_test)

# Train model with mnist data 
def train_model(x_train, y_train, x_test, y_test):
    # Callback to stop training after reaching 99% accuracy
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(logs)
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    # Normalize 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # Fit model
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    
    # Stats 
    print(history.epoch, history.history['accuracy'][-1])
    return model

# Predict digit using image passed in 
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# OpenCV part 

# Left mouse click handler 
startInference = False
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Threshold slider handler 
threshold = 100
def on_threshold(x):
    global threshold
    threshold = x

# The OpenCV display loop 
def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if startInference:
            # Frame counter for showing text 
            frameCount += 1

            # Black outer frame
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            # Get central image 
            resizedFrame = thr[240 - 75:240 + 75, 320 - 75:320 + 75]
            background[240 - 75:240 + 75, 320 - 75:320 + 75] = resizedFrame

            # Resize for inference 
            iconImg = cv2.resize(resizedFrame, (28, 28))

            # Pass to model predictor 
            res = predict(model, iconImg)

            # Clear background 
            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            # Show text 
            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320 - 80, 240 - 80), (320 + 80, 240 + 80), (255, 255, 255), thickness=3)

            # Display frame 
            cv2.imshow('background', background)
        else:
            # Display normal video 
            cv2.imshow('background', frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function 
def main():
    # If a model is already saved just load it - else build it 
    model = None
    try:
        # Corrected the file path extension to '.h5'
        model = tf.keras.models.load_model('model.h5')
        print('Loaded saved model.')
        print(model.summary())
    except:
        # Load and train data 
        print("Getting mnist data...")
        (x_train, y_train, x_test, y_test) = get_mnist_data()
        print("Training model...")
        model = train_model(x_train, y_train, x_test, y_test)
        print("Saving model...")
        model.save('model.h5')  # Corrected to .h5 extension
    
    print("Starting CV...")

    # Show OpenCV window
    start_cv(model)

# Call main
if __name__ == '__main__':
    main()
