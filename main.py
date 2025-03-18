import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model('handwritten.model')

img_name = input("Please enter the name of the input image\n(Do ensure the image type is in .jpg or jpeg)\n")
decision = input("Is this created on paint?Y/N\n")

#image_number = 1
if os.path.isfile(f"{img_name}.jpg"):
    try:
        if decision == 'N':
            # Load the scanned image
            img = cv2.imread(f"{img_name}.jpg", cv2.IMREAD_GRAYSCALE)

            # Apply binary thresholding
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Save or display the binary image
            cv2.imwrite("binary_image.jpg", img)
            # OR
            cv2.imshow("Binary Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            img = cv2.imread(f"{img_name}.jpg")[:,:,0] #the [:,:,0] is used since we don't care abt the color of the image but just the form
            
        num_white_pixels = cv2.countNonZero(img)
        num_black_pixels = img.size - num_white_pixels
        if num_white_pixels > num_black_pixels:
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = np.invert(np.array([img]))
        else:
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = np.array([img])
            
        prediction = model.predict(img) 

        print(f"The number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error")
    #finally: 
        #image_number += 1




