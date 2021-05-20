from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('keras model')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting images
img = image.load_img("input image", target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
if classes == 0:
    print("class1")
else:
    print("class2")




