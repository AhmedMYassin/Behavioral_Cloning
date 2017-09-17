import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import random

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

CSV_Data = pd.read_csv('/home/yassin/linux_sim/Data_CW/driving_log.csv')

images = []

measurements = []

correction_factor = 0.15

for idx in range(len(CSV_Data)):
	
	measurement = float(CSV_Data.iat[idx,3])
	Prob = random.uniform(0, 1)
	if Prob < 0.3 or measurement > 0.05:
		# Center Image	
		file_path = CSV_Data.iat[idx,0]
		image = mpimg.imread(file_path)
		images.append(image)
		measurements.append(measurement)
		# Flipped Center Image	
		image_flipped = np.fliplr(image)
		images.append(image_flipped)
		measurements.append(-1.0*measurement)
	else:
		# left Image	
		file_path = CSV_Data.iat[idx,1]
		image = mpimg.imread(file_path)
		images.append(image)
		measurements.append(measurement + correction_factor)
		# Right Image	
		file_path = CSV_Data.iat[idx,2]
		image = mpimg.imread(file_path)
		images.append(image)
		measurements.append(measurement - correction_factor)


X_train = np.array(images)
Y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape = X_train[0].shape))
model.add(Cropping2D(cropping = ((70,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')

#model = load_model('model.h5')

model.summary()

