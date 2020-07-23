import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd

# Load Data
# Read .csv file image and check the show function

train_file = 'facial-keypoints-detection/training/training.csv'
test_file = 'facial-keypoints-detection/test/test.csv'

'''#train_points=np.genfromtxt(train_file,dtype=int ,delimiter=',',skip_header=1,skip_footer =5000)
train_images=np.genfromtxt(train_file,dtype=float ,skip_header=1,skip_footer =5000)
'''
'''# video
height, width, layers = train_images[0].shape
size = (width,height)

img_array = []
for i in range(30):
    img_array.append(train_images[i].reshape(96,96))

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'XVID'), 5.0, (96,96),0)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
'''

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_d=train_data.dropna()
train_d.info()

train_pnt=train_d.drop(labels=['Image'],axis=1)

train_img= train_d['Image'].str.split(" " , expand = True)
x=train_img.to_numpy(dtype=np.uint8)
x=x.reshape(-1, 96, 96, 1)
print(x.shape)

y=train_pnt.to_numpy()
print(y.shape)

# Color red points on the image

figures = plt.figure(figsize=(8, 8))
figures.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
'''#
def draw_points(index):
    plt.imshow(x[index].reshape(96,96), cmap='gray')

    k = []
    for i in range(1, 31, 2):
        k.append(plt.plot(y[index,i-1],y[index,i], 'ro'))

    return k

for i in range(32, 57, 1):
    ax=figures.add_subplot(5, 5, i + 1 - 32, xticks=[], yticks=[])
    draw_points(i)


##plt.show()'''

# Split the train set
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2)
print (x_train.shape)
print(y_train.shape)

# Create the Model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(96,96,1),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())


model.add(Conv2D(64,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(96,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())


model.add(Conv2D(96,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(128,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())


model.add(Conv2D(128,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(256,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())


model.add(Conv2D(256,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(512,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())


model.add(Conv2D(512,(3,3),padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30))

model.summary()

# Training and Save Model
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
hist = model.fit(x_train,y_train,epochs=100,validation_data=(x_val,y_val))


# Reshape data to form 96x96 images
#x_train = x_train.reshape(-1, 96, 96, 1)
#x_val = x_val.reshape(-1, 96, 96, 1)

#x_train = x_train.astype("float32")/255.
#x_val = x_val.astype("float32")/255.


score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('FD_new.h5')

# Load and evaluate model
new_model = load_model('FD_new.h5')

#final_ls, final_acc=new_model.evaluate(x_val,y_val, verbose=0)

#print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_ls, final_acc))


test_data['Image'] = test_data['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
test_X = np.asarray([test_data['Image']], dtype=np.uint8).reshape(test_data.shape[0],96,96,1)
test_res = new_model.predict(test_X)
train_predicts = new_model.predict(x)
n = 11

xv = x[n].reshape((96,96))
plt.imshow(xv,cmap='gray')

for i in range(1,31,2):
    plt.plot(train_predicts[n][i-1], train_predicts[n][i], 'ro')
    plt.plot(y_train[n][i-1], y_train[n][i], 'x', color='green')

plt.show()


