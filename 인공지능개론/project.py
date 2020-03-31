from tensorflow.keras import *
from keras.utils.np_utils import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
 
# CIFAR-10 train data를 불러옴.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
 
# 기본적으로 image의 픽셀 값들은 unsigned int 8bit 표현이므로 소수점을 표현할 수 있게 0~1 사이로 정규화.
# [uint8 - 256  -> float]
train_images, test_images = train_images / 255.0, test_images / 255.0
 
# CIFAR-10 데이터가 10개의 class로 구분되는 형태이므로 [0~9] one hot encoding 진행.
# 0 - 0000000001  1 - 0000000010
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
 
# batch normalization을 하기 위해서 픽셀 값들에 대해 전처리
# 실제로는 mini batch 마다 input에 대해서 전처리를 해주는 것이 맞지만, 간편하게 구현하기 위해 한번에 전처리함.
# 각 채널마다 평균을 빼주고 표준편차로 나누어줌. 이 때 0으로 나누는 에러를 방지하기 위해 작은 숫자를 더해서 나눔.
mean = np.mean(train_images,axis=(0,1,2,3))
std = np.std(train_images,axis=(0,1,2,3))
train_images = (train_images-mean)/(std+1e-7)
test_images = (test_images-mean)/(std+1e-7)
 
 
# 다음과 같이 연속된 층의 모델을 쌓음.
model = models.Sequential()
# Convolutional layer 1 input - 32*32*3
# cifar-10 data의 인풋이 32,32,3 이므로 input_shape = (32,32,3) 을 사용.
# kernel filter의 개수를 96개, kernel의 크기를 3*3을 이용
# padding='same' 을 이용하여 원래 인풋과 사이즈가 동일하게 zero padding을 함
# convolution을 적용한 이후에는 relu 활성화 함수를 적용
# 이후에는 batch normalization을 진행.  
# 연속으로 Convolutional layer 3개를 쌓은 이후에는 maxpooling 진행
# Maxpooling 이후에는 dropout
model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(96, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(96, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
model.add(layers.Dropout(0.5))                                              
# Convolutional layer 2 16*16*96
model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(3, 3),strides=2, padding='same'))
model.add(layers.Dropout(0.5))
# Convolutional layer 3 8*8*192, 1*1 convolutional layer를 사용하여 차원수 조정 및 parameter 감소
model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(192, (1, 1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(10, (1, 1), activation='relu'))
# Fully connected layer
# 이전까지의 input들을 1차원으로 만든 후에 
# 64개의 node를 가진 fully connected layer, dropout, class 10개로 softmax
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
 
# 모델 구조 표시
model.summary()
 
# learning rate scheduler
# epoch 수에 따라서 학습률을 변경해준다. 
# 0~50 : 0.01    ,   51~100 : 0.005   , 101~150 : 0.001
def scheduler(epoch):
  if epoch <= 50:
    return 0.01
  elif epoch <= 100:
    return 0.005
  elif epoch <= 150:
    return 0.001
  return 0.001
 
# fit 함수에서 1번 epoch가 끝날 때마다 불리어 지는 callback 함수
callback = callbacks.LearningRateScheduler(scheduler)
 
# 위에서 만든 sequential model의 error function은 categorical crossentropy를 사용하고
# optimizer는 SGD를 사용하며 학습률은 0.01, momentum은 0.9를 사용하며
# 평가기준으로는 accuracy[정확도]를 사용
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.SGD(lr=scheduler(0), momentum=0.9), 
              metrics=['accuracy'])
 
# 너비와 높이를 +- 0.1 사이의 값으로 랜덤하게 이동시킴. 즉, 위아래 좌우로 랜덤하게 이동 [즉 인자로 최소 0, 최대 1의 값이 올 수 있음]
# 랜덤하게 좌우반전도 시킴
data_generator = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
# train image에 대해서 generator를 만듬.
data_generator.fit(train_images)
 
# batch size와 epoch number
b_size = 32
e_num = 150
 
# history에는 한 번 epoch 마다의 loss, accuracy, validation loss, validation accuracy가 저장됨
# data generator로 train image에 대해서 batch size 만큼 이미지를 랜덤으로 만들고
# epoch 당 몇개의 배치를 적용하는지를 steps_per_epoch 로 설정
# 총 학습횟수 [epoch] e_num
# validation_data() 로 1번의 epoch가 끝날 때마다 학습한 모델을 가지고 validation data에 evaluate 진행
# verbose - 한번 에포크마다 에포크의 진행상태를 보여주는 방식. 0 - 아무것도 표시 x  1 - progress bar 2 - 한 줄로 표시
# callbacks - 한번의 에포크가 끝날 때마다 불리어지는 함수
history = model.fit_generator(data_generator.flow(train_images, train_labels, batch_size=b_size),
                    steps_per_epoch=len(train_images) / b_size, epochs=e_num, validation_data=(test_images, test_labels), verbose=2, callbacks = [callback])
 
# 학습한 모델로 test_images와 test_labels에 대해서 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("validation loss - ",test_loss)
print("validation accuracy - ", test_acc)
 
# 학습이 진행되는 동안[각 epoch 마다] 저장된 loss, accuracy, validation loss, validation accuracy 를 
# 저장한 history를 이용하여 그래프로 표시
fig, loss_ax = plt.subplots()
 
acc_ax = loss_ax.twinx()
 
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
 
acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
 
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
 
loss_ax.legend(loc='lower left')
acc_ax.legend(loc='upper left')
 
plt.show()
