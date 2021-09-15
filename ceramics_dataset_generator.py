#%load_ext tensorboard
import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
import cv2
import segmentation_models as sm
sm.set_framework('tf.keras')
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


N_CLASSES = 18
BATCH_SIZE = 2
EPOCHS = 200
INPUT_SIZE = 512
PATCH_SIZE = 256
N_CHANNELS = 208
training = False
testing = True

dice_loss = sm.losses.DiceLoss()#class_weights=np.array([5.8179168983756275, 6.699261247237657, 11.340197193151843, 6.012864106830905, 0.06374408225901594]))#([56.41148061773509, 103.91508189401473, 2592.6238501139337, 64.57322187165651, 1.0451746816231888]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
lossFunction = total_loss#'categorical_crossentropy'
metrics = [sm.metrics.FScore(threshold=0.5), sm.metrics.IOUScore(threshold=0.5)]#
#metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES), tf.keras.metrics.CategoricalCrossentropy(),tf.keras.metrics.AUC(),tf.keras.metrics.CategoricalAccuracy()]


def Unet(pretrained_weights = None,input_size = (None,None,N_CHANNELS)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(N_CLASSES, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = lossFunction, metrics = metrics, run_eagerly=True)
    
    

    if(pretrained_weights):
      model.load_weights(pretrained_weights)

    return model

def getDataset(folder, createPatches=False):
  datasetImages = tf.data.Dataset.list_files(folder+"images/*.hdr", seed=10)#folder+"images/*.png"
  datasetMasks = tf.data.Dataset.list_files(folder+"masks/*.png", seed=10)
  
  
  def generateMask(image, values = N_CLASSES):
      masks = [(image == v) for v in range(4,21)]
      mask = np.stack(masks, axis=-1).astype('float32')
      background = 1 - mask.sum(axis=-1, keepdims=True)
      mask = np.concatenate((mask, background), axis=-1)
      return mask
  def openHDR(filename):
      return np.clip(envi.open(filename.numpy().decode("utf-8"))[256:768,:,12:220],0,1)#cv2.imread(filename.numpy().decode("utf-8"))[256:768]##[100,58,14]
  def openPNGMask(file_name):
      value = generateMask(cv2.imread(file_name.numpy().decode("utf-8"), 0)[256:768])
      return value.astype(np.float32)
  def get_patches(x, n):
    #tf.random.set_seed(5)
    return tf.reshape(
      tf.image.extract_patches(
          images=x,
          sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
          strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
          rates=[1, 1, 1, 1],
          padding='VALID'), (-1, PATCH_SIZE, PATCH_SIZE, n))#, seed=8)
  '''def get_patches_mask(x, n):
    #tf.random.set_seed(5)
    return tf.reshape(
      tf.image.extract_patches(
          images=x,
          sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
          strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
          rates=[1, 1, 1, 1],
          padding='VALID'), (-1, PATCH_SIZE, PATCH_SIZE, N_CLASSES))#, seed=8)'''
  
  
  
  datasetImages = datasetImages.map(lambda x: tf.py_function(openHDR, [x], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
  datasetMasks = datasetMasks.map(lambda x: tf.py_function(openPNGMask, [x], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
  if createPatches:
    datasetMasks = datasetMasks.map(lambda x: tf.py_function(get_patches, [x, N_CLASSES], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    datasetImages = datasetImages.map(lambda x: tf.py_function(get_patches, [x, N_CHANNELS], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
  
  dataset = tf.data.Dataset.zip((datasetImages, datasetMasks))
  batched_dataset = dataset.prefetch(tf.data.AUTOTUNE)#.shuffle(buffer_size=20, reshuffle_each_iteration=True)
  
  batched_dataset = tf.data.Dataset.range(2).interleave(lambda _: batched_dataset, num_parallel_calls=tf.data.AUTOTUNE)
  return batched_dataset

train_dataset = getDataset("/root/data/tmp/Ceramica_DatasetV3/Training/")
val_dataset = getDataset("/root/data/tmp/Ceramica_DatasetV3/Validation/")
test_dataset = getDataset("/root/data/tmp/Ceramica_DatasetV3/Test/", createPatches=False)
total_dataset = getDataset("/root/data/tmp/Ceramica_DatasetV3/", createPatches=False)
model = Unet()
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./best_model_dataset.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
    #tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1),    
  ]
  

'''counter=0
for elem in train_dataset:
      #print(i, 6-1)
      (image, gt_mask) = elem
      print(gt_mask[0].numpy().shape)
      for i in range(gt_mask[0].numpy().shape[0]):
        print(i)
        print(image[0].numpy().shape)
        print(gt_mask[0].numpy().shape)
        #print(np.amax(image.numpy()[i][:,:,[88,46,2]]), np.amin(image.numpy()[i][:,:,[88,46,2]]))
        plt.imsave("inputs_5/"+str(counter)+"_image.png", image[0].numpy()[i][:,:,[88,46,2]])
        plt.imsave("inputs_5/"+str(counter)+"_mask_0.png", np.uint8(gt_mask[0].numpy()[i][:,:,0]*255),cmap='gray')
        plt.imsave("inputs_5/"+str(counter)+"_mask_1.png", np.uint8(gt_mask[0].numpy()[i][:,:,1]*255),cmap='gray')
        plt.imsave("inputs_5/"+str(counter)+"_mask_2.png", np.uint8(gt_mask[0].numpy()[i][:,:,2]*255),cmap='gray')
        plt.imsave("inputs_5/"+str(counter)+"_mask_3.png", np.uint8(gt_mask[0].numpy()[i][:,:,3]*255),cmap='gray')
        plt.imsave("inputs_5/"+str(counter)+"_mask_4.png", np.uint8(gt_mask[0].numpy()[i][:,:,4]*255),cmap='gray')
        print(counter)
        counter+=1'''
if training:
  history = model.fit(train_dataset, validation_data=val_dataset,  epochs=EPOCHS, callbacks=callbacks)
  
  plt.figure(figsize=(30, 5))
  plt.subplot(131)
  plt.plot(history.history['iou_score'])
  plt.plot(history.history['val_iou_score'])
  plt.title('Model iou_score')
  plt.ylabel('iou_score')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
    
  # Plot training & validation loss values
  plt.subplot(132)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  
  plt.subplot(133)
  plt.plot(history.history['f1-score'])
  plt.plot(history.history['val_f1-score'])
  plt.title('Model f1-score')
  plt.ylabel('f1-score')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  '''
  plt.subplot(154)
  plt.plot(history.history['auc'])
  plt.plot(history.history['val_auc'])
  plt.title('Model auc')
  plt.ylabel('auc')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  
  plt.subplot(155)
  plt.plot(history.history['categorical_accuracy'])
  plt.plot(history.history['val_categorical_accuracy'])
  plt.title('Model categorical_accuracy')
  plt.ylabel('categorical_accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')'''
  plt.savefig("history_dataset.png")
  plt.show()


if testing:
  model.load_weights('best_model_dataset.h5')
  scores = model.evaluate(test_dataset)#total_dataset
  print("Loss: {:.5}".format(scores[0]))
  for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))
    
    
    

  if True:
    confusionMatrix = np.zeros((N_CLASSES,N_CLASSES))  
    
    i=0
    for elem in total_dataset:#test_dataset
      print(i)
      i+=1
      (image, gt_mask) = elem
      #image = np.expand_dims(image, axis=0)
      pr_mask = model.predict(image[0])
      '''for k in range(pr_mask.shape[3]):
        print(x_t[i])
        plt.imsave(str(i)+"_"+str(k)+".png",pr_mask[0,:,:,k], cmap='gray')'''
      r0 = np.argmax(gt_mask[0][0], axis=-1)
      p0 = np.argmax(pr_mask[0], axis=-1)
      r1 = np.argmax(gt_mask[0][1], axis=-1)
      p1 = np.argmax(pr_mask[1], axis=-1)
      for x in range(N_CLASSES):
        for y in range(N_CLASSES):
          confusionMatrix[x][y] += ((r0==x)*(p0==y)).sum()
          confusionMatrix[x][y] += ((r1==x)*(p1==y)).sum()
      '''
      for x in range(pr_mask.shape[1]):
        for y in range(pr_mask.shape[2]):
          #numpy.argmax(gt_mask[0][0], axis=-1)
          confusionMatrix[np.argmax(gt_mask[0][0][x][y])][np.argmax(pr_mask[0][x][y])] += 1
          confusionMatrix[np.argmax(gt_mask[0][1][x][y])][np.argmax(pr_mask[1][x][y])] += 1
          predictedMatrix[np.argmax(pr_mask[0][x][y])]+=1
          realMatrix[np.argmax(gt_mask[0][0][x][y])] += 1
          predictedMatrix[np.argmax(pr_mask[1][x][y])]+=1
          realMatrix[np.argmax(gt_mask[0][1][x][y])] += 1
          #print(np.argmax(pr_mask[0][x][y]),np.argmax(gt_mask[0][0][x][y]))'''
      print(confusionMatrix)

    realClasses = ["Real A", "Real B", "Real C", "Real D", "Real E", "Real F", "Real G", "Real H", "Real J", "Real K", "Real L", "Real M", "Real N", "Real P", "Real Q", "Real R", "Real S"]
    predictedClasses = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S"]
    fig, ax = plt.subplots(figsize = (40,40))
    im = ax.imshow(confusionMatrix,cmap="Blues")
    ax.set_xticks(np.arange(len(predictedClasses)))
    ax.set_yticks(np.arange(len(realClasses)))
    ax.set_xticklabels(predictedClasses)
    ax.set_yticklabels(realClasses)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(realClasses)):
      for j in range(len(predictedClasses)):
        text = ax.text(j, i, "%.3f" % confusionMatrix[i, j],ha="center", va="center", color="black")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig("./dataset_base.png")
    plt.show()
    print(confusionMatrix)
    confusionMatrixPredicted = confusionMatrix/np.sum(confusionMatrix,axis=0)
    confusionMatrixReal = confusionMatrix[:,:]
    realsum=np.sum(confusionMatrix,axis=1)
    print(realsum)
    confusionMatrixReal[0][:]/=realsum[0]
    confusionMatrixReal[1][:]/=realsum[1]
    confusionMatrixReal[2][:]/=realsum[2]
    confusionMatrixReal[3][:]/=realsum[3]
    confusionMatrixReal[4][:]/=realsum[4]
    realClasses = ["Real A", "Real B", "Real C", "Real D", "Real E", "Real F", "Real G", "Real H", "Real J", "Real K", "Real L", "Real M", "Real N", "Real P", "Real Q", "Real R", "Real S"]
    predictedClasses = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S"]
    fig, ax = plt.subplots(figsize = (40,40))
    im = ax.imshow(confusionMatrixPredicted,cmap="Blues")
    ax.set_xticks(np.arange(len(predictedClasses)))
    ax.set_yticks(np.arange(len(realClasses)))
    ax.set_xticklabels(predictedClasses)
    ax.set_yticklabels(realClasses)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(realClasses)):
      for j in range(len(predictedClasses)):
        text = ax.text(j, i, "%.3f" % confusionMatrixPredicted[i, j],ha="center", va="center", color="black")
    ax.set_title("Confusion matrix with predicted percentages")
    fig.tight_layout()
    fig.savefig("./dataset_predicted.png")
    plt.show()
    fig, ax = plt.subplots(figsize = (40,40))
    im = ax.imshow(confusionMatrixReal,cmap="Blues")
    ax.set_xticks(np.arange(len(predictedClasses)))
    ax.set_yticks(np.arange(len(realClasses)))
    ax.set_xticklabels(predictedClasses)
    ax.set_yticklabels(realClasses)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(realClasses)):
      for j in range(len(predictedClasses)):
        text = ax.text(j, i, "%.3f" % confusionMatrixReal[i, j],ha="center", va="center", color="black")
    ax.set_title("Confusion matrix with real percentages")
    fig.tight_layout()
    fig.savefig("./dataset_real.png")
    plt.show()
    
    print(confusionMatrixReal)
    print()
    print(confusionMatrixPredicted)
    print()
    
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(confusionMatrix[:-1,:-1],cmap="Blues")
    ax.set_xticks(np.arange(len(predictedClasses)))
    ax.set_yticks(np.arange(len(realClasses)))
    ax.set_xticklabels(predictedClasses)
    ax.set_yticklabels(realClasses)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(realClasses)-1):
      for j in range(len(predictedClasses)-1):
        text = ax.text(j, i, "%.3f" % confusionMatrix[i, j],ha="center", va="center", color="black")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig("./dataset_4_base.png")
    plt.show()
    print(confusionMatrix)
    '''#confusionMatrixPredicted = confusionMatrix/np.sum(confusionMatrix,axis=0)
    #confusionMatrixReal = confusionMatrix
    #realsum=np.sum(confusionMatrix,axis=1)
    print(realsum)
    confusionMatrixReal[0][:]/=realsum[0]
    confusionMatrixReal[1][:]/=realsum[1]
    confusionMatrixReal[2][:]/=realsum[2]
    confusionMatrixReal[3][:]/=realsum[3]'''
    realClasses = ['Real roman', 'Real iber', 'Real tardoantiguitat', 'Real modern', 'Real Background']
    predictedClasses = ['Predicted roman', 'Predicted iber', 'Predicted tardoantiguitat', 'Predicted modern', 'Predicted Background']
    fig, ax = plt.subplots()
    im = ax.imshow(confusionMatrixPredicted[:-1,:-1],cmap="Blues")
    ax.set_xticks(np.arange(len(predictedClasses)))
    ax.set_yticks(np.arange(len(realClasses)))
    ax.set_xticklabels(predictedClasses)
    ax.set_yticklabels(realClasses)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(realClasses)-1):
      for j in range(len(predictedClasses)-1):
        text = ax.text(j, i, "%.3f" % confusionMatrixPredicted[i, j],ha="center", va="center", color="black")
    ax.set_title("Confusion matrix with predicted percentages")
    fig.tight_layout()
    fig.savefig("./dataset_4_predicted.png")
    plt.show()
    fig, ax = plt.subplots()
    im = ax.imshow(confusionMatrixReal[:-1,:-1],cmap="Blues")
    ax.set_xticks(np.arange(len(predictedClasses)))
    ax.set_yticks(np.arange(len(realClasses)))
    ax.set_xticklabels(predictedClasses)
    ax.set_yticklabels(realClasses)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(realClasses)-1):
      for j in range(len(predictedClasses)-1):
        text = ax.text(j, i, "%.3f" % confusionMatrixReal[i, j],ha="center", va="center", color="black")
    ax.set_title("Confusion matrix with real percentages")
    fig.tight_layout()
    fig.savefig("./dataset_4_real.png")
    plt.show()