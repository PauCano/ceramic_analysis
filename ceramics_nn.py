import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import random
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
import cv2
import keras
import segmentation_models as sm
from segmentation_models import Unet
import albumentations as A

def visualize(**images):
	"""PLot images in one row."""
	n = len(images)
	plt.figure(figsize=(16, 5))
	for i, (name, image) in enumerate(images.items()):
		plt.subplot(1, n, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.title(' '.join(name.split('_')).title())
		plt.imshow(image,vmin=0, vmax=1)
	plt.show()
	
# helper function for data visualization	
def denormalize(x):
	"""Scale image to range 0..1 for correct plot"""
	x_max = np.percentile(x, 98)
	x_min = np.percentile(x, 2)	
	x = (x - x_min) / (x_max - x_min)
	x = x.clip(0, 1)
	return x
	

# classes for data loading and preprocessing
class Dataset:
	"""FieldLines Dataset. Read images, apply augmentation and preprocessing transformations.
	
	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline 
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing 
			(e.g. noralization, shape manipulation, etc.)
	
	"""
	
	CLASSES = ['roman', 'iber', 'tardoantiguitat', 'modern']
	
	def __init__(
			self, 
			images_dir, 
			masks_dir, 
			classes=None, 
			augmentation=None, 
			preprocessing=None,
	):
		self.ids = os.listdir(images_dir)
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
		self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
		
		# convert str names to class values on masks
		self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
		
		self.augmentation = augmentation
		self.preprocessing = preprocessing
	
	def __getitem__(self, i):
		
		# read data
		#image = cv2.imread(self.images_fps[i])
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		img = envi.open(self.images_fps[i])
		image = img[:,:,:]
		mask = cv2.imread(self.masks_fps[i], 0)
		image *= np.repeat(np.expand_dims(mask, axis=2), 224, axis=2)<250
		
		# extract certain classes from mask (e.g. cars)
		masks = [(mask == v) for v in self.class_values]
		mask = np.stack(masks, axis=-1).astype('float')
		
		# add background if mask is not binary
		if mask.shape[-1] != 1:
			background = 1 - mask.sum(axis=-1, keepdims=True)
			mask = np.concatenate((mask, background), axis=-1)
		
		# apply augmentations
		if self.augmentation:
			sample = self.augmentation(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
		
		# apply preprocessing
		if self.preprocessing:
			sample = self.preprocessing(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
			
		return image, mask
		
	def __len__(self):
		return len(self.ids)
	
	
class Dataloader(keras.utils.Sequence):
	"""Load data from dataset and form batches
	
	Args:
		dataset: instance of Dataset class for image loading and preprocessing.
		batch_size: Integet number of images in batch.
		shuffle: Boolean, if `True` shuffle image indexes each epoch.
	"""
	
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(dataset))

		self.on_epoch_end()

	def __getitem__(self, i):
		
		# collect batch data
		start = i * self.batch_size
		stop = (i + 1) * self.batch_size
		data = []
		for j in range(start, stop):
			data.append(self.dataset[j])
		
		# transpose list of lists
		batch = [np.stack(samples, axis=0) for samples in zip(*data)]
		
		return tuple(batch)
	
	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return len(self.indexes) // self.batch_size
	
	def on_epoch_end(self):
		"""Callback function to shuffle indexes each epoch"""
		if self.shuffle:
			self.indexes = np.random.permutation(self.indexes)

def get_simple_augmentation():
	"""Add paddings to make image shape divisible by 32"""
	test_transform = [
		A.PadIfNeeded(1024, 512)
	]
	return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
	"""Construct preprocessing transform
	
	Args:
		preprocessing_fn (callable): data normalization function 
			(can be specific for each pretrained neural network)
	Return:
		transform: albumentations.Compose
	
	"""
	
	_transform = [
		A.Lambda(image=preprocessing_fn),
	]
	return A.Compose(_transform)





BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)
BATCH_SIZE = 2
LR = 0.0001
EPOCHS = 40




x = os.listdir("D:\\Ceramica\\NN_Dataset\\images")
x = [name for name in x if name.split(".")[-1]=="hdr"]
y = ["D:\\Ceramica\\NN_Dataset\\masks\\"+name.split(".")[0]+".png" for name in x]
x = ["D:\\Ceramica\\NN_Dataset\\images\\"+name for name in x]

random.seed(1)
jointImages = list(zip(x,y))
random.shuffle(jointImages)
x,y = zip(*jointImages)
jointImages=[]

setTotalImages = len(x)
percentile60 = int(setTotalImages*0.6)
percentile90 = int(setTotalImages*0.9)

x_train = x[0:percentile60]
x_val = x[percentile60:percentile90]
x_test = x[percentile90:]

y_train = y[0:percentile60]
y_val = y[percentile60:percentile90]
y_test = y[percentile90:]

train = Dataset("D:\\vine_fields_dataset\\linesDatasetOutput\\", "D:\\vine_fields_dataset\\linesDataset_GT_Output_v2\\", classes=['roman', 'iber', 'tardoantiguitat', 'modern'], augmentation=get_simple_augmentation(), preprocessing = get_preprocessing(preprocess_input))#
validation = Dataset("D:\\vine_fields_dataset\\linesDatasetOutput\\", "D:\\vine_fields_dataset\\linesDataset_GT_Output_v2\\", classes=['roman', 'iber', 'tardoantiguitat', 'modern'], augmentation=get_simple_augmentation(), preprocessing = get_preprocessing(preprocess_input))
test = Dataset("D:\\vine_fields_dataset\\linesDatasetOutput\\", "D:\\vine_fields_dataset\\linesDataset_GT_Output_v2\\", classes=['roman', 'iber', 'tardoantiguitat', 'modern'], augmentation=get_simple_augmentation(), preprocessing = get_preprocessing(preprocess_input))



train.images_fps = x_train
train.masks_fps = y_train
validation.images_fps = x_val
validation.masks_fps = y_val
test.images_fps = x_test
test.masks_fps = y_test

train.ids = x_train
validation.ids = x_val
test.ids = x_test

print(x[5])
# =============================================================================
image, mask = train[5] # get some sample
visualize(
	image=image[:,:,[100,58,16]], 
	roman_mask=mask[..., 0].squeeze(),
	iber_mask=mask[..., 1].squeeze(),
	tado_mask=mask[..., 2].squeeze(),
	modern_mask=mask[..., 3].squeeze(),
	background_mask=mask[..., 4].squeeze()
)
# =============================================================================
train_dataloader = Dataloader(train,batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = Dataloader(validation,batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = Dataloader(test,batch_size=BATCH_SIZE, shuffle=False)

#assert train_dataloader[0][0].shape == (BATCH_SIZE, 672, 672, 3)
#assert validation_dataloader[0][1].shape == (BATCH_SIZE, 672, 672, 3)

callbacks = [
	keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
	keras.callbacks.ReduceLROnPlateau(),
]

model = Unet(BACKBONE, encoder_weights=None, input_shape=(1024, 512, 224), classes=5, activation="softmax")
optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, dice_loss, metrics)
keras.utils.plot_model(model, show_shapes=True, to_file="model.png")
exit()

if True:
	print("training")
	history = model.fit(
		train_dataloader, 
		steps_per_epoch=len(train_dataloader), 
		epochs=EPOCHS, 
		callbacks=callbacks, 
		validation_data=validation_dataloader, 
		validation_steps=len(validation_dataloader),
	)
	
	
	# Plot training & validation iou_score values
	plt.figure(figsize=(30, 5))
	plt.subplot(121)
	plt.plot(history.history['iou_score'])
	plt.plot(history.history['val_iou_score'])
	plt.title('Model iou_score')
	plt.ylabel('iou_score')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	
	# Plot training & validation loss values
	plt.subplot(122)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
else:
	print("testing")
	model.load_weights('best_model.h5')
		
	scores = model.evaluate(test_dataloader)
	print("Loss: {:.5}".format(scores[0]))
	for metric, value in zip(metrics, scores[1:]):
		print("mean {}: {:.5}".format(metric.__name__, value))
	
	
	n=3
	ids = np.random.choice(np.arange(len(test)), size=n)
	for i in ids:
		print(x_test[i])
		image, gt_mask = test[i]
		image = np.expand_dims(image, axis=0)
		pr_mask = model.predict(image)
		visualize(
			image=denormalize(image.squeeze()[:,:,[100,58,16]]),
			gt_mask=gt_mask.squeeze(),
			pr_mask=pr_mask.squeeze(),
		)
		
	
	if False:
		confusionMatrix = np.zeros((4,4))
		for i in range(len(test)):
			print(i, len(test)-1)
			image, gt_mask = test[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = model.predict(image)
			for x in range(pr_mask.shape[1]):
				for y in range(pr_mask.shape[2]):
					confusionMatrix[np.argmax(pr_mask[0][x][y])][np.argmax(gt_mask[x][y])] += 1
		confusionMatrixPredicted = confusionMatrix/np.sum(confusionMatrix,axis=0)
		confusionMatrixReal = confusionMatrix
		realsum=np.sum(confusionMatrix,axis=1)
		confusionMatrixReal[0][:]/=realsum[0]
		confusionMatrixReal[1][:]/=realsum[1]
		confusionMatrixReal[2][:]/=realsum[2]
		realClasses = ['Real roman', 'Real iber', 'Real tardoantiguitat', 'Real modern']
		predictedClasses = ['Real roman', 'Real iber', 'Real tardoantiguitat', 'Real modern']
		fig, ax = plt.subplots()
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
		plt.show()
		fig, ax = plt.subplots()
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
		plt.show()
		
	
	

