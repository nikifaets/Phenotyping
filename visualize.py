import numpy as np
import matplotlib.pyplot as plt
from get_samples import get_samples
from normalize import normalize 
import cv2

def visualize(test_imgs, gt_imgs):

	fig=plt.figure(figsize=(8, 8))
	columns = (len(gt_imgs)+len(test_imgs))*2
	rows = 2
	count=0
	for i in range(1, len(test_imgs)):
	    gt1 = gt_imgs[i][0]
	    gt2 = gt_imgs[i][1]
	    cell1 = test_imgs[i][0]
	    cell2 = test_imgs[i][1]
	    fig.add_subplot(rows, columns, i+count); count+=1
	    plt.imshow(cell2)
	plt.show()

def visualize_ae(num, samples, model, data):

	num=700
	samples=3	
	test_data=data[num:num+samples]
	test_data = normalize(test_data)
	res = model.predict(test_data)
	print("range res", np.min(res), np.max(res))
	res*=255
	data+=0.5
	data*=255
	
	