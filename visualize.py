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
	print("range test_data", np.min(test_data), np.max(test_data))
	print("range data", np.min(data), np.max(data))
	print("range res", np.min(res), np.max(res))
	#print(np.max(res), "MEAN", np.mean(res))
	test_imgs = list()
	gt_imgs = list()
	for i in range(samples):
		print('i', i)
		cell1,cell2 = get_samples(i,res)
		gt1, gt2 = get_samples(num+i, data)
		gt1 = cv2.resize(gt1,None, fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
		gt2 = cv2.resize(gt2,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
		cell1 = cv2.resize(cell1,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
		cell2 = cv2.resize(cell2,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
		test_imgs.append((cell1, cell2))
		gt_imgs.append((gt1,gt2))

	print(len(gt_imgs), len(test_imgs))

	counter = 0
	for i in test_imgs:
		cell1,cell2 = i
		cv2.imshow("nucleus1_"+str(counter), cell1)
		cv2.imshow("nucleus2_"+str(counter), cell2)
		counter+=1

	counter = 0
	for j in gt_imgs:
		gt1,gt2 = j
		cv2.imshow("ground_truth1_"+str(counter), gt1)
		cv2.imshow("ground_truth2_"+str(counter), gt2)
		counter+=1
		
	cv2.waitKey()