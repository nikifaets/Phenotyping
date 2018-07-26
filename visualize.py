import numpy as np
import matplotlib.pyplot as plt

def visualize(test_imgs, gd_imgs):

	fig=plt.figure(figsize=(8, 8))
	columns = (len(gd_imgs)+len(test_imgs))*2
	rows = 2
	count=0
	for i in range(1, len(test_imgs)):
	    gd1 = gd_imgs[i][0]
	    gd2 = gd_imgs[i][1]
	    cell1 = test_imgs[i][0]
	    cell2 = test_imgs[i][1]
	    fig.add_subplot(rows, columns, i+count); count+=1
	    plt.imshow(cell2)
	plt.show()