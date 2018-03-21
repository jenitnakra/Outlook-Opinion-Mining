from queue import Queue
from threading import *
from fetch import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
from train import classifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
def animate(i,l):
		c,v = l
		# vectorizer = TfidfVectorizer(stop_words=None,decode_error='ignore',min_df=5, max_df = 0.8,use_idf=True,ngram_range=(1,3),norm='l2')
		graph_data = open('twitter-out.txt','r').read()
		lines = graph_data.split('\n')
		vector = v.transform(np.array(lines))
		xs = []
		ys = []
		x = 0
		y = 0
		col = 0
		for line in vector[-200:]:
			# print (type(line))
			
			pred = c.predict(line)
			x += 1
			if pred == 0:
				col = 0
				y -= 1
			elif pred == 1:
				col = 1
				y += 1

			xs.append(x)
			ys.append(y)

		ax1.clear()
		if (col == 0):
			ax1.plot(xs, ys,color ="red")
		else:
			ax1.plot(xs, ys,color ="blue")
		plt.title("Twitter Sentiment Analysis")
		plt.xlabel("Tweet")
		plt.ylabel("Sentiment Analyzed")

if __name__ == '__main__':
	os.system("rm twitter-out.txt && touch twitter-out.txt")
	style.use('ggplot')
	
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	print("Select the classfier to use:-\n 0.Logistic Regression\n 1.SVM\n 2.Random Forest \n 3.Naive Bayesian \n Wrong input: Naive bayesian ")
	i=input()
	c=classifier(i)
	query = input("Enter query text: ")
	p1 = Thread(target=fetch,args=(query, ))
	# p2 = Thread(target=anim)
	p1.start()
	# p2.start()
	ani = animation.FuncAnimation(fig, animate, fargs = (c,), interval=100	)
	
	plt.show()
	p1.join()
	# p2.join()
