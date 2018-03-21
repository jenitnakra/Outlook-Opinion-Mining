from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import matplotlib.pyplot as plt
import pandas as pd
import json
tweets_data_path = './twitter_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
		try:
			tweet = json.loads(line)
			if line not in tweets_data:
				tweets_data.append(tweet)
		except:
			continue    

tweets = pd.DataFrame()       

for i in tweets_data:
	print(i['text'])
