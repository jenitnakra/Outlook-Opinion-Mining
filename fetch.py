import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


class listener(StreamListener):
	def on_data(self, data):
		try:
			all_data = json.loads(data)
			tweet = all_data["text"]
			# sentiment_value,confidence = s.sentiment(tweet)
			# print(tweet, sentiment_value)
			output = open("twitter-out.txt","a")
			output.write(tweet)
			output.write('\n')
			output.close()
			return True
		except:
			return "Error in retrieval"

	def on_error(self, status):
        	print(status)

def fetch(query):
	#consumer key, consumer secret, access token, access secret.
	ckey="sU8KmrbicJGzzpVWNGjQk2A4z"
	csecret="JYV84LWnE0Ux7vj4L07tIpMxWuFQiMpec1XezN64sonqb2hDOi"
	atoken="950404623411552258-B79z3xlz1r07zmfEV5BcnVpUBwnFVXz"
	asecret="O0bW2jK1xIjPlzryWTW2YxyfwcniZwLqBlXnhBVddWBX1"
	auth = OAuthHandler(ckey, csecret)
	auth.set_access_token(atoken, asecret)

	twitterStream = Stream(auth, listener())
	twitterStream.filter(track=[query])
