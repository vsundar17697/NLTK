import posAndNegDataSet as p
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json


ckey="hjXX40UnnIQLXaZNY4U5btac7"
csecret="QvX24laLcFbDi6avFSU5BTWSMNbUOvMPoLwsRN33fbJAPi4hsc"
atoken="4308619815-SG9lVOrMFmp8zJaNhPcq7Ww38mgfDnw1uSrGxpG"
asecret="kFcPczzrzJtu6dYbuypkToUN9fFywx7FjvtYRSKRe1JHn"


class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"].encode('utf-8')
        sent_val , conf_val = p.sentiment(tweet)
        print(tweet , sent_val , conf_val)

        if conf_val*100 >= 60:
            output = open('twitter-out.txt' , 'a')
            output.write(tweet+','+sent_val+'\n')
            output.close()

        return True

    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["video"])