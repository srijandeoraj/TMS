from datetime import timedelta, date, timezone
import twint
import pandas as pd

start = date(2019, 7, 15)
end = date(2020, 7, 10)
#end = date(2019,7,20)
delta = end-start
dateList = []

for i in range(delta.days):
    new_day = start + timedelta(i)
    dateList.append(new_day)

dates = pd.DataFrame(dateList, columns = ["Date"])

response = input("Please enter Keyword: ")

print("Fetch twitter data for "+ response + " company keyword....")

keyword = response

c = twint.Config()
c.Store_object = True
c.Pandas = True
c.Search = keyword
c.Limit = 50
c.Lang = 'en'

df = pd.DataFrame()
for i in range(len(dateList)-1):
    dayTweets = []
    c.Since = str(dateList[i])
    c.Until = str(dateList[i+1])
    twint.run.Search(c)
    Tweets_df = twint.storage.panda.Tweets_df
    df = pd.concat([df, Tweets_df])

drop_columns = df.drop(columns = ["id", "conversation_id", "created_at", "timezone", "place", "hashtags", "cashtags", "user_id", "user_id_str", "username", "name", "day", "hour", "link", "retweet","nlikes","nreplies","nretweets","quote_url","search","near","geo","source","user_rt_id","user_rt","retweet_id","reply_to","retweet_date", "translate","trans_src", "trans_dest"])
drop_columns['date'] = pd.to_datetime(drop_columns["date"], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')

tweets = drop_columns.groupby("date").agg(lambda x: x.tolist())
tweets_separate = tweets.tweet.apply(pd.Series)

dates = dates[:-1]
tweets_df= pd.concat([dates.reset_index(drop=True),tweets_separate.reset_index(drop=True)], axis=1)

tweets_df.to_csv("tweets.csv", index = False)
