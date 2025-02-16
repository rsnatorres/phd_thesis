#Importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os

#Creates a crawler that retrieves a tweet database, according to keywords, and exports it to a .csv
#@param key_words: a list of strings that contains all keywords to be searched in Twitter
#@param year_start_end_tuples: a list of numeric tuples that contains pairs of first and last years to search Twitter
#@param path_dir: a string containing the absolute path to the directory storing files
#@param today: a datetime of the current day
def crawl_tweets(key_words, year_start_end_tuples, today, path_dir):
    for key in key_words:
        for year_start, year_end in year_start_end_tuples:
            print(year_start, year_end)
            print(year_start_end_tuples)
            query = f'{key} since:{year_start}-01-01 until:{year_end}-01-01 lang:pt'
            tweets_list = []
            try:
                for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                    tweets_list.append([tweet.date, tweet.user.username, tweet.content,
                                        tweet.user.displayname, tweet.user.description,
                                        tweet.user.verified, tweet.user.followersCount,
                                        tweet.user.friendsCount, tweet.user.statusesCount,
                                        tweet.user.favouritesCount, tweet.user.listedCount,
                                        tweet.user.mediaCount, tweet.user.location,
                                        tweet.user.protected, tweet.outlinks,
                                        tweet.tcooutlinks, tweet.replyCount,
                                        tweet.retweetCount, tweet.likeCount,
                                        tweet.quoteCount, tweet.lang, tweet.source, 
                                        tweet.media, tweet.retweetedTweet, tweet.quotedTweet,
                                        tweet.mentionedUsers])
            except Exception as e:
                print('Exception reached!')
                print(e)
                crawl_tweets(key_words, year_start_end_tuples, today, path_dir)
            tweets_df = pd.DataFrame(tweets_list, columns=['datetime', 'username', 'content', 'display name',
                                                   'user description', 'verified', 'follower count',
                                                  'friends count', 'statuses count', 'favourites count',
                                                  'listed count', 'media count', 'location', 'protected',
                                                  'outlinks', 'tcooutlinks', 'reply count', 
                                                   'retweet count', 'like count', 'quote count', 'lang',
                                                   'source', 'media', 'retweeted tweet', 'quoted tweet',
                                                   'mentioned users'])
            file_name = f'df query {key} {year_start} done {today} len {len(tweets_df)}.csv'
            path_file = os.path.join(path_dir, file_name)
            tweets_df.to_csv(path_file)
            print(f'{file_name} CRAWLED')
            print(year_start, year_end)
            print(year_start_end_tuples)