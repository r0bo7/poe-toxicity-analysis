import requests
import json
import re
import time

def fetchObjects(subreddit, type, after):
    params = {
        "sort_type": "created_utc",
        "sort": "asc",
        "size": 1000,
        "subreddit": subreddit,
        "type": type,
        "after": after
    }

    print(params)
    r = requests.get("http://api.pushshift.io/reddit/" + type + "/search/", params=params, timeout=30)

    if r.status_code == 200:
        response = json.loads(r.text)
        data = response['data']
        sorted_data_by_id = sorted(data, key=lambda x: int(x['id'],36))
        return sorted_data_by_id
    else:
        print('Error making request', r)
        return None

def extract_reddit_data(subreddit, type, filename, start_timestamp):
    with open(filename, "a") as file:
        objects = fetchObjects(subreddit, type, after=start_timestamp)
        max_timestamp = 0
        while len(objects) > 0:
            for object in objects:
                max_timestamp = object['created_utc'] if object['created_utc'] > max_timestamp else max_timestamp
                file.write(json.dumps(object,sort_keys=True,ensure_ascii=True) + "\n")
            objects = fetchObjects(subreddit, type, after=max_timestamp)

extract_reddit_data(subreddit="pathofexile",type="submission", filename="json/pathofexile.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="diablo3",type="submission", filename="json/diablo3.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="Wolcen",type="submission", filename="json/Wolcen.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="Warframe",type="submission", filename="json/Warframe.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="leagueoflegends",type="submission", filename="json/leagueoflegends.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="GlobalOffensive",type="submission", filename="json/GlobalOffensive.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="DotA2",type="submission", filename="json/DotA2.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="Overwatch",type="submission", filename="json/Overwatch.json", start_timestamp=1577836800)
extract_reddit_data(subreddit="VALORANT",type="submission", filename="json/VALORANT.json", start_timestamp=1577836800)
