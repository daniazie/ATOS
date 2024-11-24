from __future__ import print_function
import base64
import json
import requests

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from scipy.spatial.distance import euclidean, cosine
    
try:
    import urllib.request, urllib.error
    import urllib.parse as urllibparse
except ImportError:
    import urllib as urllibparse


# Set up Spotify API base URL
SPOTIFY_API_BASE_URL = 'http://api.spotify.com'
API_VERSION = 'v1'
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)

# Set up authorization URL
SPOTIFY_AUTH_BASE_URL = 'https://accounts.spotify.com/{}'
SPOTIFY_AUTH_URL = SPOTIFY_AUTH_BASE_URL.format('authorize')
SPOTIFY_TOKEN_URL = SPOTIFY_AUTH_BASE_URL.format('api/token')

# Client keys
CLIENT = json.load(open('./conf.json', 'r+'))
CLIENT_ID = CLIENT['id']
CLIENT_SECRET = CLIENT['secret']

# Server side parameters
REDIRECT_URI = "http://localhost/callback"
SCOPE = 'user-top-read playlist-modify-public playlist-modify-private user-read-private user-read-email'
STATE = ""
SHOW_DIALOG_bool = True
SHOW_DIALOG_str = str(SHOW_DIALOG_bool).lower()

auth_query_parameters = {
    "response_type": "code",
    "redirect_uri": REDIRECT_URI,
    "scope": SCOPE,
    "client_id": CLIENT_ID
}


URL_ARGS = "&".join(["{}={}".format(key, urllibparse.quote(val))
                     for key, val in list(auth_query_parameters.items())])
AUTH_URL = "{}/?{}".format(SPOTIFY_AUTH_URL, URL_ARGS)

def authorize(auth_token):
    code_payload = {
        "grant_type": "authorization_code",
        "code": str(auth_token),
        "redirect_uri": REDIRECT_URI
    }

    base64encoded = base64.b64encode(("{}:{}".format(CLIENT_ID, CLIENT_SECRET)).encode())
    headers = {"Authorization": "Basic {}".format(base64encoded.decode())}

    post_request = requests.post(SPOTIFY_TOKEN_URL, data=code_payload, headers=headers)

    response_data = json.loads(post_request.text)
    access_token = response_data["access_token"]

    auth_header = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(access_token)}
    return auth_header


SPOTIFY_GET_PROFILE_URL = 'https://api.spotify.com/v1/me'

def get_user_profile(auth_header):
    response = requests.get(
        SPOTIFY_GET_PROFILE_URL,
        headers = auth_header
    )
    response.json()
    return response.json()['id']

def get_top_tracks(auth_header):
    SPOTIFY_GET_TOP_TRACKS_URL = 'https://api.spotify.com/v1/me/top/tracks?time_range=short_term&limit=10'
    response = requests.get(SPOTIFY_GET_TOP_TRACKS_URL, headers = auth_header)
    resp_json = response.json()

    top_tracks = []
    try:
        for tracks in resp_json['items']:
            track_id = tracks['id']
            track_name = tracks['name']
            artists = tracks['artists']
            artists_name = ', '.join(
            [artist['name'] for artist in artists]
            )

            top_tracks_dict = {
                "track_id": track_id,
                "track_name": track_name,
                "artists_name": artists_name
            }

            top_tracks.append(top_tracks_dict)

        df = pd.DataFrame(top_tracks)
        df['rank'] = range(1, len(df)+1)
    

        return df
    except:
        print(response.status_code)
        return "failed"

def get_top_tracks_features(auth_header, top_tracks_df):
    top_tracks_list = top_tracks_df['track_id'].values
    top_tracks_rank = top_tracks_df['rank'].values
    top_tracks_features_list = []

    for track in range(len(top_tracks_list)):
        track_id = top_tracks_list[track]
        rank = top_tracks_rank[track]
        SPOTIFY_GET_TRACK_AUDIO_FEATURES_URL = 'https://api.spotify.com/v1/audio-features/'+track_id
        response = requests.get(
            SPOTIFY_GET_TRACK_AUDIO_FEATURES_URL,
            headers = auth_header
        )
        resp_json = response.json()   
        for items in resp_json:
            danceability = resp_json.get('danceability')
            energy = resp_json.get('energy')
            loudness = resp_json.get('loudness')
            mode = resp_json.get('mode')
            speechiness = resp_json.get('speechiness')
            acousticness = resp_json.get('acousticness')
            instrumentalness = resp_json.get('instrumentalness')
            liveness = resp_json.get('liveness')
            valence = resp_json.get('valence')
            tempo = resp_json.get('tempo')

            track_features = {
                'track_id': track_id,
                'rank': rank,
                'danceability': danceability,
                'energy': energy,
                'loudness': loudness,
                'mode': mode,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'valence': valence,
                'tempo': tempo
            }
        top_tracks_features_list.append(track_features)
            
        df = pd.DataFrame(top_tracks_features_list)
    return df

def get_data():
    df = pd.read_csv('spotify-tracks.csv').drop('Unnamed: 0', axis=1)
    df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    df = df[df['track_genre'] != 'children']
    df = df[df['track_genre'] != 'kids']
    df = df[df['track_genre'] != 'sleep']
    df = df[df['artists'] != 'Billboard Baby Lullabies']
    df = df.dropna().reset_index()
    return df

features = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
mm = MinMaxScaler()

def get_data_features(data_df):
     df = pd.DataFrame(data_df, columns=features)
     df['loudness'] = mm.fit_transform(df['loudness'].values.reshape(-1, 1))
     df['liveness'] = mm.fit_transform(df['liveness'].values.reshape(-1, 1))
     return df

def get_weighted_vector(top_tracks_features_df):
     max_rank = top_tracks_features_df['rank'].max()
     top_tracks_features_df['rank_weight'] = (max_rank - top_tracks_features_df['rank'] + 1) / max_rank

     top_tracks_features_df['loudness'] = mm.fit_transform(top_tracks_features_df['loudness'].values.reshape(-1, 1))
     top_tracks_features_df['liveness'] = mm.fit_transform(top_tracks_features_df['liveness'].values.reshape(-1, 1))
     avg_vector = np.average(top_tracks_features_df[features].drop(columns='mode').values, axis=0, weights=top_tracks_features_df['rank_weight'])
     avg_vector = np.insert(avg_vector, 3, top_tracks_features_df['mode'].mode())
     return avg_vector, top_tracks_features_df

kmeans = KMeans(n_clusters=5, tol=0.0001, random_state=42, max_iter=10000)

def find_clusters(data_df, features_df):
     data_df['cluster'] = kmeans.fit_predict(features_df)
     
     return data_df

def predict(avg_vector):
    cluster = kmeans.predict(avg_vector.reshape(1,-1))
    return cluster[0]

def get_nontop_tracks(data_df, top_tracks_df):
     df = data_df[~data_df['track_id'].isin(top_tracks_df['track_id'].values)]
     df = df.reset_index()
     df = df.drop(columns='index')
     return df

def get_distance(nontop_tracks_df, avg_vector):
    nontop_tracks_df['distance'] = nontop_tracks_df[features].apply(
         lambda row: cosine(row.values, avg_vector), axis=1
    )
    return nontop_tracks_df


def get_recs(nontop_tracks_df, cluster, metric):
    if metric == 'cluster':
        recs_df = nontop_tracks_df[['track_id', 'track_name', 'artists', 'track_genre', 'cluster', 'distance']].loc[nontop_tracks_df['cluster'] == cluster].sample(n=15)

    elif metric == 'distance':
        recs_df = nontop_tracks_df[['track_id', 'track_name', 'artists', 'track_genre', 'cluster', 'distance']].sort_values(by='distance', ascending=True)
    else:
        recs_df = nontop_tracks_df[['track_id', 'track_name', 'artists', 'track_genre', 'cluster', 'distance']].loc[nontop_tracks_df['cluster'] == cluster].sort_values(by='distance', ascending=True)
    recs_df = recs_df.reset_index().drop(columns='index')
    recs_df = recs_df.head(15)
    if recs_df.empty:
        print("empty")
    else:
        print("not empty")
    return recs_df

def gen_recs_list(df):
    lst = []
    for i in range(15):
        dict = {
            'track_id': df['track_id'].values[i],
            'name': df['track_name'].values[i],
            'artists': df['artists'].values[i]
        }

        lst.append(dict)
    return lst

def get_id_list(lst):
    id_lst = []
    for i in range(15):
        id = lst[i]['track_id']
        id_lst.append(id)
    return id_lst

def gen_uri_dict(lst):
    uri_list = []
    for i in range(15):
        uri = "spotify:track:"+lst[i]['track_id']
        uri_list.append(uri)

    dic = {'uris': uri_list}
    return dic

def create_playlist(auth_header, user_id):
    response = requests.post(
        'https://api.spotify.com/v1/users/'+user_id+'/playlists',
        headers = auth_header,
        json = {
            "name": "Top 15 Recommendations",
            "public": "false"
        }
    )

    return response.json()['id']

def add_tracks_to_playlist(auth_header, playlist_id, dic):
    response = requests.post(
        'https://api.spotify.com/v1/playlists/'+playlist_id+'/tracks',
        headers = auth_header,
        json = dic
    )
    return response.json()

def get_track_url(auth_header, track_id):
    response = requests.get(
        'https://api.spotify.com/v1/tracks/'+track_id,
        headers = auth_header
    )
    return response.json()['preview_url']