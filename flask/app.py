from flask import Flask, render_template, request, redirect, session, make_response, url_for
from flask_session import Session
import recs
import pandas as pd

app = Flask(__name__)
app.config.from_pyfile('config.py')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

@app.route('/auth')
def auth():
    return redirect(recs.AUTH_URL)

@app.route('/callback')
def callback():
    auth_token = request.args['code']
    auth_header = recs.authorize(auth_token)
    session['auth_header'] = auth_header
    return profile()

def valid_token(resp):
    return resp is not None and not 'error' in resp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/go', methods=['GET', 'POST'])
def go():
    session.clear()
    session['num_tracks'] = '20'
    session['time_range'] = 'short_term'
    base_url = recs.AUTH_URL
    response = make_response(redirect(base_url), 302)
    return response

@app.route('/profile')
def profile():
    data_df = recs.get_data()
    if 'auth_header' in session:
        auth_header = session['auth_header']
        top_tracks_df = recs.get_top_tracks(auth_header)
        if "failed" not in top_tracks_df:
            top_tracks_features_df = recs.get_top_tracks_features(auth_header, top_tracks_df)
            nontop_tracks_df = recs.get_nontop_tracks(data_df, top_tracks_df)
            features_df = recs.get_data_features(nontop_tracks_df)
            avg_vector, top_tracks_features_df = recs.get_weighted_vector(top_tracks_features_df)
            nontop_tracks_df = recs.find_clusters(nontop_tracks_df, features_df)
            cluster = recs.predict(avg_vector)
            nontop_tracks_df = recs.get_distance(nontop_tracks_df, avg_vector)
            session['cluster'] = cluster
            print(session['cluster'])
            session['nontop_tracks_dict'] = nontop_tracks_df.to_dict('records')
    return redirect(url_for('recommendations'))

@app.route('/recommendations', methods=['GET'])
def recommendations():
    if 'auth_header' in session:
        auth_header = session['auth_header']
        metric = request.args.get('metric', 'NA')
        nontop_tracks_df = pd.DataFrame(session['nontop_tracks_dict'])
        cluster = session['cluster']
        recs_df = recs.get_recs(nontop_tracks_df, cluster, metric)
        tracks_list = recs.gen_recs_list(recs_df)
        track_ids = recs.get_id_list(tracks_list)
        try:
            for i in range(len(tracks_list)):
                tracks_list[i]['track_url'] = recs.get_track_url(auth_header, track_ids[i])
                tracks_list[i]['msg'] = 'succeed'
        except:
            tracks_list[i]['msg'] = 'failed'
        session['tracks_list'] = tracks_list
        status = "success"
        return render_template("done.html", tracks_list = tracks_list, status = status, metric=metric)
    return render_template('done.html')

@app.route('/create', methods=['POST'])
def create():
    if 'auth_header' in session:
        auth_header = session['auth_header']
        user_id = recs.get_user_profile(auth_header)
        tracks_list = session.get('tracks_list')
        uri_dict = recs.gen_uri_dict(tracks_list)

        if request.method == 'POST':
            try:
                playlist_id = recs.create_playlist(auth_header, user_id)
                recs.add_tracks_to_playlist(auth_header, playlist_id, uri_dict)
                msg = 'succeed'
                message = "Playlist created."
                status = 'success'
                return render_template('done.html', tracks_list=tracks_list, uri_dict=uri_dict, message=message, msg=msg, status=status)
            except:
                message = "Failed to create playlist."
                msg = "Failed"
                status = "Failure"
                return render_template('done.html', tracks_list=tracks_list, message=message, msg=msg, status=status)
    return render_template('done.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html') 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)