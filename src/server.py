from flask import Flask, render_template, json, request
import flask
from flask_bootstrap import Bootstrap
import numpy as np

app = Flask(__name__)
Bootstrap(app)

@app.route('/initState', methods=['POST'])
def initState():
    # state = np.array(
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    #         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53])
    state = np.array(
        [-1, -1, 8, 1, 4, 7, 0, 3, 6, 11, -1, 17, 10, 13, 16, 9, 12, 15, 20, 23, 26, 19, 22, 25, 18, 21, 24, 29, 32, 35,
        28, 31, 34, 27, 30, 33, 42, 39, 36, -1, 40, 37, 44, -1, 38, 47, 50, 53, 46, 49, 52, 45, 48, 51]
    )
    state = state.tolist()
    state = json.dumps(state)
    response = flask.jsonify({"state": state})
    response.headers.add('Access-Control-Allow-Origin', '*')
    print(response)
    return response

app.run(debug=True)
