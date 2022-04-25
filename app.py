from glob import glob
from unittest import result
from flask import Flask, request, jsonify
import json
from flask_restful import Resource, Api
from rec import *

app = Flask(__name__)
api = Api(app)


# def write_json():
#     with open('recommendation.json', 'w') as f:
#         json.dump(recommendation, f)


class HelloWorld(Resource):
    def get(self):
        return "Welcome to Book Recommender System"


class PopularityBased(Resource):
    def get(self):
        # inputchr = str(request.args['query'])
        result = pop_based_colle['Book-Title'].to_list()
        with open('popularity_based.json', 'w') as f:
            json.dump(result, f)
        popular_based = result
        return popular_based


class ContentBased(Resource):
    def get(self):
        inputchr = str(request.args['query'])
        result = content_based(inputchr)
        with open('content_based.json', 'w') as f:
            json.dump(result, f)
        return result


class Collaborative(Resource):
    def get(self):
        inputchr = str(request.args['query'])
        collaborative = getTopRecommandations(m[k.index(inputchr)])
        # with open('collaborative.json', 'w') as f:
        #     json.dump(collaborative, f)
        return jsonify(collaborative)


api.add_resource(PopularityBased, '/recommendation/popular-based')
api.add_resource(ContentBased, '/recommendation/content-based')
api.add_resource(Collaborative, '/recommendation/collaborative')
api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)
