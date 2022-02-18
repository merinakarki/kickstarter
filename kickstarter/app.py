'''Import Libraries'''
# import os
from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

from kickstarter import app

APP = Flask(__name__)


def create_app():

    # Root route
    @APP.route("/")
    def root():

        return render_template('landing.html')

    @APP.route('/about')
    def about():
        return render_template('index.html')

    @APP.route('/predict/', methods=['GET', 'POST'])
    def predict():

        if request.method == 'POST':
            # Get form data
            goal = request.form.get('goal')
            year = request.form.get('year')
            a = request.form.get('a')
            b = request.form.get('b')
            # c = request.form.get('year')
            d = request.form.get('year')
            e = request.form.get('year')
            duration = request.form.get('duration')
            time_since_last_project = request.form.get(
                'time_since_last_project')
            # month = request.form.get('month')
            currency = request.form.get('currency')
            country = request.form.get('country')
            # main_category = request.form.get('main_category')

            prediction = preprocessDataAndPredict(
                goal, duration, currency, country)

            print(prediction[0])
            return render_template('predict.html', prediction=prediction[0])

    def preprocessDataAndPredict(goal, duration, currency, country):

        test_data = (goal, duration, currency, country)

        print(test_data)

        test_data = np.array(test_data)
        dftest = pd.DataFrame(test_data).T
        dftest.columns = ['country', 'currency', 'duration', 'goal']
        print(dftest)
        print(dftest.shape)

        # test_data = test_data.reshape(1, -1)
        # print(test_data)

        #file = open("model.pkl", "wb")
        model = pickle.load(
            open('kickstarter/kick_model1', 'rb'))
        # model = pickle.load(
        #     open('Kickstarter2/kickstarter/kick_model(1)', 'rb'))

        prediction = model.predict(dftest)

        # print(prediction)

        return prediction

    return APP


if __name__ == '__main__':
    APP.run()
