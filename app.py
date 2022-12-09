from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K


app = Flask(__name__)

app.config['DEBUG'] = False
app.config['SECRET_KEY'] = 'Thisisasecret'

@app.route('/risk', methods=['GET', 'POST'])
def risk():
    model = pickle.load(open('finalized_model.sav', 'rb'))
    # print(x)
    age = request.form['age']
    edulevel = request.form['edulevel']
    married = request.form['married']
    kids = request.form['kids']
    occupation = request.form['occupation']
    income = request.form['income']
    networth = request.form['networth']
    risk = request.form['risk']
    data = [age, edulevel, married, kids, occupation, income, networth, risk]
    data = np.array(data)
    prediction = model.predict([data])
    return jsonify(riskintolerance=prediction[0])


@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    x = []
    y = []
    if request.method == 'POST':
        risk = request.form['risk']
        risk = float(risk)
        class Model:
            def __init__(self,risk):
                self.data = None
                self.model = None
                self.std = None
                self.risk = risk

            def __build_model(self, input_shape, outputs):
                '''
                Builds and returns the Deep Neural Network that will compute the allocation ratios
                that optimize the Sharpe Ratio of the portfolio

                inputs: input_shape - tuple of the input shape, outputs - the number of assets
                returns: a Deep Neural Network model
                '''
                model = Sequential([
                    LSTM(64, input_shape=input_shape),
                    Flatten(),
                    Dense(outputs, activation='softmax')
                ])

                def sharpe_loss_with_risk(_, y_pred):
                    # make all time-series start at 1
                    data = tf.divide(self.data, self.data[0])  
                    x = self.data.numpy()
                    x = pd.DataFrame(x)
                    x = x.ewm(span=50).std()
                    x = x.to_numpy()
                    for i in range(0,x.shape[1]):
                        x[1:,i] = x[1:,i]/x[-1,i]
                    # print(np.linalg.norm(x[1:]))
                    # print(x[-1,0])

                    # value of the portfolio after allocations applied
                    # 
                    portfolio_values = tf.reduce_sum(tf.multiply(risk,tf.divide(tf.multiply(data[1:], y_pred),x[1:])), axis=1) 
                    # portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 

                    portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

                    sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

                    # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
                    #   we can negate Sharpe (the min of a negated function is its max)
                    return -sharpe
                def sharpe_loss_without_risk(_, y_pred):
                    # make all time-series start at 1
                    data = tf.divide(self.data, self.data[0]) 
                    # value of the portfolio after allocations applied
                    # 
                    portfolio_values = tf.reduce_sum(tf.multiply(data[1:], y_pred), axis=1) 
                    # portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 

                    portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

                    sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

                    # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
                    #   we can negate Sharpe (the min of a negated function is its max)
                    return -sharpe
                if self.risk:
                    model.compile(loss=sharpe_loss_with_risk, optimizer='adam')
                else:
                    model.compile(loss=sharpe_loss_without_risk, optimizer='adam')
                return model

            def get_allocations(self, data: pd.DataFrame):
                '''
                Computes and returns the allocation ratios that optimize the Sharpe over the given data

                input: data - DataFrame of historical closing prices of various assets

                return: the allocations ratios for each of the given assets
                '''

                # data with returns
                data_w_ret = np.concatenate([ data[1:], data.pct_change()[1:] ], axis=1)

                data = data.iloc[1:]
                self.data = tf.cast(tf.constant(data), float)

                if self.model is None:
                    self.model = self.__build_model(data_w_ret.shape, len(data.columns))

                fit_predict_data = data_w_ret[np.newaxis,:]        
                self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=5, shuffle=False)
                return self.model.predict(fit_predict_data)[0]
        findata = pd.read_csv("findata.csv")
        # findata = findata.to_numpy()
        findata = findata.astype('float64')
        m = Model(risk = True)
        x = m.get_allocations(data = findata)
        x = list(x)
        x = [str(i) for i in x]
        m = Model(risk = False)
        y = m.get_allocations(data = findata)
        y = list(y)
        y = [str(i) for i in y]
    print(x)
    print(y)
    return jsonify(allocations_risk = x,allocations_without_risk=y)
    
if __name__ == '__main__':
    app.run()