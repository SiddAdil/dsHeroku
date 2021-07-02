import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask('__name--')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ''' 
    For rendering result on html GUI
    '''
    exp = request.form.get('experience')
    test_score = request.form.get('test_score')
    inter_score = request.form.get('interview_score')

    # convert all the features into int
    int_features = [int(exp), int(test_score), int(inter_score)]
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text = 'Employee salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
