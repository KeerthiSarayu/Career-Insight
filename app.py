from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
import pickle

app = Flask(__name__)

# Load the trained model
clf = pickle.load(open("trained_model.pkl", 'rb'))  # Update with the actual filename

with open('label_encoder.pkl', 'rb') as le_file:
    labelencoder = pickle.load(le_file)

# features = ['CGPA', 'WebDev',
#        'DataAnalysis', 'ReadWrite',
#        'TechPerson', 'NonTechSociety', 
#        'Coding', 'MobileApps',
#        'Communication',
#        'Security',
#        'LargeDB',
#        'Stats',        
#        'English', 'Event',
#        'TechBlogs', 'Marketing',  
#        'ML', 'Connections', 
#        'LiveProject']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    CGPA = request.form.get('CGPA')
    WebDev = 1 if request.form.get('WebDev') == "Yes" else 0

    # Assuming 'Done WebDev' to 'Build live project' are the features
    # input_data = np.array([
    #     request.form.get('CGPA'),
    #     request.form.get('Done WebDev'),
    #     request.form.get('Data Analysis'),
    #     request.form.get('reading & writing skills'),
    #     request.form.get('Tech person'),
    #     request.form.get('In a non-tech society'),
    #     request.form.get('Good at Coding'),
    #     request.form.get('Developed mobile app'),
    #     request.form.get('Good at communication'),
    #     request.form.get('Specialization in security'),
    #     request.form.get('Handled large databases'),
    #     request.form.get('Have knowledge of Statistics and Data Science'),
    #     request.form.get('Proficient in English'),
    #     request.form.get('Managed an event'),
    #     request.form.get('Wrote tech blogs'),
    #     request.form.get('Like marketing'),
    #     request.form.get('ML expert'),
    #     request.form.get('Lot of connections'),
    #     request.form.get('Build live project')
    # ])

    # # Reshape the input to ensure it's a 2D array
    # input_data = input_data.reshape(1, -1)
    # print(input_data)

    # result = clf.predict(input_data)[0]

    input = np.array([[CGPA]])
    result = clf.predict(input)[0]

    # return jsonify({'Career': result})
    result = labelencoder.inverse_transform([result])[0]

    return render_template('output.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
