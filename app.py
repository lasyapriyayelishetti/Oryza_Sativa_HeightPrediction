# #to run use python -m flask run

#we cant define same function more than
#debug =true used in app.route() helps in updating the website whenever we make any changes in the file.
#py -3 -m venv .venv
#.venv\scripts\activate

from flask import Flask, render_template, request
import csv
from utils import data_processing

app = Flask(__name__)

model_trained = False  # Flag to check if the model has been trained

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        input_sequence = request.form['sequence']
        predicted_subpopulation, predicted_height = data_processing.predict_height(input_sequence)
        
        save_to_history(input_sequence, predicted_subpopulation, predicted_height)
        return render_template('output.html', input_sequence=input_sequence, predicted_subpopulation=predicted_subpopulation, predicted_height=predicted_height)
    return render_template('prediction.html')

def read_csv_file(file_path):
    csv_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            csv_data.append(row)
    return csv_data

@app.route('/data')
def display_csv():
    csv_path1 = 'static/plantheight.csv'  # Replace with the actual path to your first CSV file
    csv_data1 = read_csv_file(csv_path1)
    
    csv_path2 = 'static/cluster_data.csv'  # Replace with the actual path to your second CSV file
    csv_data2 = read_csv_file(csv_path2)
    
    return render_template('data.html', csv_data1=csv_data1, csv_data2=csv_data2)

@app.route('/history')
def history():
    history_data = get_history_data()
    if not history_data:
        no_data_available = True
    else:
        no_data_available = False
    return render_template('history.html', history_data=history_data, no_data_available=no_data_available)

def get_history_data():
    history_data = []
    with open('history.csv', 'r+') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            history_data.append(row)
    return history_data

def save_to_history(input_sequence, predicted_subpopulation, predicted_height):
    with open('history.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == input_sequence:
                return  # Input sequence already exists in the file, so no need to write it again

    with open('history.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([input_sequence, predicted_subpopulation, predicted_height])

if __name__ == '__main__':
    data_processing.train_model()  # Train the model when the app starts
    app.run(debug=True)
