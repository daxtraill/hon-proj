from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return "Scaffold Selection Home"

if __name__ == '__main__':
    app.run(debug=True)


def load_data():
    return pd.read_csv('/Volumes/dax-hd/project-data/search-files/merged-data.csv')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    keyword = data.get('keyword', '')
    min_value = data.get('min_value', None)
    max_value = data.get('max_value', None)
    
    # Assuming you have a function to load your CSV data into a DataFrame
    df = load_data()
    
    # Implement your search logic here
    # For example, filter by keyword and a numeric range
    results = df[(df['column_name'].str.contains(keyword, case=False)) & 
                 (df['numeric_column'] >= min_value) & 
                 (df['numeric_column'] <= max_value)]
                 
    return jsonify(results.to_dict(orient='records'))