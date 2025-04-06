from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['news']
    result = predict_news(user_input)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
