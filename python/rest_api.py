#!flask/bin/python
from flask import Flask,jsonify,request
from flask_cors import CORS
import call_model


app = Flask(__name__)
CORS(app)

@app.route('/forecast' , methods=['GET'])
def index():
    date1 = request.args.get('forecast_date')
    model = request.args.get('model')
    return jsonify({'forecast': str(call_model.forecast(date1,model))})
    
if __name__ == '__main__' :
    app.run()