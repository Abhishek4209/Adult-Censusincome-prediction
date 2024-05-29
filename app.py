from flask import Flask,render_template,request
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.logger import logging
# from src.pipelines.prediction_pipeline import PredictPipeline
# from src.pipelines.prediction_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')

def home_page():
    return render_template("index.html")
    logging.info("index page loaded")
@app.route('/predict',methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("predict.html")
    
    
    else:
        data=CustomData(
            age=float(request.form.get('age')),
            workclass = (request.form.get('workclass')),
            fnlwgt = float(request.form.get('fnlwgt')),
            education = (request.form.get('education')),
            marital_status= (request.form.get('marital_status')),
            occupation = (request.form.get('occupation')),
            relationship = (request.form.get('relationship')),
            race = (request.form.get('race')),
            sex = (request.form.get('sex')),
            capital_gain = float(request.form.get('capital_gain')),
            capital_loss = float(request.form.get('capital_loss')),
            hours_per_week = float(request.form.get('hours_per_week')),
            country= (request.form.get('country'))    
                                
        )


    final_new_data=data.get_data_as_dataframe()
    Predict_pipeline=PredictPipeline()
    pred=Predict_pipeline.predict(final_new_data)
    
    # PredictPipeline
    results=pred

    return render_template("predict.html",final_result=results)


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
