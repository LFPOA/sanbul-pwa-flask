import tensorflow as tf
from tensorflow import keras
import joblib

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
Bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    avg_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[
            float(form.longitude.data),
            float(form.latitude.data),
            float(form.avg_temp.data),
            float(form.max_temp.data),
            float(form.avg_wind_speed.data),
        ]])
        data = pd.read_csv('./sanbul2district-divby100.csv', sep=',')
        data['burned_area'] = np.log(data['burned_area'] + 1)

        X = data.drop(columns=['burned_area'])
        y = data['burned_area']

        num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
        cat_attribs = ["month", "day"]
        full_pipeline = ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OneHotEncoder(),    cat_attribs),
        ])
        full_pipeline.fit(X)

        df_input = pd.DataFrame([{
            "longitude":      float(form.longitude.data),
            "latitude":       float(form.latitude.data),
            "month":          form.month.data,
            "day":            form.day.data,
            "avg_temp":       float(form.avg_temp.data),
            "max_temp":       float(form.max_temp.data),
            "max_wind_speed": float(form.avg_wind_speed.data),
            "avg_wind":       float(form.avg_wind.data),
        }])
        X_test_prepared = full_pipeline.transform(df_input)

        model = keras.models.load_model('fires_model.keras')
        log_pred = model.predict(X_test_prepared)[0, 0]       
        burned_area_pred = np.exp(log_pred) - 1             

        res = float(np.round(burned_area_pred, 2))

        return render_template('result.html', area_pred=res)

    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()