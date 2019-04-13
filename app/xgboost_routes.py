import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from utils import onehotCategorical
import io
import base64
import seaborn as sns
import urllib
sns.set_style("dark")
sns.despine()

app = Flask(__name__,template_folder='templates')

def meansales_of_past_years(store,month):
    data=joblib.load("traindatapkl.pkl")
    data=data.loc[:,['Date','Sales',"Store"]]
    def converttoformat(var):
        ln=len(str(var))
        if(ln==1):
            var17="-0"+str(var)+"-2017"
            var16="-0"+str(var)+"-2016"
        else: 
            var17="-"+str(var)+"-2017"
            var16="-"+str(var)+"-2016"
        return var17,var16

    variable17,variable16=converttoformat(month)
    data17=data.loc[(data['Date'].str.contains(variable17)) & (data['Store']==store)]
    data16=data.loc[(data['Date'].str.contains(variable16)) & (data['Store']==store)]
    meansales17=data17['Sales'].mean()
    meansales16=data16['Sales'].mean()
    return meansales17,meansales16
    


@app.route("/")
def index():
    return flask.render_template('index1.html')

@app.route("/home",methods=['POST','GET'])
def home():
    return flask.render_template('index.html')
day=0
month=0
store=0
predictedvalue=0
@app.route('/predict', methods=['POST','GET'])
def make_prediction():
    model = joblib.load('model.pkl')
    data=joblib.load('traindatapkl.pkl')
    global day
    global month
    global store
    global predictedvalue
    if request.method=='POST':
        entered_li = []
        month = request.form['Month']
        date_entry = month
        year, month, day = map(int, date_entry.split('-'))
        promo = int(request.form['Promo'])
        stateH = int(request.form['StateH'])
        schoolH = int(request.form['SchoolH'])
        assortment = int(request.form['Assortment'])
        storeType = int(request.form['StoreType'])
        store = int(request.form['store'])
        
       
        # one-hot encode categorical variables
        stateH_encode = onehotCategorical(stateH, 4)
        assortment_encode = onehotCategorical(assortment, 3)
        storeType_encode = onehotCategorical(storeType, 4)
        store_encode = onehotCategorical(store, 1115, store=1)

        comp_dist = 5458.1
        entered_li.extend(store_encode)
        entered_li.extend(storeType_encode)
        entered_li.extend(assortment_encode)
        entered_li.extend(stateH_encode)
        entered_li.extend([comp_dist])
        #entered_li.extend([promo2])
        entered_li.extend([promo])
        entered_li.extend([day])
        entered_li.extend([month])
        entered_li.extend([schoolH])
        
        data = [[store,1270,promo,schoolH,storeType,assortment,stateH,6,day,month,year,50,132,0,0]]
        df = pd.DataFrame(data,columns=['Store','CompetitionDistance','Promo','SchoolHoliday','StoreType','Assortment',
                                        'StateHoliday','DayOfWeek','Month','Day','Year','WeekOfYear','CompetitionOpen','PromoOpen','IsPromoMonth'])
        #data = [[1,1270,1,0,3,1,0,6,9,15,2019,37,132,0,0]]
        #df = pd.DataFrame(data,columns=['Store','CompetitionDistance','Promo','SchoolHoliday','StoreType','Assortment','StateHoliday','DayOfWeek','Month','Day','Year','WeekOfYear','CompetitionOpen','PromoOpen','IsPromoMonth'])     
        prediction = model.predict(xgb.DMatrix(df))
        prediction=np.expm1(prediction)
        #prediction = model.predict(entered_li.values.reshape(1, -1))
        
        predictedvalue=str(np.squeeze(prediction.round(2)))
        predvalueint=np.squeeze(prediction.round(2))
        label = "$"+predictedvalue
        ########################################################
    meansales,meansales16=meansales_of_past_years(store,month)
    percentincint=((predvalueint-meansales)/(predvalueint))*100
    rawvalue=percentincint.item()
    percentinc=str(np.squeeze(percentincint.round(2))) + "%"
    if rawvalue>0:
        positive=1
    else: 
        positive=0
    return render_template('index.html', label=label,label1=percentinc,label2=positive)

@app.route('/plot',methods=['POST'])
def build_plot():
    meansales,meansales16=meansales_of_past_years(store,month)
    if np.isnan(predictedvalue):
        value = np.nan_to_num(predictedvalue)    
    img = io.BytesIO()
    data=pd.DataFrame({
            'y':[int(meansales16),int(meansales),int(predictedvalue)],
            'x':[str(month)+'/2016',str(month)+'/2017',str(month)+'/2018']})
    #data=data.sort_values(by=['x'])
    plt.figure(figsize=(12,8))
    colors = sns.color_palette("BuGn_r") #BuGn_r,GnBu_d
    ax = sns.barplot(y = 'y', x = 'x',data=data, palette=colors)#, orient='h'

    ax.set_xlabel(xlabel='date', fontsize=16)
    ax.set_ylabel(ylabel='Sales', fontsize=16)
    ax.set_title(label='Comparision with past Sales data', fontsize=20)
    fig = ax.get_figure()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())
    return render_template('test.html',plot_url=plot_url)

if __name__ == '__main__':
    # start API
    app.run()
