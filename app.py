import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
import numpy as np
import armagarch as ag
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


st.title("ARIMA-GARCH Model")
st.write("请选择要预测的时长")
head = st.slider('时间', 1, 365, 200)

df = pd.read_excel("daily.xlsx")
df.drop("Unnamed: 0",axis=1,inplace=True)
ff2 = df.sort_values(by='date', ascending=True)
ff3 = ff2.reset_index().drop('index', axis=1)
# define mean, vol and distribution
meanMdl = ag.ARMA(order = {'AR':1,'MA':0})
volMdl = ag.garch(order = {'p':1,'q':1})
distMdl = ag.normalDist()

# create a model
model = ag.empModel(ff3['price'].to_frame(), meanMdl, volMdl, distMdl)
# fit model
model.fit()

# get the conditional mean
Ey = model.Ey
#st.line_chart(Ey)
# get conditional variance
ht = model.ht
cvol = np.sqrt(ht)
#st.line_chart(cvol)

# get standardized residuals
stres = model.stres
#st.line_chart(stres)

# make a prediction of mean and variance over next 3 days.
pred = model.predict(nsteps = head)

pp = pd.DataFrame(pred[0])

pp.columns = ['price']

#dd = ff3.tail(1)['price'].values

xx = pd.concat([ff3['price'].to_frame(),pp],axis=0, ignore_index=True)


option = {
  "xAxis": {
    "type": 'category',
    "data": list(xx.index),
  },
  "yAxis": {
    "type": 'value'
  },
  "series": [
    {
      "data": list(xx['price']),
      "type": 'line'
    }
  ]
};

st_echarts(option,height='400px')




# plot in three subplots
import datetime

#start = datetime.datetime(2024,03,27)
#result_date1 = start + datetime.timedelta(days=head)

dfa = pd.DataFrame(pred)
dfa.index = ['均值(预测值)','预测波动率']
st.dataframe(dfa)



#st.write(pred)
# pred is a list of two-arrays with first array being prediction of mean
# and second array being prediction of variance
