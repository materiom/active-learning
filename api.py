from fastapi import FastAPI
import uvicorn
import json
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import GPy
import emukit
from numpy.random import seed
import datetime, psutil
import asyncio
from easycharts import ChartServer
from easyschedule import EasyScheduler

seed(12345)

app = FastAPI()

data = {}

@app.on_event("startup")
async def startup_event():
    """asyncio.create_task(scheduler.start())
    app.charts = await ChartServer.create(
        app,
        charts_db = "charts_database",
        chart_prefix = '/mycharts'
    )
    
    time_now = datetime.datetime.now().isoformat()[11:19]
    
    await server.charts.create_dataset(
        'cpu',
        labels=[time_now],
        dataset=[psutil.cpu_percent()]
    )
    
    await server.charts.create_dataset(
        'mem',
        labels=[time_now],
        dataset=[psutil.virtual_memory().percent]
    )
    """
    f = lambda xi: np.sin(xi * 2) + .2 * xi
    
    N = 100
    num_measurements = 5
    X_grid = np.linspace(-np.pi, np.pi * 3/ 4, N)[:, None]
    Y_grid = f(X_grid)
    Y_grid.shape
    
    X_samples = np.random.uniform(-np.pi, np.pi * 3/4, (num_measurements, 1))
    Y_samples = f(X_samples) + np.random.normal(0., .1, (X_samples.shape[0], 1))
    k = GPy.kern.RBF(1, lengthscale=1, variance=1)
    m = GPy.models.GPRegression(X_samples, Y_samples, k)
    
    m.optimize('bfgs', max_iters=100)
    
    mean,Cov = m.predict(X_grid, full_cov=True)
    
    data[1] = mean
    data[2] = Cov
    return data

scheduler = EasyScheduler()
server = FastAPI()
every_minute = '* * * * *'

@app.get("/")
def home():
    lists = data[1].tolist()
    json_str = json.dumps(lists)
    lists1 = data[2].tolist()
    json_str1 = json.dumps(lists1)
    return {'Data': {1:json_str, 2: json_str1}}

if __name__ == "__main__":
    uvicorn.run("__main__:app", host = "0.0.0.0", port=8000, reload=True, workers=2)
    
