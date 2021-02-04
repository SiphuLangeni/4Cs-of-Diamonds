import uvicorn
from fastapi import FastAPI
from model import Diamond, DiamondModel


app = FastAPI()
model = DiamondModel()


@app.get('/')
def index():
    return {'message': 'Diamond Price Prediction is ready.'}

@app.post('/predict')
def predict_price(diamond: Diamond):
    diamond = diamond.dict()
    prediction = model.predict_price(
        diamond['cut'], diamond['colour'], diamond['clarity'], diamond['carat']
    )
    return {
        'prediction': prediction
    }

@app.post('/metrics')
def get_metrics():
    mae, mape, rmse, rmspe = model.get_metrics()

    return {
        'MAE': int(mae),
        'MAPE': round(mape, 2),
        'RMSE': int(rmse),
        'RMSPE': round(rmspe, 2)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
     