from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import preprocessor

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/make-prediction/")
async def make_prediction_endpoint(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        file_location = f"prediction_data/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        result = preprocessor.make_prediction(file_location)
        baseline_probability = preprocessor.get_probability()
        each_probabilities = preprocessor.get_probabilities_for_each_intervention()
        overall_probability = preprocessor.get_overall_probability_with_all_interventions()
        triggered_interventions = preprocessor.get_triggered_interventions(file_location)
        
        return JSONResponse(status_code=200, content={
            "message": "Prediction made.", 
            "result": result,
            "baseline_probability": baseline_probability,
            "each_probabilities": each_probabilities,
            "overall_probability": overall_probability,
            "triggered_interventions": triggered_interventions
        })
    else:
        return JSONResponse(status_code=400, content={"message": "Invalid file extension."})


@app.post("/train-model/")
async def train_model_endpoint(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        file_location = f"training_data/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        preprocessor.train_model(file_location)
        
        return JSONResponse(status_code=200, content={"message": "Model training started."})
    else:
        return JSONResponse(status_code=400, content={"message": "Invalid file extension."})

os.makedirs('training_data', exist_ok=True)
os.makedirs('prediction_data', exist_ok=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)