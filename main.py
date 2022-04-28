import joblib
from fastapi import FastAPI
from constants import *
from utils import *
from data_model import *


app = FastAPI()
model = joblib.load(MODEL_NAME) # Load from constants

def apply_model(model, inference_data):
	# Prepare prediction dataframe
	inf_df = pd.DataFrame([[inference_data.No_of_Mobile_No, 
						   inference_data.TOTAL_PRODUCTS , 
						   inference_data.Loan Tenure, 
						   inference_data.Loan Amount (Principal),
						   inference_data.AMOUNT_IN_NAIRA,
						   inference_data.Loan Maturity length,
						   inference_data.TRN_Len,
						   inference_data.Loan Disbursement lenth]], columns = ORIGINAL_FEATURES)
	inf_df[ORIGINAL_FEATURES[0]] = inference_data.No_of_Mobile_No
	inf_df[ORIGINAL_FEATURES[1]] = inference_data.TOTAL_PRODUCTS
	inf_df[ORIGINAL_FEATURES[2]] = inference_data.Loan Tenure
	inf_df[ORIGINAL_FEATURES[3]] = inference_data.Loan Amount (Principal)
	inf_df[ORIGINAL_FEATURES[4]] = inference_data.AMOUNT_IN_NAIRA
	inf_df[ORIGINAL_FEATURES[5]] = inference_data.Loan Maturity length
	inf_df[ORIGINAL_FEATURES[11]] = inference_data.TRN_Len
	inf_df[ORIGINAL_FEATURES[12]] = inference_data.Loan Disbursement lenth
	
	processed_inference_data = apply_pre_processing(inf_df)
	pred = model.predict(processed_inference_data)[0]
	
	if pred == 1:
		pred_value = True
		pred_class = "Bad Loan"
	else:
		pred_value = False
		pred_class = "Not Bad loan"

	return pred_value, pred_class

@app.get('/')
def get_root():

	return {'message': 'Welcome to the Heart Disease Detection API'}
	
@app.post("/predict", response_model=OutputDataModel)
async def post_predictions(inference_data: InputDataModel):

    pred_value, pred_class = apply_model(model, inference_data)

    response = {
        "predicted_value": pred_value,
        "predicted_class": pred_class
    }
    return response