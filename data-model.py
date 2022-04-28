from pydantic import BaseModel
from typing import Optional

class InputDataModel(BaseModel):
	No_of_Mobile_No: float
	TOTAL_PRODUCTS: int
	Loan Tenure: int
	Loan Amount (Principal): int
	AMOUNT_IN_NAIRA: float
	Relationship_Start_length: int
	Loan Disbursement lenth: int
	Loan Maturity length: int
	No_of_Mobile_No: int
	
class OutputDataModel(BaseModel):
    predicted_value: bool
    predicted_class: str