from pydantic import BaseModel
from decimal import Decimal


class Predict(BaseModel):
    benz_92: Decimal
    benz_95: Decimal
    benz_98: Decimal
