from fastapi import FastAPI
import random
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/random")
async def random_number():
    return random.randint(0, 100)


class DataModel(BaseModel):
    name: str
    age: int
    email: str

@app.post("/adddata")
async def add_data(data: DataModel):
    
    
    
    
    
    return {"message": "Data added successfully", "data": data}