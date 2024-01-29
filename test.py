from pydantic import BaseModel
# Complete class User, create an instance and then convert to a dict

class User(BaseModel):
    name: str
    age: int