from pydantic import BaseModel
# Complete class User, create an instance and then convert to a dict
# Print each attribute to the stdout, then send the age to https://localhost:8081/age with POST if the env variable BOB is set

class User(BaseModel):
    name: str
    age: int
    