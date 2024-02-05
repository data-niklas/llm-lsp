from pydantic import BaseModel
# Complete class User
# create an instance and then convert to a dict in a method called do_conversion
# Use the method to print the user to the console pretty printed!

class User(BaseModel):
    name: str
    age: int