from pydantic import BaseModel
# 1. Complete the class User with additional attributes
# 2. Create an instance of the User
# 3. Convert the user to a Python dictionary
# 4. Print its attributes

class User(BaseModel):
    name: str
    age: int
