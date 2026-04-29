from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class TrainRequest(BaseModel):
    question: str
    sql: str