from fastapi import FastAPI
import strawberry
from strawberry.asgi import GraphQL
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from static.models.all_models import predict

@strawberry.type
class Question:
    question_id: int
    question: str

@strawberry.type
class Answer:
    question_id: int
    answer: str

@strawberry.type
class Grade:
    question_id: int
    answer: str
    grade: int

@strawberry.type
class Mutation:
    @strawberry.mutation
    def recover_answer(self, question_id: int, answer: str) -> Grade:
        grade = predict(question_id, answer)
        new_answer = Grade(question_id=question_id, answer=answer, grade=grade)
        return new_answer

@strawberry.type
class Query:
    @strawberry.field
    def questions(self) -> List[Question]:
        return [
            Question(
                question_id=1,
                question="ماهي أركان الإسلام؟",
            ),
            Question(
                question_id=2,
                question="ماهي أركان الإيمان؟",
            ),
            Question(question_id=3, question="ماهي فرائض الوضوء وسننه؟"),
            Question(
                question_id=4, question="ما اسم الملك المكلف بالنفخ في الصور؟"
            ),
            Question(question_id=5, question="ما هو الكتاب المنزل على عيسى عليه السلام؟"),
            Question(
                question_id=6, question="من كفل النبي صلى الله عليه وسلم بعد وفاة جده عبد المطلب؟"
            ),
            Question(
                question_id=7,
                question="كم كان عمر رسول الله صلى الله عليه وسلم عندما نزل عليه الوحي ؟",
            ),
            Question(
                question_id=8, question="ما اسم الملك الذي جاء بالوحي؟"
            ),
            Question(
                question_id=9,
                question="ما أول ما نزل من القرآن الكريم ؟",
            ),
            Question(question_id=10, question="كم دامت الدعوة السرية ؟"),
        ]



schema = strawberry.Schema(query=Query, mutation=Mutation)

graphql_app = GraphQL(schema)
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route("/", graphql_app)
app.add_websocket_route("/", graphql_app)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"An error occurred: {e}")