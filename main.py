from fastapi import FastAPI

app = FastAPI()

items_db = []

@app.get("/")
def read_root():
    return {"Hello": "World"}

# CREATE
@app.post("/items")
def create_item(item: dict):
    items_db.append(item)
    return item

# GET
@app.get("/items")
def get_items():
    return items_db

# UPDATE (PUT)
@app.put("/items/{index}")
def update_item(index: int, item: dict):
    if index >= len(items_db):
        return {"error": "Not found"}
    items_db[index] = item
    return item

# DELETE
@app.delete("/items/{index}")
def delete_item(index: int):
    if index >= len(items_db):
        return {"error": "Not found"}
    return items_db.pop(index)