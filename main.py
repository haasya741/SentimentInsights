# main.py

import argparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gl_classifier import (
    preprocess_data,
    describe_data,
    balance_data,
    train_and_save_model,
    load_and_evaluate_model,
    predict_gl_code,
)
from similarity import create_embeddings, query_embeddings
import pandas as pd

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictionRequest(BaseModel):
    description: str
    method: str = "tfidf_svm"
    k: int = 5


@app.get("/")
async def get_home():
    return FileResponse("index.html")


@app.get("/train")
async def get_home():
    return FileResponse("test.html")


@app.get("/test")
async def get_home():
    return FileResponse("test.html")


@app.post("/find")
def predict(request: PredictionRequest):
    try:
        results = query_embeddings.query_embeddings(request.description, request.k)
        # results.reverse()
        # the score is L2 distance (between 2 to 0), the smaller is better.
        # to make it normalized in 0 to 1 scale (with 1 being better), we do some basic normalization
        return list(
            map(
                lambda doc: {
                    "score": 1 - doc[1].item() / 2,
                    "meta": doc[0].metadata,
                    "item": doc[0].page_content,
                },
                results,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(request: PredictionRequest):
    gl_model_path = "GL_Code_model"
    hsn_model_path = "HSN_Code_model"
    method = request.method

    if method == "embedding_lstm":
        gl_model_path += ".h5"
        hsn_model_path += ".h5"
    else:
        gl_model_path += ".joblib"
        hsn_model_path += ".joblib"

    try:
        gl_prediction, gl_confidence = predict_gl_code(
            gl_model_path, request.description, method
        )
        hsn_prediction, hsn_confidence = predict_gl_code(
            hsn_model_path, request.description, method
        )
        return {
            "gl_prediction": gl_prediction,
            "gl_confidence": gl_confidence,
            "hsn_prediction": hsn_prediction.item(),
            "hsn_confidence": hsn_confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="GL Code Classifier")
    parser.add_argument(
        "--embeddings", action="store_true", help="Create vector embeddings"
    )
    parser.add_argument("input", help="Query to find", nargs="?", default="")
    parser.add_argument("--find", action="store_true", help="Query the embeddings")
    parser.add_argument("--top", type=int, help="Top k items to return", default=3)
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument(
        "--type",
        type=str,
        choices=["HSN_Code", "GL_Code"],
        default="HSN_Code",
        help="Training type",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tfidf_svm", "embedding_lstm"],
        default="tfidf_svm",
        help="Method to use for classification",
    )
    args = parser.parse_args()

    if args.embeddings:
        create_embeddings.create_embeddings(args.input)
    elif args.find:
        results = query_embeddings.query_embeddings(args.input, int(args.top))
        for doc, score in results:
            print(
                f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}"
            )
    elif args.train:
        train_df = pd.read_csv("./data/hsn_train.csv")
        describe_data(train_df, "Item_Description", args.type)
        train_df = preprocess_data(train_df, "Item_Description")
        describe_data(train_df, "Item_Description", args.type)
        # train_df = balance_data(train_df, args.type)
        model_path = args.type + "_model"
        if args.method == "embedding_lstm":
            model_path += ".h5"
        else:
            model_path += ".joblib"
        train_and_save_model(
            train_df, "Item_Description", args.type, model_path, method=args.method
        )
    elif args.test:
        test_df = pd.read_csv("./data/hsn_test.csv")
        test_df = preprocess_data(test_df, "Item_Description")
        # describe_data(train_df, 'Item_Description', args.type)
        model_path = args.type + "_model"

