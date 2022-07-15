from flask import Flask, request
from flask_json import FlaskJSON, JsonError, as_json
from werkzeug.utils import secure_filename
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    pipeline,
    FillMaskPipeline,
)
import numpy as np
import json
from scipy.special import softmax

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = "./"
app.config["APPLICATION_ROOT"] = APP_ROOT
app.config["UPLOAD_FOLDER"] = "files/"
app.config["JSON_ADD_STATUS"] = False
app.config["JSON_SORT_KEYS"] = False


json_app = FlaskJSON(app)


model_path = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, max_length=512)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=10)


def predict_text(content):
    text_predict = fill_mask(content)
    return prepare_output_format(text_predict)


def prepare_output_format(predict):
    list_options = list()
    for result in predict:
        list_options.append({"content": result["token_str"], "score": result["score"]})

    return {"response": {"type": "texts", "texts": list_options}}


@as_json
@app.route("/predict_json", methods=["POST"])
def predict_json():

    data = request.get_json()
    if data.get("type") != "text":
        return generate_failure_response(
            status=400,
            code="elg.request.type.unsupported",
            text="Request type {0} not supported by this service",
            params=[data["type"]],
            detail=None,
        )

    if "content" not in data:
        return invalid_request_error(
            None,
        )

    content = data.get("content")
    try:
        output = predict_text(content)
        return output
    except Exception as e:
        text = "Unexpected error. Possible causes are that your input text may be too long or your input does not contain '[MASK]' element"
        # Standard message for internal error - the real error message goes in params
        return generate_failure_response(
            status=500,
            code="elg.service.internalError",
            text="Internal error during processing: {0}",
            params=[text],
            detail=e.__str__(),
        )


@json_app.invalid_json_error
def invalid_request_error(e):
    """Generates a valid ELG "failure" response if the request cannot be parsed"""
    raise JsonError(
        status_=400,
        failure={
            "errors": [
                {"code": "elg.request.invalid", "text": "Invalid request message"}
            ]
        },
    )


def generate_failure_response(status, code, text, params, detail):
    error = {}
    if code:
        error["code"] = code
    if text:
        error["text"] = text
    if params:
        error["params"] = params
    if detail:
        error["detail"] = {"message": detail}

    raise JsonError(status_=status, failure={"errors": [error]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8866)
