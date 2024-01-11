
from flask import Flask, request, render_template
from run import TextPredictionModel
from waitress import serve
import threading

app = Flask(__name__, template_folder="../templates")


model = TextPredictionModel.from_artefacts(
    r"../../train/data/artefacts/2024-01-09-12-20-51")


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def predict():
    model_input = request.form.get('model_input')
    num_predictions = int(request.form.get('num_predictions', 1))
    preds = model.predict([model_input], top_k=num_predictions)

    return render_template('results.html', predictions=preds)


def run_server():
    serve(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    print("Server starting...")
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    print("Server started")
    print("Navigate to http://127.0.0.1:5000 to use the app")
