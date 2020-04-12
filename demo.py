__author__ = 'nittella'
from sentiment_classifier import SentimentClassifier
from codecs import open
import time
from flask import Flask, render_template, request
app = Flask(__name__)


print("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print("Classifier is ready")
print(time.time() - start_time, "seconds")

@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message="", probability_message=""):
    if request.method == "POST":
        text = request.form["text"]
        logfile = open("ydf_demo_logs.txt", "a", "utf-8")
        print(text)
        logfile.write("<response>")
        logfile.write(text)
        prediction_message, probability_message = classifier.get_prediction_message(text)
        print(prediction_message)
        logfile.write(prediction_message)
        logfile.write("</response>")
        logfile.close()

    return render_template('hello.html',
                           text=text,
                           prediction_message=prediction_message,
                           probability_message=probability_message)


if __name__ == "__main__":
    app.run(port=80, debug=False)
