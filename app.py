from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    data = vectorizer.transform([text])
    result = model.predict(data)[0]

    emoji = ""
    if result == "positive":
        emoji = "😊"
    elif result == "negative":
        emoji = "😡"
    else:
        emoji = "😐"

    return render_template("index.html", prediction=result, emoji=emoji)

if __name__ == "__main__":
    app.run(debug=True)
    import sqlite3

conn = sqlite3.connect("data.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS results (text TEXT, sentiment TEXT)")
cursor.execute("INSERT INTO results VALUES (?, ?)", (text, result))
conn.commit()
import matplotlib.pyplot as plt

labels = ['Positive', 'Negative', 'Neutral']
values = [10, 5, 3]

plt.bar(labels, values)
plt.title("Sentiment Analysis Result")
plt.show()