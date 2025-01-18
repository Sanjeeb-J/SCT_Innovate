from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open("Naive_model.pkl", "rb"))

@app.route('/', methods=["GET", "POST"])
def main_function():
    if request.method == "POST":
        # Get input from form
        email_content = request.form['email']
        # Prepare input for the model
        prediction = pipe.predict([email_content])[0]
        # Render the result page with the prediction
        return render_template("show.html", prediction=prediction)
    else:
        # Render the input form page
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
