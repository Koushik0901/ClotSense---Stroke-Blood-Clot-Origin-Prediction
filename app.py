import mysql.connector
from flask import Flask, jsonify, redirect, render_template, request, url_for

from engine import InferenceEngine

app = Flask(__name__)
engine = InferenceEngine()
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    database="bloodClot",
    auth_plugin="mysql_native_password",
)
cursor = mydb.cursor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    if request.method == "POST":
        user = request.form["email"]
        password = request.form["password"]
        cursor.execute(
            "SELECT * FROM users WHERE email = %s AND password = %s", (user, password)
        )
        results = cursor.fetchall()
        if len(results) != 0:
            return redirect(url_for("predict"))
        else:
            login_error = "Invalid email or password"
            return render_template("index.html", login_error=login_error)


@app.route("/register", methods=["POST"])
def register():
    if request.method == "POST":
        user = request.form["email"]
        password = request.form["password"]
        cursor.execute("SELECT * FROM users WHERE email = %s", (user,))
        results = cursor.fetchall()
        if len(results) != 0:
            register_error = "User already exists"
            return render_template("index.html", register_error=register_error)
        else:
            cursor.execute(
                f"INSERT INTO users (email, password) VALUES ('{user}', '{password}')"
            )
            mydb.commit()
            print(cursor.rowcount, "record inserted.")
            return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        file.save(file.filename)
        print(file.filename)
        # pred_class = engine.predict_image(file.filename)
        pred_class = "CE"
        return jsonify({"class": pred_class})
    else:
        return render_template("upload.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
