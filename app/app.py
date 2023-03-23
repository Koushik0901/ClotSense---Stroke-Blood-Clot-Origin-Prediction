import os

import pymysql
from engine import InferenceEngine
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flaskext.mysql import MySQL

app = Flask(__name__)
app.config["SAVE_DIR"] = "data"

engine = InferenceEngine()

conn = pymysql.connect(
    host="db",
    user="root",
    password="root",
    database="clotDB",
    port=3306,
)

cursor = conn.cursor()


mysql = MySQL()
# MySQL configurations
app.config["MYSQL_DATABASE_USER"] = "root"
app.config["MYSQL_DATABASE_PASSWORD"] = "root"
app.config["MYSQL_DATABASE_DB"] = "clotDB"
app.config["MYSQL_DATABASE_HOST"] = "db"
mysql.init_app(app)



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["POST", "GET"])
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
    else:
        return render_template("login.html")


@app.route("/register", methods=["POST", "GET"])
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
            conn.commit()
            print(cursor.rowcount, "record inserted.")
            return redirect(url_for("predict"))
    else:
        return render_template("register.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        save_path = os.path.join(app.config["SAVE_DIR"], file.filename)
        file.save(save_path)
        pred_class = engine.predict_image(save_path)
        return jsonify({"class": pred_class})
    else:
        return render_template("predict.html")


@app.route("/about", methods=["GET"])
def about():
    return redirect(url_for("index") + "#about")


@app.route("/contact", methods=["GET"])
def contact():
    return redirect(url_for("index") + "#contact")


@app.route("/logout", methods=["GET"])
def logout():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
