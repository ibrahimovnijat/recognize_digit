from digit_app import app 
from flask import render_template, request, url_for, redirect
from flask import send_file, current_app, send_from_directory, jsonify, make_response, send_file
import webbrowser
import werkzeug


@app.route("/", methods=["GET", "POST"])
def recognize_digit_app():
    return render_template("digit_recognize.html")
