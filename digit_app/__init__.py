from flask import Flask

app = Flask(__name__)

import digit_app.navigation
import digit_app.backend.main
