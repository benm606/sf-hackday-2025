from flask import Flask, render_template
import sys
import os
# Add the parent directory to the Python path so we can import from db/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.vector_db import VectorDB

app = Flask(__name__)

@app.route('/')
def index():
    db = VectorDB()
    clusters = db.cluster_problems()
    return render_template('index.html', clusters=clusters)

if __name__ == '__main__':
    app.run(debug=True) 