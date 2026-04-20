import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_123")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///app.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    searches = db.relationship('SearchHistory', backref='user', lazy=True)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.query(User).get(int(user_id))

with app.app_context():
    db.create_all()


VECTORSTORE_PATH = "faiss_index_movies"


embeddings = CohereEmbeddings(
    model="embed-english-v3.0"
)


llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0
)

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    return None

vectorstore = load_vectorstore()

@app.route('/')
def index():
    recent_searches = []
    if current_user.is_authenticated:
        recent_searches = db.session.query(SearchHistory).filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).limit(5).all()
    return render_template('index.html', recent_searches=recent_searches)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if db.session.query(User).filter_by(username=username).first():
            return render_template('signup.html', error="Username already exists")
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = db.session.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():
    if not vectorstore:
        return jsonify({"error": "Database not found."}), 500
    
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    if not current_user.is_authenticated:
        guest_searches = session.get('guest_searches', 0)
        if guest_searches >= 2:
            return jsonify({
                "error": "You've reached the guest limit (2 searches). Please <a href='/signup' class='underline'>sign up</a> or <a href='/login' class='underline'>login</a> to continue.",
                "require_login": True
            }), 403
        session['guest_searches'] = guest_searches + 1
    else:
        new_search = SearchHistory(user_id=current_user.id, query=user_query)
        db.session.add(new_search)
        db.session.commit()

    try:
        results = vectorstore.similarity_search(user_query, k=5)
        context = "\n\n".join(doc.page_content for doc in results)

        prompt = f"""
        You are a movie expert.
        QUESTION: Find the movie titled "{user_query}" and provide a short summary.
        If not found, say "I couldn't find that movie."
        CONTEXT: {context}
        """
        response = llm.invoke(prompt)
        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)