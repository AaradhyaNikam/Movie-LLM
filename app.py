import os
import re
import json
from datetime import datetime
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_123")

# Auto-fix PostgreSQL URL protocol for SQLAlchemy on Render
db_url = os.environ.get("DATABASE_URL", "sqlite:///app.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ==========================================
# 1. DATABASE MODELS
# ==========================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    searches = db.relationship('SearchHistory', backref='user', lazy=True)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.String(500), nullable=False)
    detected_mood = db.Column(db.String(100), nullable=True)  # Stores sentiment/mood
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.query(User).get(int(user_id))

# Auto-Run Database Schema Update
with app.app_context():
    db.create_all()
    try:
        db.session.execute(text('ALTER TABLE search_history ADD COLUMN IF NOT EXISTS detected_mood VARCHAR(100);'))
        db.session.commit()
        print("✅ Database schema updated successfully (or column already exists).")
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Schema update skipped or failed: {e}")

# ==========================================
# 2. AI VECTOR STORE & LLM INITIALIZATION
# ==========================================
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

# ==========================================
# 3. N-GRAM AUTOCOMPLETE ENGINE (DYNAMIC)
# ==========================================
def build_bigram_model_from_vectorstore(v_store):
    """Dynamically builds an N-gram model using the entire PDF database."""
    bigrams = defaultdict(Counter)
    
    if not v_store:
        print("⚠️ Vectorstore not found. Skipping N-gram generation.")
        return bigrams

    print("🧠 Extracting vocabulary from FAISS database...")
    
    # Access the raw documents stored inside the FAISS index
    for doc_id, doc in v_store.docstore._dict.items():
        text_content = doc.page_content
        
        # Extract only alphabetic words and convert to lowercase
        tokens = re.findall(r'[a-z]+', text_content.lower())
        
        # Build the Bigram pairs (Word 1 -> Word 2)
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            bigrams[w1][w2] += 1
            
    print(f"✅ N-gram model trained on {len(bigrams)} unique words!")
    return bigrams

# Train the model instantly when the server boots up
BIGRAM_MODEL = build_bigram_model_from_vectorstore(vectorstore)

def get_user_dominant_mood(user_id):
    """Calculates the user's most frequently searched movie mood."""
    history = db.session.query(SearchHistory).filter_by(user_id=user_id).all()
    moods = [h.detected_mood for h in history if h.detected_mood]
    if not moods:
        return "Action & Thriller"
    return Counter(moods).most_common(1)[0][0]

def parse_llm_json(raw_text):
    """Safely extracts JSON from LLM output."""
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return json.loads(clean_text.strip())

# ==========================================
# 4. APP ROUTES
# ==========================================
@app.route('/')
def index():
    recent_searches = []
    favorite_mood = None
    if current_user.is_authenticated:
        recent_searches = db.session.query(SearchHistory).filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).limit(5).all()
        favorite_mood = get_user_dominant_mood(current_user.id)
    return render_template('index.html', recent_searches=recent_searches, favorite_mood=favorite_mood)

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

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """N-Gram predictive text suggestions."""
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])
    
    words = query.split()
    last_word = words[-1]
    
    predictions = BIGRAM_MODEL.get(last_word, {}).most_common(3)
    suggested_words = [word for word, count in predictions]
    
    suggestions = [" ".join(words[:-1] + [last_word, next_w]).strip() for next_w in suggested_words]
    return jsonify(suggestions)

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

    try:
        results = vectorstore.similarity_search(user_query, k=5)
        context = "\n\n".join(doc.page_content for doc in results)

        prompt = f"""
        You are a movie and sentiment expert.
        QUESTION: Find details for "{user_query}".
        CONTEXT: {context}

        Return strictly a raw JSON object with no markdown surrounding text with these exact keys:
        1. "summary": A concise overview of the movie.
        2. "mood": A 1-to-3 word classification of the movie's mood/tone (e.g., "Dark & Suspenseful", "Feel-Good Comedy", "Mind-Bending Sci-Fi", "Action Thriller").

        JSON format:
        {{"summary": "...", "mood": "..."}}
        """
        response = llm.invoke(prompt)
        parsed_data = parse_llm_json(response.content)
        
        summary = parsed_data.get("summary", "No summary available.")
        detected_mood = parsed_data.get("mood", "Neutral")

        if current_user.is_authenticated:
            new_search = SearchHistory(
                user_id=current_user.id, 
                query=user_query,
                detected_mood=detected_mood
            )
            db.session.add(new_search)
            db.session.commit()

        return jsonify({
            "response": summary,
            "detected_mood": detected_mood
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations', methods=['GET'])
@login_required
def recommendations():
    """Netflix-style mood-based recommendation endpoint."""
    user_mood = get_user_dominant_mood(current_user.id)
    
    try:
        results = vectorstore.similarity_search(user_mood, k=5)
        context = "\n\n".join(doc.page_content for doc in results)

        prompt = f"""
        The user loves movies with the mood tone: "{user_mood}".
        Recommend 3 movies from the following context that match this mood profile:
        CONTEXT: {context}
        """
        response = llm.invoke(prompt)
        return jsonify({
            "favorite_mood": user_mood,
            "recommendations": response.content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)