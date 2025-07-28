import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sqlite3
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import altair as alt
import streamlit.components.v1 as components
from datetime import datetime
import time
import base64
import io

# For OCR & File Parsing
try:
    from PIL import Image
    import pytesseract
    import PyPDF2
    import docx
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# For ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# For TTS Voice Output (Automatic Playback)
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# -----------------------------------------------------------------------------
#   1) SET PAGE CONFIG AS THE FIRST STREAMLIT COMMAND
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Fraud Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
#   2) HELPER & UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def safe_rerun():
    """Refresh/re-run the Streamlit app if supported."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Page refresh not supported in this version of Streamlit.")

def play_audio(label):
    """
    Placeholder for playing an audio clip (e.g., a beep or chime).
    You can replace with something more sophisticated if needed.
    """
    pass

def play_voice_output(message):
    """
    Converts text to speech using gTTS, then automatically plays audio 
    using an HTML5 <audio> tag with autoplay.
    """
    if TTS_AVAILABLE:
        tts = gTTS(text=message, lang='en')
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)
        audio_bytes = open(audio_file, "rb").read()
        # Encode to base64 so we can autoplay in HTML
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay="true">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        pass

def slow_celebration():
    """Trigger Streamlit balloons for fun."""
    st.balloons()


# -----------------------------------------------------------------------------
#   3) SESSION STATE DEFAULTS & INITIALIZATION
# -----------------------------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "Unique"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_fraud_type" not in st.session_state:
    st.session_state.selected_fraud_type = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "can_change_settings" not in st.session_state:
    st.session_state.can_change_settings = (st.session_state.user_role == "admin")

# Default color settings
if "primary_color" not in st.session_state:
    st.session_state.primary_color = "#FF6F61"
if "secondary_color" not in st.session_state:
    st.session_state.secondary_color = "#6B5B95"
if "background_color" not in st.session_state:
    st.session_state.background_color = "#F7CAC9"
if "card_bg" not in st.session_state:
    st.session_state.card_bg = "#92A8D1"
if "text_color" not in st.session_state:
    st.session_state.text_color = "#034F84"
if "sidebar_bg" not in st.session_state:
    st.session_state.sidebar_bg = "#BFD8B8"
if "sidebar_text" not in st.session_state:
    st.session_state.sidebar_text = "#D64161"

if "font_family" not in st.session_state:
    st.session_state.font_family = "Roboto Mono"
if "font_size" not in st.session_state:
    st.session_state.font_size = "16px"

# Header customization
if "header_title" not in st.session_state:
    st.session_state.header_title = "Business Fraud Detection Platform"
if "header_subtitle" not in st.session_state:
    st.session_state.header_subtitle = "AI-Powered Analysis & Prevention of Fraudulent Activities"
if "header_bg_image" not in st.session_state:
    st.session_state.header_bg_image = ""  # Empty => use gradient background


# -----------------------------------------------------------------------------
#   4) THEME PRESETS (ADDITIONAL THEMES & COLORS)
# -----------------------------------------------------------------------------

preset_themes = {
    "Unique": {
        "primary_color": "#FF6F61",
        "secondary_color": "#6B5B95",
        "background_color": "#F7CAC9",
        "card_bg": "#92A8D1",
        "text_color": "#034F84",
        "sidebar_bg": "#BFD8B8",
        "sidebar_text": "#D64161"
    },
    "Blue": {
        "primary_color": "#1E90FF",
        "secondary_color": "#87CEFA",
        "background_color": "#D6EAF8",
        "card_bg": "#AED6F1",
        "text_color": "#00008B",
        "sidebar_bg": "#D0E9F5",
        "sidebar_text": "#1E90FF"
    },
    "Green": {
        "primary_color": "#28a745",
        "secondary_color": "#71dd8a",
        "background_color": "#e9f7ef",
        "card_bg": "#c7eccf",
        "text_color": "#155724",
        "sidebar_bg": "#d4edda",
        "sidebar_text": "#28a745"
    },
    "Red": {
        "primary_color": "#dc3545",
        "secondary_color": "#f5a6aa",
        "background_color": "#f8d7da",
        "card_bg": "#f1b0b7",
        "text_color": "#721c24",
        "sidebar_bg": "#f5c6cb",
        "sidebar_text": "#dc3545"
    },
    "Purple": {
        "primary_color": "#6f42c1",
        "secondary_color": "#bfa2e0",
        "background_color": "#e2d6f9",
        "card_bg": "#d1b3f0",
        "text_color": "#4a148c",
        "sidebar_bg": "#d6c8e3",
        "sidebar_text": "#6f42c1"
    },
    "Orange": {
        "primary_color": "#fd7e14",
        "secondary_color": "#ffbd69",
        "background_color": "#ffe5d0",
        "card_bg": "#ffd1a9",
        "text_color": "#c1440e",
        "sidebar_bg": "#ffe1cc",
        "sidebar_text": "#fd7e14"
    },
    "Teal": {
        "primary_color": "#20c997",
        "secondary_color": "#70e2cc",
        "background_color": "#d1f0ec",
        "card_bg": "#a0e8dc",
        "text_color": "#0b5345",
        "sidebar_bg": "#bde0ea",
        "sidebar_text": "#20c997"
    },
    "Pink": {
        "primary_color": "#e83e8c",
        "secondary_color": "#f4a6c9",
        "background_color": "#fce4ec",
        "card_bg": "#f8bbd0",
        "text_color": "#880e4f",
        "sidebar_bg": "#f8bbd0",
        "sidebar_text": "#e83e8c"
    },
    "Brown": {
        "primary_color": "#795548",
        "secondary_color": "#a98274",
        "background_color": "#d7ccc8",
        "card_bg": "#bcaaa4",
        "text_color": "#4e342e",
        "sidebar_bg": "#d7ccc8",
        "sidebar_text": "#795548"
    },
    # New/Extra example themes
    "Neon": {
        "primary_color": "#39FF14",
        "secondary_color": "#FF00FF",
        "background_color": "#000000",
        "card_bg": "#1a1a1a",
        "text_color": "#ffffff",
        "sidebar_bg": "#2c2c2c",
        "sidebar_text": "#39FF14"
    },
    "Cyberpunk": {
        "primary_color": "#ff0090",
        "secondary_color": "#00ffea",
        "background_color": "#080808",
        "card_bg": "#1f1f1f",
        "text_color": "#ffffff",
        "sidebar_bg": "#2e2e2e",
        "sidebar_text": "#ff0090"
    },
    "Vintage": {
        "primary_color": "#b76e79",
        "secondary_color": "#c8ad7f",
        "background_color": "#faf3e0",
        "card_bg": "#f2e8da",
        "text_color": "#5b4b3a",
        "sidebar_bg": "#e4d5c7",
        "sidebar_text": "#b76e79"
    },
    "Oceanic": {
        "primary_color": "#006994",
        "secondary_color": "#00a7c4",
        "background_color": "#E0F7FA",
        "card_bg": "#B2EBF2",
        "text_color": "#004d60",
        "sidebar_bg": "#80DEEA",
        "sidebar_text": "#006994"
    },
    "Material": {
        "primary_color": "#6200ea",
        "secondary_color": "#03dac6",
        "background_color": "#f5f5f5",
        "card_bg": "#ffffff",
        "text_color": "#000000",
        "sidebar_bg": "#eeeeee",
        "sidebar_text": "#6200ea"
    },
}

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Unique"

# Load chosen theme into session state
for key, value in preset_themes[st.session_state.theme_name].items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------------------------------------------------
#   5) DATABASE INITIALIZATION & FUNCTIONS
# -----------------------------------------------------------------------------
def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                name TEXT,
                company TEXT,
                role TEXT,
                approved INTEGER DEFAULT 0
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS fraud_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                fraud_type TEXT,
                prediction TEXT,
                probability REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        # Ensure there is an admin user
        c.execute("SELECT * FROM users WHERE username='admin'")
        if c.fetchone() is None:
            c.execute("INSERT INTO users (username, password, name, company, role, approved) VALUES (?, ?, ?, ?, ?, ?)",
                      ("admin", "admin", "Administrator", "YourCompany", "admin", 1))
            conn.commit()
    except Exception as e:
        st.error("DB Error: " + str(e))
    finally:
        conn.close()

init_db()


def create_user(username, password, name, company):
    """Create a new user in the database."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute(
            'INSERT INTO users (username, password, name, company, role, approved) VALUES (?, ?, ?, ?, ?, ?)',
            (username, password, name, company, "client", 0)
        )
        conn.commit()
        st.success("User account created successfully! Your account is pending admin approval.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

def authenticate_user(username, password):
    """Check username/password credentials."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute('SELECT password, role, approved FROM users WHERE username=?', (username,))
        result = c.fetchone()
        if result and result[0] == password:
            return {"username": username, "role": result[1], "approved": result[2]}
        else:
            return None
    except Exception as e:
        st.error("Authentication Error: " + str(e))
        return None
    finally:
        conn.close()

def log_history(username, fraud_type, prediction, probability):
    """Log a fraud detection result into the database."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute('INSERT INTO fraud_history (username, fraud_type, prediction, probability) VALUES (?, ?, ?, ?)',
                  (username, fraud_type, prediction, probability))
        conn.commit()
    except Exception as e:
        st.error("History Logging Error: " + str(e))
    finally:
        conn.close()

# Admin functions
def get_pending_users():
    """Get all users waiting for admin approval."""
    conn = sqlite3.connect('frauds.db')
    df = pd.read_sql_query("SELECT username, name, company FROM users WHERE approved=0 AND role='client'", conn)
    conn.close()
    return df

def approve_user(username):
    """Approve a pending user."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute("UPDATE users SET approved=1 WHERE username=?", (username,))
        conn.commit()
        st.success(f"User '{username}' approved.")
    except Exception as e:
        st.error("Error approving user: " + str(e))
    finally:
        conn.close()

def reject_user(username):
    """Reject (delete) a pending user."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
        st.success(f"User '{username}' rejected and removed.")
    except Exception as e:
        st.error("Error rejecting user: " + str(e))
    finally:
        conn.close()

def get_all_users():
    """Get a list of all users."""
    conn = sqlite3.connect('frauds.db')
    df = pd.read_sql_query("SELECT username, name, company, role FROM users", conn)
    conn.close()
    return df

def delete_user(username):
    """Delete a user from the database."""
    try:
        conn = sqlite3.connect('frauds.db')
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
        st.success(f"User '{username}' removed.")
    except Exception as e:
        st.error("Error deleting user: " + str(e))
    finally:
        conn.close()


# -----------------------------------------------------------------------------
#   6) SIMPLE FRAUD MODEL CLASS & PREDICTION LOGIC
# -----------------------------------------------------------------------------
class FraudModel:
    """
    A simple illustrative fraud model that checks for suspicious keywords
    and numeric anomalies in text/transaction data.
    """
    def __init__(self, fraud_type):
        self.fraud_type = fraud_type

    def predict_proba(self, df):
        suspicious_keywords = [
            "suspicious", "urgent", "immediately", "fraud", "scam", "error",
            "fake", "alert", "anomaly", "warning", "blocked", "compromise", 
            "re-confirm", "win", "big sale", "security check", "verify now", 
            "money", "bitcoin", "gift card"
        ]
        prob = 0.0
        text_fields = []
        # Check text-based columns
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, str):
                text_fields.append(val)
        if text_fields:
            combined_text = " ".join(text_fields).lower()
            count = sum(combined_text.count(word) for word in suspicious_keywords)
            base_prob = 0.3
            prob = base_prob + 0.1 * count

        # Check numeric anomalies
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, (int, float)) and val > 1000:
                prob += 0.1

        prob = min(prob, 1.0)
        return [[1 - prob, prob]]

@st.cache_resource
def get_fraud_model(fraud_type):
    """Return a simple instance of FraudModel for non-transaction fraud types."""
    return FraudModel(fraud_type)

def predict_fraud(model, df):
    """Predict Fraud using the simple FraudModel."""
    try:
        prob = model.predict_proba(df)[0][1]
        pred = "Fraud Detected" if prob > 0.5 else "No Fraud"

        # Extra logic: if text data is too large, slightly increase probability
        text_cols = [
            'content', 'post_content', 'ad_text', 'invoice_content',
            'claim_description', 'communication_content', 'review_text',
            'image_text', 'applicant_details'
        ]
        if any(col in df.columns for col in text_cols):
            text_data = " ".join([
                str(df[col].iloc[0]) 
                for col in df.columns 
                if col in text_cols and df[col].iloc[0] is not None
            ])
            if len(text_data) > 100:
                prob = min(prob + 0.1, 1.0)
                pred = "Fraud Detected" if prob > 0.5 else "No Fraud"
        return pred, prob
    except Exception as e:
        st.error("Prediction error: " + str(e))
        return "Unknown", 0.0

def prevention_info(fraud_type, pred, prob):
    """Return recommended prevention or action steps based on the result."""
    if pred == "No Fraud":
        return ("SAFE",
                "No fraudulent activity detected. Continue routine monitoring.",
                "Contact: support@yourcompany.com")
    else:
        if prob > 0.8:
            return ("CRITICAL",
                    "Immediate action required – suspend activity and investigate.",
                    "Hotline: 911-BIZ")
        elif prob > 0.6:
            return ("WARNING",
                    "Possible fraud detected – review transaction details and verify sender authenticity.",
                    "Helpdesk: 1800-BIZ-HELP")
        else:
            return ("CAUTION",
                    "Minor anomalies detected – monitor and flag if trends persist.",
                    "Support: support@yourcompany.com")


# -----------------------------------------------------------------------------
#   7) SYNTHETIC DATA & ADVANCED TRANSACTION MODELS
# -----------------------------------------------------------------------------
@st.cache_data
def load_synthetic_data_basic():
    np.random.seed(42)
    size = 500
    X1 = np.random.normal(loc=50, scale=10, size=size)
    X2 = np.random.normal(loc=100, scale=20, size=size)
    X3 = np.random.randint(0, 2, size=size)
    X4 = np.random.normal(loc=0, scale=1, size=size)
    y = []
    for i in range(size):
        score = 0
        if X1[i] > 55:
            score += 1
        if X2[i] > 120:
            score += 1
        if X3[i] == 1:
            score += 1
        if X4[i] > 0.5:
            score += 1
        y.append(1 if score >= 2 else 0)
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'FraudLabel': y
    })
    return df

df_synthetic_basic = load_synthetic_data_basic()
X_basic = df_synthetic_basic.drop('FraudLabel', axis=1)
y_basic = df_synthetic_basic['FraudLabel']

@st.cache_data
def train_basic_models(X, y):
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }
    accuracies = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[model_name] = acc
    return accuracies

model_accuracies_basic = train_basic_models(X_basic, y_basic)

@st.cache_data
def load_advanced_transaction_data(num_samples=5000, random_state=42):
    np.random.seed(random_state)
    df_size = num_samples
    amount = np.random.uniform(0, 10000, df_size)
    time_of_day = np.random.randint(0, 24, df_size)
    location_score = np.random.randint(0, 101, df_size)
    device_score = np.random.randint(0, 101, df_size)
    transaction_type = np.random.randint(0, 4, df_size)
    currency = np.random.randint(0, 4, df_size)
    user_auth_level = np.random.randint(0, 3, df_size)
    velocity = np.random.uniform(0.0, 5.0, df_size)
    suspicious_past_transactions = np.random.randint(0, 6, df_size)
    blacklisted_merchant = np.random.randint(0, 2, df_size)

    labels = []
    for i in range(df_size):
        suspicious_count = 0
        if blacklisted_merchant[i] == 1:
            suspicious_count += 1
        if suspicious_past_transactions[i] > 2:
            suspicious_count += 1
        if amount[i] > 8000:
            suspicious_count += 1
        if (transaction_type[i] == 2) and (user_auth_level[i] == 0):
            suspicious_count += 1
        if (location_score[i] < 20) and (device_score[i] < 20):
            suspicious_count += 1
        if velocity[i] > 2.5:
            suspicious_count += 1
        labels.append(1 if suspicious_count >= 2 else 0)

    df = pd.DataFrame({
        "amount": amount,
        "time_of_day": time_of_day,
        "location_score": location_score,
        "device_score": device_score,
        "transaction_type": transaction_type,
        "currency": currency,
        "user_auth_level": user_auth_level,
        "velocity": velocity,
        "suspicious_past_transactions": suspicious_past_transactions,
        "blacklisted_merchant": blacklisted_merchant,
        "FraudLabel": labels
    })
    return df

df_advanced_tx = load_advanced_transaction_data()

@st.cache_resource
def train_advanced_transaction_models(df):
    X = df.drop("FraudLabel", axis=1)
    y = df["FraudLabel"]
    classifiers = {
        "LogisticRegression": LogisticRegression(),
        "SVC": SVC(probability=True),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "GaussianNB": GaussianNB()
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    trained_models = {}
    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = clf
        accuracies[name] = acc
    return trained_models, accuracies

advanced_models, advanced_accuracies = train_advanced_transaction_models(df_advanced_tx)


# -----------------------------------------------------------------------------
#   8) ENHANCED CUSTOM CSS (EXTRA UI/UX ELEMENTS & ANIMATIONS)
# -----------------------------------------------------------------------------

font_family = st.session_state.get("font_family", "Roboto Mono")
font_size = st.session_state.get("font_size", "16px")
primary_color = st.session_state.get("primary_color", "#FF6F61")
secondary_color = st.session_state.get("secondary_color", "#6B5B95")
background_color = st.session_state.get("background_color", "#F7CAC9")
card_bg = st.session_state.get("card_bg", "#92A8D1")
text_color = st.session_state.get("text_color", "#034F84")
sidebar_bg = st.session_state.get("sidebar_bg", "#BFD8B8")
sidebar_text = st.session_state.get("sidebar_text", "#D64161")

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={font_family.replace(" ", "+")}&display=swap');

html, body {{
    margin: 0;
    padding: 0;
    background: {background_color};
    color: {text_color};
    font-family: '{font_family}', sans-serif;
    font-size: {font_size};
    scroll-behavior: smooth;
    overflow-x: hidden;
}}

body {{
    background: linear-gradient(135deg, {background_color} 0%, {card_bg} 100%);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}}
@keyframes gradientBG {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

.card {{
    background: {card_bg};
    border: 1px solid {primary_color};
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 40px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: {text_color};
    opacity: 0;
    animation: fadeInCard 0.8s forwards; 
}}
@keyframes fadeInCard {{
    to {{opacity: 1;}}
}}
.card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}}

.stButton>button {{
    background-color: {primary_color};
    color: {text_color};
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    transition: transform 0.2s ease, background 0.2s ease;
}}
.stButton>button:hover {{
    transform: scale(1.05);
    background: {secondary_color};
}}

input, .stTextInput>div>div>input {{
    background: {card_bg} !important;
    border: 1px solid {primary_color} !important;
    color: {text_color} !important;
    border-radius: 4px;
    padding: 8px;
    transition: box-shadow 0.2s ease;
}}
input:focus, .stTextInput>div>div>input:focus {{
    outline: none;
    box-shadow: 0 0 5px 2px {primary_color};
}}

.stDownloadButton button {{
    background-color: {secondary_color};
    color: {text_color};
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    transition: transform 0.2s ease, background 0.2s ease;
}}
.stDownloadButton button:hover {{
    transform: scale(1.03);
    background: {primary_color};
}}

.footer {{
    background: {background_color};
    border-top: 2px solid {primary_color};
    text-align: center;
    padding: 20px;
    margin: 40px 20px;
    border-radius: 10px;
}}
.footer p {{ margin: 0; font-size: 15px; color: {primary_color}; }}

.back-to-top {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: {primary_color};
    color: {text_color};
    border-radius: 50%;
    padding: 12px;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    z-index: 10000;
    transition: background 0.3s ease, transform 0.3s ease;
}}
.back-to-top:hover {{
    background: {secondary_color};
    transform: scale(1.1);
}}

.fraud-type-container .stButton {{
  margin: 10px;
}}
.fraud-type-container .stButton button {{
  width: 220px;
  height: 80px;
  font-size: 18px;
  border-radius: 12px;
  transition: background 0.3s ease, transform 0.3s ease, box-shadow 0.2s ease;
}}
.fraud-type-container .stButton button:hover {{
  background: {secondary_color};
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}}

.nav-container {{
  background-color: {sidebar_bg};
  border-radius: 12px;
  padding: 10px 20px;
  margin-bottom: 20px;
  animation: fadeInNav 0.5s forwards;
  opacity: 0;
}}
@keyframes fadeInNav {{
  to {{opacity: 1;}}
}}
.header-datetime {{
  text-align: right;
  font-size: 14px;
  color: {text_color};
}}

.element-container .stDataFrame tr:hover {{
    background-color: #ffffff22 !important; 
    filter: brightness(1.05);
    cursor: pointer;
}}

.css-1ht1j8u, .css-1ieh10n {{
    background: {card_bg} !important;
    border: 1px solid {primary_color} !important;
    border-radius: 10px;
}}
.css-1cpxqw2 p {{
    color: {secondary_color} !important;
    font-weight: 600 !important;
}}

.css-1cyszio, .stFileUploaderLabel {{
    background: {card_bg} !important;
    border: 2px dashed {primary_color} !important;
    color: {text_color} !important;
}}
.css-1cyszio:hover {{
    border: 2px dashed {secondary_color} !important;
}}

.stSlider > div > div > span > div[role='slider'] {{
    background-color: {primary_color} !important;
}}
.stSlider > div > div > span > div[role='slider']:hover {{
    background-color: {secondary_color} !important;
}}

.streamlit-expanderHeader {{
    font-weight: 600;
    background: {card_bg} !important;
    color: {primary_color} !important;
    border-left: 4px solid {primary_color};
    padding: 6px 10px;
    border-radius: 8px;
}}
.css-1l02zno.e1fqkh3o4 {{
    border-radius: 0 0 8px 8px !important;
}}

.stRadio > label > div, .stCheckbox > label > div {{
    color: {text_color} !important;
}}
.stRadio label:hover, .stCheckbox label:hover {{
    color: {secondary_color} !important;
}}

[data-tooltip] {{
    position: relative;
    cursor: pointer;
}}
[data-tooltip]:hover::after {{
    content: attr(data-tooltip);
    position: absolute;
    top: -35px;
    left: 0;
    padding: 6px 12px;
    background: {primary_color};
    color: {card_bg};
    border-radius: 4px;
    white-space: nowrap;
    font-size: 14px;
    z-index: 999;
}}

@keyframes fadeInCard {{
    0% {{opacity: 0; transform: translateY(20px);}}
    100% {{opacity: 1; transform: translateY(0);}}
}}
.fade-in {{
    opacity: 0;
    animation: fadeInContent 0.7s forwards;
}}
@keyframes fadeInContent {{
    to {{opacity: 1;}}
}}

/* Floating Help Button (example) */
.float-help {{
  position: fixed;
  bottom: 90px;
  right: 20px;
  background: {secondary_color};
  color: #fff;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  display: flex;
  justify-content: center;
  align-items: center;
  box-shadow: 0 4px 10px rgba(0,0,0,0.3);
  cursor: pointer;
  z-index: 10001;
  transition: transform 0.3s ease;
}}
.float-help:hover {{
  transform: translateY(-3px) scale(1.05);
}}
.help-tooltip {{
  display: none;
  position: absolute;
  bottom: 60px;
  right: 0;
  background: {card_bg};
  color: {text_color};
  padding: 10px;
  border-radius: 8px;
  border: 1px solid {primary_color};
  font-size: 14px;
  max-width: 220px;
}}
.float-help:hover .help-tooltip {{
  display: block;
}}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#   9) HEADER WITH DATE/TIME AT TOP
# -----------------------------------------------------------------------------
if st.session_state.header_bg_image.strip():
    header_style = f"""
        background-image: url('{st.session_state.header_bg_image}');
        background-size: cover; 
        padding: 50px 30px;
        border-radius: 15px;
        animation: fadeInContent 0.6s forwards;
    """
else:
    header_style = f"""
        background: linear-gradient(135deg, {primary_color}, {secondary_color}); 
        padding: 30px; 
        border-radius: 15px;
        animation: fadeInContent 0.6s forwards;
    """

st.markdown(
    f"""
    <div class="card" style="{header_style}">
        <div class="header-datetime">
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        <h2 style="text-align:center; color: {text_color}; margin-top: 10px;">
            {st.session_state.header_title}
        </h2>
        <p style="text-align:center; color: {text_color}; margin-bottom: 0;">
            {st.session_state.header_subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
#   10) NAVIGATION BAR (Option Menu)
# -----------------------------------------------------------------------------
options = [
    "Home",
    "User Onboarding",
    "Fraud Detection",
    "Chatbot",
    "Dashboard",
    "Fraud Analytics",
    "Fraud Prevention",
    "Advanced Transaction Fraud",
    "Document & Media Fraud Detection",
    "FAQ",
    "About",
    "Contact Us",
    "Settings"
]
if st.session_state.logged_in and st.session_state.user_role == "admin":
    options.insert(1, "Admin Panel")

with st.container():
    st.markdown("<div class='card nav-container fade-in'>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="",
        options=options,
        icons=[
            "house",
            "person-lines-fill",
            "person-plus",
            "exclamation-triangle",
            "chat-dots",
            "bar-chart-line",
            "graph-up",
            "shield-check",
            "credit-card",
            "file-earmark-text",
            "question-circle",
            "info-circle",
            "envelope",
            "gear"
        ],
        menu_icon="list-task",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "10px", "background-color": sidebar_bg},
            "icon": {"color": primary_color, "font-size": "22px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": secondary_color,
                "color": sidebar_text
            },
            "nav-link-selected": {
                "background-color": primary_color,
                "color": text_color
            }
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#   11) PAGE ROUTING
# -----------------------------------------------------------------------------

if selected == "Home":
    st.markdown("<div class='card fade-in'><h2>Welcome!</h2><p>This platform uses advanced AI to detect and prevent fraudulent activities across various business channels.</p></div>", unsafe_allow_html=True)
    st.image("https://source.unsplash.com/1600x900/?business,security", use_container_width=True)
    st.markdown("<div class='card fade-in'><h3>Daily Fraud Tip</h3><p>Always verify email senders and cross-check suspicious transactions with internal records.</p></div>", unsafe_allow_html=True)

elif selected == "User Onboarding":
    if not st.session_state.logged_in:
        mode = st.radio("Choose Option", ["Login", "Sign Up"])
        if mode == "Sign Up":
            st.markdown("<div class='card fade-in'><h2>Create Account</h2></div>", unsafe_allow_html=True)
            name = st.text_input("Full Name", help="Enter your full name.")
            username = st.text_input("Username", help="Choose a unique username.")
            password = st.text_input("Password", type="password", help="Enter a secure password.")
            company = st.text_input("Company Name", help="Enter your company name.")
            if st.button("Create Account"):
                if name and username and password and company:
                    create_user(username, password, name, company)
                else:
                    st.warning("Please fill in all fields.")
        else:
            st.markdown("<div class='card fade-in'><h2>User Login</h2></div>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user_info = authenticate_user(username, password)
                if user_info:
                    if user_info["approved"] == 0:
                        st.error("Your account is pending admin approval. Please wait for approval before logging in.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = user_info["role"]
                        st.success("Logged in successfully!")
                        play_voice_output(f"Hello {username}, you are now logged in.")
                else:
                    st.error("Invalid credentials!", icon="🚨")
    else:
        st.markdown(f"<div class='card fade-in'><h2>Welcome, {st.session_state.username}!</h2></div>", unsafe_allow_html=True)
        st.write("Your account details are on file.")
        if st.session_state.user_role == "admin":
            st.info("You are logged in as an Admin.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_role = None
            safe_rerun()

elif selected == "Admin Panel":
    if st.session_state.logged_in and st.session_state.user_role == "admin":
        st.markdown("<div class='card fade-in'><h2>Admin Panel</h2><p>Manage user registrations and accounts.</p></div>", unsafe_allow_html=True)
        st.subheader("Pending User Registrations")
        pending_df = get_pending_users()
        if pending_df.empty:
            st.info("No pending registrations.")
        else:
            st.dataframe(pending_df)
            for index, row in pending_df.iterrows():
                col1, col2, col3 = st.columns([2,2,2])
                with col1:
                    st.write(row["username"])
                with col2:
                    if st.button(f"Approve {row['username']}", key=f"approve_{row['username']}"):
                        approve_user(row["username"])
                        safe_rerun()
                with col3:
                    if st.button(f"Reject {row['username']}", key=f"reject_{row['username']}"):
                        reject_user(row["username"])
                        safe_rerun()

        st.subheader("All Users")
        all_users_df = get_all_users()
        st.dataframe(all_users_df)
        st.write("To remove a user, click the corresponding remove button below:")
        for index, row in all_users_df.iterrows():
            if row["username"] != "admin":
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"{row['username']} ({row['role']})")
                with col2:
                    if st.button(f"Remove {row['username']}", key=f"remove_{row['username']}"):
                        delete_user(row["username"])
                        safe_rerun()
    else:
        st.error("Access denied. Admins only.")

elif selected == "Fraud Detection":
    if not st.session_state.logged_in:
        st.error("Please log in to access Fraud Detection.")
    else:
        st.markdown("""<div class='card fade-in'>
            <h2>Fraud Detection</h2>
            <p>Analyze your data to detect potential fraud. A comparison analysis shows how your current result compares to your past history.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card fade-in'>
            <h3>Select a Fraud Type</h3>
            <p>Choose from the categories below to run an AI-driven fraud analysis on your data.</p>
        </div>""", unsafe_allow_html=True)

        fraud_types = [
            "Email Fraud", "Social Media Fraud", "Web Ad Fraud",
            "Transaction Fraud", "Access Log Anomaly", "Invoice Fraud",
            "Insurance Fraud", "Loan Application Fraud", "Employee Fraud",
            "Customer Review Fraud", "Document Fraud"
        ]
        st.markdown('<div class="fraud-type-container">', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, fraud in enumerate(fraud_types):
            col = cols[idx % 3]
            if col.button(fraud):
                st.session_state.selected_fraud_type = fraud
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.selected_fraud_type is None:
            st.info("Please select a fraud type from the options above.")
        else:
            if st.button("Reset Fraud Type Selection"):
                st.session_state.selected_fraud_type = None
                safe_rerun()

            st.markdown(f"<div class='card fade-in'><h3>Selected Fraud Type: {st.session_state.selected_fraud_type}</h3></div>", unsafe_allow_html=True)
            fraud_type = st.session_state.selected_fraud_type
            input_df = None
            model = None

            with st.expander("Step-by-Step Instructions"):
                st.markdown("""
                1. Enter all relevant details below.
                2. Press **Run Fraud Analysis** to generate a prediction.
                3. A result card will show the prediction, probability, and recommended actions.
                4. A comparison analysis is provided based on your historical results.
                5. Optionally, download or email the report.
                """)

            # Different forms by fraud type
            if fraud_type == "Email Fraud":
                st.subheader("Email Fraud Analysis")
                subject = st.text_input("Email Subject")
                sender = st.text_input("Sender Email")
                content = st.text_area("Email Content")
                features = {"subject": subject, "sender": sender, "content": content}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Social Media Fraud":
                st.subheader("Social Media Fraud Analysis")
                post_content = st.text_area("Post Content")
                profile_url = st.text_input("Profile URL")
                features = {"post_content": post_content, "profile_url": profile_url}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Web Ad Fraud":
                st.subheader("Web Advertisement Fraud Analysis")
                ad_text = st.text_area("Advertisement Text")
                landing_page = st.text_input("Landing Page URL")
                features = {"ad_text": ad_text, "landing_page": landing_page}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Transaction Fraud":
                st.subheader("Advanced Transaction Fraud Analysis")
                model_choice = st.selectbox("Select a Model", options=list(advanced_models.keys()))
                st.write(f"Selected Model Accuracy: **{advanced_accuracies[model_choice]*100:.2f}%**")
                amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
                time_of_day = st.slider("Time of Day (0-23)", 0, 23, 12)
                location_score = st.slider("Location Score (0-100)", 0, 100, 50)
                device_score = st.slider("Device Score (0-100)", 0, 100, 50)
                transaction_type = st.selectbox("Transaction Type", ["Purchase (0)", "Refund (1)", "Transfer (2)", "Payment (3)"])
                transaction_type_val = int(transaction_type.split("(")[-1].replace(")", ""))
                currency = st.selectbox("Currency", ["USD (0)", "EUR (1)", "INR (2)", "GBP (3)"])
                currency_val = int(currency.split("(")[-1].replace(")", ""))
                user_auth_level = st.selectbox("User Auth Level", ["basic (0)", "2FA (1)", "biometric (2)"])
                user_auth_val = int(user_auth_level.split("(")[-1].replace(")", ""))
                velocity = st.slider("Transaction Velocity (0.0 - 5.0)", 0.0, 5.0, 1.0)
                suspicious_past_transactions = st.slider("Suspicious Past Transactions (0 - 5)", 0, 5, 0)
                blacklisted_merchant = st.selectbox("Blacklisted Merchant?", ["No (0)", "Yes (1)"])
                blacklisted_val = int(blacklisted_merchant.split("(")[-1].replace(")", ""))

                input_df = pd.DataFrame([{
                    "amount": amount,
                    "time_of_day": time_of_day,
                    "location_score": location_score,
                    "device_score": device_score,
                    "transaction_type": transaction_type_val,
                    "currency": currency_val,
                    "user_auth_level": user_auth_val,
                    "velocity": velocity,
                    "suspicious_past_transactions": suspicious_past_transactions,
                    "blacklisted_merchant": blacklisted_val
                }])
                model = advanced_models[model_choice]

            elif fraud_type == "Access Log Anomaly":
                st.subheader("Access Log Fraud Analysis")
                ip_address = st.text_input("IP Address")
                device_info = st.text_input("Device Information")
                timestamp = st.text_input("Access Timestamp (YYYY-MM-DD HH:MM:SS)")
                features = {"ip_address": ip_address, "device_info": device_info, "timestamp": timestamp}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Invoice Fraud":
                st.subheader("Invoice Fraud Analysis")
                invoice_content = st.text_area("Invoice Content")
                vendor_email = st.text_input("Vendor Email")
                invoice_amount = st.number_input("Invoice Amount", min_value=0.0, value=0.0)
                features = {"invoice_content": invoice_content, "vendor_email": vendor_email, "invoice_amount": invoice_amount}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Insurance Fraud":
                st.subheader("Insurance Fraud Analysis")
                claim_description = st.text_area("Claim Description")
                policy_number = st.text_input("Policy Number")
                claim_amount = st.number_input("Claim Amount", min_value=0.0, value=0.0)
                features = {"claim_description": claim_description, "policy_number": policy_number, "claim_amount": claim_amount}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Loan Application Fraud":
                st.subheader("Loan Application Fraud Analysis")
                applicant_details = st.text_area("Applicant Details")
                credit_score = st.number_input("Credit Score", min_value=0, value=600)
                requested_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=1000.0)
                features = {"applicant_details": applicant_details, "credit_score": credit_score, "requested_amount": requested_amount}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Employee Fraud":
                st.subheader("Employee Fraud Analysis")
                employee_email = st.text_input("Employee Email")
                communication_content = st.text_area("Communication Content")
                features = {"employee_email": employee_email, "communication_content": communication_content}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Customer Review Fraud":
                st.subheader("Customer Review Fraud Analysis")
                review_text = st.text_area("Review Text")
                reviewer_name = st.text_input("Reviewer Name")
                features = {"review_text": review_text, "reviewer_name": reviewer_name}
                input_df = pd.DataFrame([features])
                model = get_fraud_model(fraud_type)

            elif fraud_type == "Document Fraud":
                st.subheader("Document Fraud Analysis")
                st.info("For document analysis, please use the Document & Media Fraud Detection page.")

            if st.button("Run Fraud Analysis"):
                with st.spinner("Analyzing data..."):
                    if model is None:
                        st.error("Model not loaded. Please check the model file.")
                    else:
                        if fraud_type == "Transaction Fraud":
                            pred_numeric = model.predict(input_df)[0]
                            prob = model.predict_proba(input_df)[0][1]
                            label = "Fraud Detected" if pred_numeric == 1 else "No Fraud"
                        else:
                            label, prob = predict_fraud(model, input_df)

                        # Play short audio alert & TTS
                        play_audio(label)  
                        play_voice_output(f"Result: {label}. Probability: {prob:.2f}")

                        st.markdown(f"<div class='card fade-in'><h3>Result: {label}</h3><p>Fraud Probability: {prob:.2f}</p></div>", unsafe_allow_html=True)
                        alert, advice, support = prevention_info(fraud_type, label, prob)
                        st.markdown(f"<div class='card fade-in'><h3>Alert Level: {alert}</h3><p>{advice}</p><p><strong>Contact:</strong> {support}</p></div>", unsafe_allow_html=True)

                        # Log in DB
                        log_history(st.session_state.username, fraud_type, label, prob)

                        # Compare with historical average
                        try:
                            conn = sqlite3.connect('frauds.db')
                            c = conn.cursor()
                            c.execute("SELECT AVG(probability) FROM fraud_history WHERE username=? AND fraud_type=?",
                                      (st.session_state.username, fraud_type))
                            avg_prob = c.fetchone()[0]
                            conn.close()
                            if avg_prob is not None:
                                comp = "above" if prob > avg_prob else "below"
                                st.markdown(f"<div class='card fade-in'><p>Your current fraud probability of {prob:.2f} is {comp} your historical average of {avg_prob:.2f}.</p></div>", unsafe_allow_html=True)
                            else:
                                st.info("No historical data to compare.")
                        except Exception as e:
                            st.error("Error in comparison analysis: " + str(e))

                        # Download & Email
                        report = (
                            f"Fraud Type: {fraud_type}\n"
                            f"Result: {label}\n"
                            f"Probability: {prob:.2f}\n"
                            f"Alert: {alert}\n"
                            f"Advice: {advice}\n"
                            f"Contact: {support}"
                        )
                        st.download_button("Download Report", report, file_name="fraud_report.txt", mime="text/plain")
                        if st.button("Send Report via Email"):
                            recipient = st.text_input("Enter Email Address:")
                            if recipient:
                                st.info(f"Simulated email sent to {recipient}:\n\n{report}")
                            else:
                                st.warning("Please enter a valid email address.")
                        slow_celebration()

elif selected == "Chatbot":
    if not st.session_state.logged_in:
        st.error("Please log in to access the Fraud Advisor Chatbot.")
    else:
        st.markdown("<div class='card fade-in'><h2>Fraud Advisor Chatbot</h2><p>Ask any questions about fraud detection and prevention. The chatbot is now enhanced with more options.</p></div>", unsafe_allow_html=True)
        user_name = st.session_state.get("username", "Guest")

        def get_chatbot_html(user_name):
            extended_responses_js = ""
            # Generate a big dummy set of responses for demonstration
            for i in range(1, 21):
                extended_responses_js += f'"topic{i}": "Automated response for topic {i}.",\n'

            base_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Fraud Advisor Chatbot</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {{
      --primary-color: {primary_color};
      --secondary-color: {secondary_color};
      --bg-color: {background_color};
      --text-color: {text_color};
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: '{font_family}', monospace;
      background: var(--bg-color);
      color: var(--text-color);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .chat-container {{
      width: 100%;
      max-width: 700px;
      background: {card_bg};
      border-radius: 15px;
      border: 1px solid var(--primary-color);
      box-shadow: 0 10px 30px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .chat-header {{
      background: var(--primary-color);
      padding: 20px;
      text-align: center;
      color: var(--text-color);
      cursor: move;
    }}
    .chat-header h2 {{
      font-size: 2em;
      margin-bottom: 5px;
    }}
    .chat-box {{
      flex: 1;
      padding: 20px;
      background: #E6F0FA;
      overflow-y: auto;
      max-height: 400px;
    }}
    .message {{
      display: flex;
      align-items: flex-start;
      margin-bottom: 15px;
    }}
    .message .icon {{
      font-size: 24px;
      margin-right: 10px;
      color: var(--primary-color);
    }}
    .user-message, .bot-message {{
      max-width: 75%;
      padding: 12px 16px;
      border-radius: 20px;
      word-wrap: break-word;
    }}
    .user-message {{
      background: var(--secondary-color);
      color: var(--bg-color);
      margin-left: auto;
      border-bottom-right-radius: 0;
      text-align: right;
    }}
    .bot-message {{
      background: var(--primary-color);
      color: var(--bg-color);
      margin-right: auto;
      border-bottom-left-radius: 0;
      text-align: left;
    }}
    .timestamp {{
      font-size: 10px;
      color: #fdd;
      margin-top: 5px;
    }}
    .input-section {{
      background: #E6F0FA;
      padding: 15px;
      border-top: 1px solid var(--primary-color);
    }}
    .input-container {{
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .input-container input {{
      flex: 1;
      padding: 12px;
      border-radius: 8px;
      border: 2px solid var(--primary-color);
      font-size: 16px;
      background: {card_bg};
      color: var(--text-color);
    }}
    .input-container button {{
      padding: 12px 20px;
      border: none;
      background: var(--primary-color);
      color: var(--text-color);
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      transition: transform 0.3s ease;
    }}
    .input-container button:hover {{
      transform: scale(1.05);
    }}
    .quick-replies {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: center;
      margin-bottom: 10px;
    }}
    .quick-replies button {{
      padding: 8px 12px;
      border: none;
      background: var(--primary-color);
      color: var(--bg-color);
      border-radius: 20px;
      cursor: pointer;
      font-size: 14px;
      transition: transform 0.2s ease;
    }}
    .quick-replies button:hover {{
      transform: scale(1.05);
    }}
    .clear-btn {{
      background: var(--secondary-color);
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 8px;
      font-size: 16px;
      cursor: pointer;
      font-weight: 600;
    }}
    .clear-btn:hover {{
      background: var(--primary-color);
    }}
  </style>
</head>
<body>
  <div class="chat-container" id="chat-container">
    <div class="chat-header" id="chat-header">
      <h2><i class="fa-solid fa-robot icon"></i> Welcome, {user_name}!</h2>
      <p>Your Fraud Advisor is here to assist.</p>
    </div>
    <div class="chat-box" id="chat-box">
      <div class="message bot-message">
        <i class="fa-solid fa-robot icon"></i>
        <div id="bot-initial-message"></div>
      </div>
    </div>
    <div class="input-section">
      <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your query..." onkeypress="handleEnter(event)">
        <button onclick="sendMessage()"><i class="fa-solid fa-paper-plane"></i></button>
      </div>
      <div class="quick-replies">
        <button onclick="sendQuickReply('Email Verification')"><i class="fa-solid fa-envelope"></i> Email</button>
        <button onclick="sendQuickReply('Transaction Check')"><i class="fa-solid fa-money-bill-wave"></i> Transaction</button>
        <button onclick="sendQuickReply('Social Media Alert')"><i class="fa-solid fa-hashtag"></i> Social</button>
        <button onclick="sendQuickReply('Access Log')"><i class="fa-solid fa-laptop"></i> Log</button>
        <button onclick="sendQuickReply('Invoice Inquiry')"><i class="fa-solid fa-file-invoice"></i> Invoice</button>
        <button onclick="sendQuickReply('Loan Query')"><i class="fa-solid fa-hand-holding-dollar"></i> Loan</button>
        <button onclick="sendQuickReply('Employee Check')"><i class="fa-solid fa-user-tie"></i> Employee</button>
        <button onclick="sendQuickReply('Review Analysis')"><i class="fa-solid fa-comments"></i> Review</button>
        <button onclick="sendQuickReply('FAQ')"><i class="fa-solid fa-question"></i> FAQ</button>
      </div>
      <button class="clear-btn" onclick="clearChat()"><i class="fa-solid fa-trash"></i> Clear Chat</button>
    </div>
  </div>

  <script>
    const initialMessage = "Hello {user_name}, how can I help you with fraud detection today?";
    const typingSpeed = 40;
    let charIndex = 0;
    function typeWriter() {{
      const container = document.getElementById("bot-initial-message");
      if(!container) return;
      if(charIndex < initialMessage.length) {{
        container.innerHTML += initialMessage.charAt(charIndex);
        charIndex++;
        setTimeout(typeWriter, typingSpeed);
      }} else {{
        const timeSpan = document.createElement("div");
        timeSpan.className = "timestamp";
        timeSpan.innerText = getCurrentTime();
        container.appendChild(timeSpan);
      }}
    }}
    window.addEventListener('load', function() {{
      typeWriter();
    }});
    function getCurrentTime() {{
      const now = new Date();
      return now.toLocaleTimeString([], {{hour: '2-digit', minute:'2-digit'}});
    }}
    function sendMessage() {{
      const inputField = document.getElementById("user-input");
      const message = inputField.value.trim();
      if(!message) return;
      const chatBox = document.getElementById("chat-box");
      const userMsgContainer = document.createElement("div");
      userMsgContainer.className = "message user-message";
      userMsgContainer.innerHTML = `
        <i class="fa-solid fa-user icon"></i>
        <div>
          ${{message}}
          <div class="timestamp">${{getCurrentTime()}}</div>
        </div>
      `;
      chatBox.appendChild(userMsgContainer);
      inputField.value = "";
      chatBox.scrollTop = chatBox.scrollHeight;
      handleBotResponse(message);
    }}
    function sendQuickReply(reply) {{
      document.getElementById("user-input").value = reply;
      sendMessage();
    }}
    function handleEnter(event) {{ 
      if(event.key === "Enter") {{
        sendMessage(); 
      }} 
    }}
    function handleBotResponse(message) {{
      const baseResponses = {{
        "email": "Verify that the sender's email matches known domains.",
        "transaction": "Review the transaction details for anomalies.",
        "social": "Double-check social media profiles for authenticity.",
        "log": "Analyze access logs for unusual activity.",
        "invoice": "Ensure invoice details match company records.",
        "loan": "Examine loan application details for discrepancies.",
        "employee": "Review employee communication for red flags.",
        "review": "Evaluate review content for biased or fake entries.",
        "faq": "You can check our FAQ section for common fraud queries.",
        "hello": "Hello! How can I assist with fraud detection?",
        "hi": "Hi there! Ask me about fraud verification tips."
      }};
      const extendedResponses = {{
        {extended_responses_js}
      }};
      let responses = {{...baseResponses, ...extendedResponses}};

      let response = "Could you clarify that? I'm here to help with fraud detection.";
      const msgLower = message.toLowerCase();
      Object.keys(responses).forEach((key) => {{
        if(msgLower.includes(key)) {{
          response = responses[key];
        }}
      }});
      showBotResponse(response);
    }}
    function showBotResponse(response) {{
      const chatBox = document.getElementById("chat-box");
      const typingDiv = document.createElement("div");
      typingDiv.className = "message bot-message";
      typingDiv.innerHTML = `
        <i class="fa-solid fa-robot icon"></i>
        <div>Typing...
          <div class="timestamp">${{getCurrentTime()}}</div>
        </div>
      `;
      chatBox.appendChild(typingDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
      setTimeout(() => {{
        chatBox.removeChild(typingDiv);
        const botMsgContainer = document.createElement("div");
        botMsgContainer.className = "message bot-message";
        botMsgContainer.innerHTML = `
          <i class="fa-solid fa-robot icon"></i>
          <div>
            ${{response}}
            <div class="timestamp">${{getCurrentTime()}}</div>
          </div>
        `;
        chatBox.appendChild(botMsgContainer);
        chatBox.scrollTop = chatBox.scrollHeight;
      }}, 1200);
    }}
    function clearChat() {{
      const chatBox = document.getElementById("chat-box");
      chatBox.innerHTML = "";
      const clearedMsg = document.createElement("div");
      clearedMsg.className = "message bot-message";
      clearedMsg.innerHTML = `
        <i class="fa-solid fa-robot icon"></i>
        <div>
          Chat cleared. How else can I help you today?
          <div class="timestamp">${{getCurrentTime()}}</div>
        </div>
      `;
      chatBox.appendChild(clearedMsg);
    }}
  </script>
</body>
</html>
"""
            return base_html

        chatbot_html = get_chatbot_html(user_name)
        components.html(chatbot_html, height=700)

elif selected == "Dashboard":
    if not st.session_state.logged_in:
        st.error("Please log in to view your dashboard.")
    else:
        st.markdown("<div class='card fade-in'><h2>Fraud Alert Dashboard</h2><p>Review your past fraud analysis, trends, and key metrics.</p></div>", unsafe_allow_html=True)
        try:
            conn = sqlite3.connect('frauds.db')
            df_history = pd.read_sql_query("SELECT fraud_type, prediction, probability, timestamp FROM fraud_history WHERE username=?", conn, params=(st.session_state.username,))
            conn.close()

            if df_history.empty:
                st.info("No history available.")
            else:
                total_alerts = len(df_history)
                avg_probability = df_history['probability'].mean()
                unique_fraud_types = df_history['fraud_type'].nunique()

                st.markdown("<div class='card fade-in'><h3>Summary Metrics</h3></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Fraud Alerts", total_alerts)
                col2.metric("Avg. Fraud Probability", f"{avg_probability:.2f}")
                col3.metric("Fraud Types Detected", unique_fraud_types)

                analysis_options = ["All Fraud History", "Document & Media Fraud", "Advanced Transaction Fraud"]
                selected_analysis = st.selectbox("Select Analysis View", analysis_options)
                if selected_analysis == "All Fraud History":
                    filtered_df = df_history.copy()
                elif selected_analysis == "Document & Media Fraud":
                    filtered_df = df_history[df_history["fraud_type"]=="Document Fraud"]
                elif selected_analysis == "Advanced Transaction Fraud":
                    filtered_df = df_history[df_history["fraud_type"]=="Transaction Fraud"]

                st.dataframe(filtered_df)
                filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

                st.markdown("### Fraud Probability Trend")
                chart = alt.Chart(filtered_df).mark_line(point=True).encode(
                    x='timestamp:T',
                    y='probability:Q',
                    color='fraud_type:N',
                    tooltip=['timestamp:T', 'probability:Q', 'fraud_type:N']
                ).properties(width=700, height=400)
                st.altair_chart(chart, use_container_width=True)

                st.markdown("### Fraud Type Breakdown")
                breakdown = filtered_df.groupby('fraud_type')['prediction'].count().reset_index()
                fig, ax = plt.subplots()
                ax.pie(breakdown['prediction'], labels=breakdown['fraud_type'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                st.markdown("### Average Fraud Probability by Type")
                avg_df = filtered_df.groupby('fraud_type')['probability'].mean().reset_index()
                chart2 = alt.Chart(avg_df).mark_bar().encode(
                    x=alt.X("fraud_type:N", title="Fraud Type"),
                    y=alt.Y("probability:Q", title="Average Probability"),
                    color="fraud_type:N"
                ).properties(width=700, height=400)
                st.altair_chart(chart2, use_container_width=True)

        except Exception as e:
            st.error("Error loading dashboard: " + str(e))

elif selected == "Fraud Analytics":
    if not st.session_state.logged_in:
        st.error("Please log in to view Fraud Analytics.")
    else:
        st.markdown("<div class='card fade-in'><h2>Model Performance & Trends</h2><p>Review the evolution of our fraud detection models on synthetic data.</p></div>", unsafe_allow_html=True)
        st.markdown("""<div class='card fade-in'>
            <h3>Advanced Analytics</h3>
            <p>Besides basic accuracy, we track additional metrics. Future updates will include cross-validation and more.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("### Multiple Algorithms on Basic Synthetic Fraud Data")
        df_acc_basic = pd.DataFrame({
            'Model': list(model_accuracies_basic.keys()),
            'Accuracy': list(model_accuracies_basic.values())
        })
        st.dataframe(df_acc_basic)

        st.markdown("### Model Accuracy Comparison (Basic Dataset)")
        chart_basic = alt.Chart(df_acc_basic).mark_bar().encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Accuracy:Q", title="Accuracy"),
            color=alt.Color("Model:N", legend=None)
        ).properties(width=700, height=400)
        st.altair_chart(chart_basic, use_container_width=True)

        st.markdown("### Advanced Transaction Fraud Model Comparison")
        df_adv_acc = pd.DataFrame({
            "Model": list(advanced_accuracies.keys()),
            "Accuracy": list(advanced_accuracies.values())
        }).sort_values(by="Accuracy", ascending=False)
        st.dataframe(df_adv_acc.reset_index(drop=True))

        chart_adv = alt.Chart(df_adv_acc).mark_bar().encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Accuracy:Q", title="Accuracy"),
            color="Model:N"
        ).properties(width=700, height=400)
        st.altair_chart(chart_adv, use_container_width=True)

elif selected == "Fraud Prevention":
    if not st.session_state.logged_in:
        st.error("Please log in to access Fraud Prevention guidelines.")
    else:
        st.markdown("<div class='card fade-in'><h2>Fraud Prevention Guidelines</h2></div>", unsafe_allow_html=True)
        st.markdown("""
            **Common Fraud Schemes:**
            - **Email Phishing:** Verify sender addresses and use spam filters.
            - **Fake Invoices:** Cross-check invoice details with known records.
            - **Transaction Fraud:** Monitor for unusual payment amounts.
            - **Social Media Impersonation:** Confirm business profiles.
            - **Credential Stuffing:** Use strong passwords and multi-factor authentication.
            - **Insider Threats:** Restrict unnecessary access.
            - **Malware/Ransomware:** Keep software updated.
            
            **Red Flags:**
            - Urgent requests for wire transfers.
            - Slightly off email domains.
            - Logins from unusual IPs.
            - Requests to bypass verification.
            - Sudden vendor detail changes.
            
            **Preventive Measures:**
            - **MFA:** Use OTPs or biometrics.
            - **Anti-Malware:** Regular endpoint scans.
            - **Employee Education:** Regular training on phishing.
            - **Network Segmentation:** Limit lateral movement.
            - **Regular Audits:** Monitor for anomalies.
        """)
        st.image("https://source.unsplash.com/800x600/?fraud,security", use_container_width=True)

elif selected == "Advanced Transaction Fraud":
    if not st.session_state.logged_in:
        st.error("Please log in to use Advanced Transaction Fraud features.")
    else:
        st.markdown("<div class='card fade-in'><h2>Advanced Transaction Fraud</h2><p>Predict potential fraud using our advanced ML models.</p></div>", unsafe_allow_html=True)
        st.markdown("### Model Accuracies on Advanced Transaction Dataset")
        df_adv_acc = pd.DataFrame({
            "Model": list(advanced_accuracies.keys()),
            "Accuracy": list(advanced_accuracies.values())
        }).sort_values(by="Accuracy", ascending=False)
        st.dataframe(df_adv_acc.reset_index(drop=True))

        model_choice = st.selectbox("Select a Model", options=list(advanced_models.keys()))
        chosen_model = advanced_models[model_choice]
        st.write(f"Selected Model Accuracy: **{advanced_accuracies[model_choice]*100:.2f}%**")

        with st.form("advanced_tx_form"):
            amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0)
            time_of_day = st.slider("Time of Day (0-23)", 0, 23, 12)
            location_score = st.slider("Location Score (0-100)", 0, 100, 50)
            device_score = st.slider("Device Score (0-100)", 0, 100, 50)
            transaction_type = st.selectbox("Transaction Type", ["Purchase (0)", "Refund (1)", "Transfer (2)", "Payment (3)"])
            transaction_type_val = int(transaction_type.split("(")[-1].replace(")", ""))
            currency = st.selectbox("Currency", ["USD (0)", "EUR (1)", "INR (2)", "GBP (3)"])
            currency_val = int(currency.split("(")[-1].replace(")", ""))
            user_auth_level = st.selectbox("User Auth Level", ["basic (0)", "2FA (1)", "biometric (2)"])
            user_auth_val = int(user_auth_level.split("(")[-1].replace(")", ""))
            velocity = st.slider("Transaction Velocity (0.0 - 5.0)", 0.0, 5.0, 1.0)
            suspicious_past_transactions = st.slider("Suspicious Past Transactions (0 - 5)", 0, 5, 0)
            blacklisted_merchant = st.selectbox("Blacklisted Merchant?", ["No (0)", "Yes (1)"])
            blacklisted_val = int(blacklisted_merchant.split("(")[-1].replace(")", ""))

            submitted = st.form_submit_button("Predict Fraud")

        if submitted:
            input_data = pd.DataFrame([{
                "amount": amount,
                "time_of_day": time_of_day,
                "location_score": location_score,
                "device_score": device_score,
                "transaction_type": transaction_type_val,
                "currency": currency_val,
                "user_auth_level": user_auth_val,
                "velocity": velocity,
                "suspicious_past_transactions": suspicious_past_transactions,
                "blacklisted_merchant": blacklisted_val
            }])
            pred = chosen_model.predict(input_data)[0]
            prob = chosen_model.predict_proba(input_data)[0][1]
            label = "Fraud Detected" if pred == 1 else "No Fraud"

            play_audio(label)
            play_voice_output(f"Prediction: {label}. Fraud Probability: {prob:.2f}")

            if pred == 1:
                st.error(f"Prediction: **Fraud Detected** (Probability: {prob:.2f})")
            else:
                st.success(f"Prediction: **No Fraud** (Probability: {prob:.2f})")

            slow_celebration()

            csv_data = df_advanced_tx.to_csv(index=False).encode('utf-8')
            st.download_button("Download Synthetic Transaction Dataset", data=csv_data, file_name="advanced_transaction_data.csv", mime="text/csv")

elif selected == "Document & Media Fraud Detection":
    if not st.session_state.logged_in:
        st.error("Please log in to access Document & Media Fraud Detection.")
    else:
        st.markdown("<div class='card fade-in'><h2>Document & Media Fraud Detection</h2><p>Upload an image, PDF, Word doc, or text file for scam content analysis.</p></div>", unsafe_allow_html=True)
        if not OCR_AVAILABLE:
            st.error("OCR and required libraries are not installed. Please install pytesseract, Pillow, PyPDF2, and python-docx.")
        else:
            uploaded_file = st.file_uploader("Choose a file (png/jpg/jpeg/pdf/docx/txt)", type=["png", "jpg", "jpeg", "pdf", "docx", "txt"])
            if uploaded_file is not None:
                extracted_text = ""
                file_name = uploaded_file.name.lower()
                try:
                    if file_name.endswith((".png", ".jpg", ".jpeg")):
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                        extracted_text = pytesseract.image_to_string(image)
                    elif file_name.endswith(".pdf"):
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            extracted_text += page.extract_text() or ""
                    elif file_name.endswith(".docx"):
                        doc = docx.Document(uploaded_file)
                        for para in doc.paragraphs:
                            extracted_text += para.text + "\n"
                    elif file_name.endswith(".txt"):
                        extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")

                if extracted_text.strip():
                    st.markdown(f"<div class='card fade-in'><h3>Extracted Text Preview</h3><pre>{extracted_text[:500]}{'...' if len(extracted_text)>500 else ''}</pre></div>", unsafe_allow_html=True)
                    st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt", mime="text/plain")
                    if st.button("Analyze for Fraud"):
                        df_input = pd.DataFrame([{"image_text": extracted_text}])
                        doc_model = get_fraud_model("Document Fraud")
                        pred, prob = predict_fraud(doc_model, df_input)
                        play_audio(pred)
                        play_voice_output(f"Result: {pred}. Fraud Probability: {prob:.2f}")

                        st.markdown(f"<div class='card fade-in'><h3>Result: {pred}</h3><p>Fraud Probability: {prob:.2f}</p></div>", unsafe_allow_html=True)
                        alert, advice, support = prevention_info("Document Fraud", pred, prob)
                        st.markdown(f"<div class='card fade-in'><h3>Alert Level: {alert}</h3><p>{advice}</p><p><strong>Contact:</strong> {support}</p></div>", unsafe_allow_html=True)
                        slow_celebration()
                else:
                    st.warning("No text could be extracted from the file.")

elif selected == "FAQ":
    if not st.session_state.logged_in:
        st.error("Please log in to view FAQs.")
    else:
        st.markdown("<div class='card fade-in'><h2>Frequently Asked Questions</h2></div>", unsafe_allow_html=True)
        faqs = {
            "What is the Business Fraud Detection Platform?": "Our platform is an AI-powered solution designed to detect and prevent fraudulent activities across various business channels.",
            "How does the platform detect fraud?": "It uses advanced machine learning and natural language processing techniques to analyze patterns in data and identify anomalies.",
            "What types of fraud can be detected?": "The system can detect email fraud, social media fraud, web ad fraud, transaction fraud, invoice fraud, insurance fraud, and more.",
            "How secure is the platform?": "We use encrypted databases, secure protocols, and regular audits to ensure your data is safe.",
            "Can the platform integrate with existing systems?": "Yes, our platform offers APIs and webhooks for seamless integration.",
            "Who has access to the platform?": "Access is restricted to approved users; administrators have additional privileges for managing accounts.",
            "What is the response time for fraud alerts?": "Fraud alerts are generated in real-time, with customizable notification settings.",
            "How can I contact support?": "You can reach our support team via email at support@yourcompany.com or call 1800-BIZ-HELP."
        }
        for question, answer in faqs.items():
            st.markdown(f"**Q: {question}**")
            st.markdown(f"A: {answer}")
            st.markdown("---")

elif selected == "About":
    if not st.session_state.logged_in:
        st.error("Please log in to view the About page.")
    else:
        st.markdown("<div class='card fade-in'><h2>About the Platform</h2></div>", unsafe_allow_html=True)
        st.markdown("""
            **Business Fraud Detection Platform**

            Leveraging state-of-the-art AI, this platform monitors and detects fraudulent activities in business communications and transactions.

            **Key Features:**
            - Multi-channel data ingestion (emails, social media, ads, transactions, logs, documents)
            - Advanced NLP and ML for real-time fraud prediction
            - Comprehensive dashboards for historical analysis and trends
            - Fraud prevention guidelines and educational resources
            - AI-powered chatbot for fraud inquiries
            - OCR-based parsing for document/image analysis
            - Continuous model updates to adapt to evolving threats

            **Developed by:** Your Team (2025)
        """)
        st.image("https://source.unsplash.com/1600x900/?technology,business", use_container_width=True)

elif selected == "Contact Us":
    if not st.session_state.logged_in:
        st.error("Please log in to access the Contact Us form.")
    else:
        st.markdown("<div class='card fade-in'><h2>Contact Us</h2></div>", unsafe_allow_html=True)
        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            submitted = st.form_submit_button("Send Message")
            if submitted:
                if name and email and message:
                    st.success("Thank you for contacting us! We will get back to you soon.")
                    play_voice_output("Thank you for contacting us. We will get back to you soon.")
                else:
                    st.warning("Please fill in all fields.")

elif selected == "Settings":
    if not st.session_state.logged_in:
        st.error("Please log in to access Settings.")
    else:
        if st.session_state.user_role != "admin":
            st.error("Only administrators can change the settings.")
        else:
            st.markdown("<div class='card fade-in'><h2>Settings</h2><p>Adjust platform appearance and header customization below.</p></div>", unsafe_allow_html=True)
            st.subheader("General Appearance")

            theme_choice = st.selectbox("Select UI Theme", list(preset_themes.keys()), 
                                        index=list(preset_themes.keys()).index(st.session_state.theme_name))
            if theme_choice != st.session_state.theme_name:
                st.session_state.theme_name = theme_choice
                for key, value in preset_themes[theme_choice].items():
                    st.session_state[key] = value
                safe_rerun()

            selected_font = st.selectbox("Font Family", ["Roboto Mono", "Arial", "Times New Roman", "Courier New"],
                                         index=["Roboto Mono", "Arial", "Times New Roman", "Courier New"].index(st.session_state.font_family))
            st.session_state.font_family = selected_font

            font_size_value = st.slider("Base Font Size", 12, 24, int(st.session_state.font_size.replace("px", "")))
            st.session_state.font_size = f"{font_size_value}px"

            st.session_state.primary_color = st.color_picker("Primary Color", st.session_state.primary_color)
            st.session_state.secondary_color = st.color_picker("Secondary Color", st.session_state.secondary_color)
            st.session_state.background_color = st.color_picker("Background Color", st.session_state.background_color)
            st.session_state.card_bg = st.color_picker("Card Background Color", st.session_state.card_bg)
            st.session_state.text_color = st.color_picker("Text Color", st.session_state.text_color)
            st.session_state.sidebar_bg = st.color_picker("Sidebar Background", st.session_state.sidebar_bg)
            st.session_state.sidebar_text = st.color_picker("Sidebar Text Color", st.session_state.sidebar_text)

            button_style = st.selectbox("Button Style", ["Rounded", "Square"])
            alert_sound = st.selectbox("Alert Sound", ["Default", "Chime", "None"])
            notification_freq = st.radio("Notification Frequency", ["Real-time", "Hourly", "Daily"], index=0)
            auto_refresh = st.slider("Auto-Refresh Interval (sec)", 10, 300, 60, 10)
            download_format = st.selectbox("Data Download Format", ["CSV", "JSON", "XLSX"])
            dark_mode = st.checkbox("Enable Dark Mode", value=False)
            time_zone = st.selectbox("Time Zone", ["UTC", "Local", "Custom"])
            language_pref = st.selectbox("Preferred Language", ["English", "Spanish", "French", "German"])
            privacy_mode = st.checkbox("Enable Privacy Mode", value=True)
            session_timeout = st.slider("Session Timeout (mins)", 5, 120, 30, 5)
            captcha_enabled = st.checkbox("Enable Captcha for Login", value=False)
            analytics_tracking = st.checkbox("Enable Analytics Tracking", value=True)
            user_activity_log = st.checkbox("Log User Activity", value=True)

            st.subheader("Header Customization")
            header_title = st.text_input("Header Title", value=st.session_state.header_title)
            header_subtitle = st.text_input("Header Subtitle", value=st.session_state.header_subtitle)
            header_bg_image = st.text_input("Header Background Image URL", value=st.session_state.header_bg_image)

            st.markdown("### Additional Options")
            advanced_settings = st.checkbox("Enable Advanced Analytics", value=True)
            fraud_alerts = st.checkbox("Enable Fraud Alerts", value=True)
            st.markdown("---")

            if st.button("Save Changes"):
                st.session_state.header_title = header_title
                st.session_state.header_subtitle = header_subtitle
                st.session_state.header_bg_image = header_bg_image
                st.success("Settings have been saved successfully!")
                play_voice_output("Your settings have been saved successfully.")
                safe_rerun()


# -----------------------------------------------------------------------------
#   12) FOOTER & FLOATING HELP BUTTON
# -----------------------------------------------------------------------------

st.markdown("<div class='footer fade-in'><p>© 2025 Business Fraud Detection Platform. All rights reserved.</p></div>", unsafe_allow_html=True)

# Floating help button example
help_button_html = f"""
<div class="float-help">
  <i class="fa fa-question"></i>
  <div class="help-tooltip">
    Need Help?<br>
    - Try toggling themes<br>
    - Explore "Fraud Detection"<br>
    - Use the Chatbot feature!
  </div>
</div>
"""
st.markdown(help_button_html, unsafe_allow_html=True)

st.markdown("""
<div class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">
    <i class="fa fa-arrow-up"></i>
</div>
""", unsafe_allow_html=True)
