import streamlit as st
from streamlit_webrtc import webrtc_streamer
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import base64
from gtts import gTTS

# --- CONFIGURATION EXPERT ---
st.set_page_config(page_title="Julien-Scan IA", page_icon="🔍", layout="wide")

# Signature Développeur Julien Banze Kandolo
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .footer { position: fixed; bottom: 10px; right: 10px; color: #58a6ff; font-size: 12px; }
    .dev-name { color: #238636; font-weight: bold; }
    </style>
    <div class="footer">Développé par <span class="dev-name">Julien Banze Kandolo</span> | Expert IA</div>
    """, unsafe_allow_html=True)

# --- INITIALISATION ---
FACES_DIR = "faces"
DB_FILE = "log_julien_scan.csv"

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    # Scan du dossier faces pour charger les photos
    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                # Le nom du fichier devient le nom de la personne
                name = filename.split(".")[0].replace("_", " ").title()
                known_names.append(name)
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()

# --- FONCTION AUDIO (TTS) ---
def play_audio(name):
    text = f"Identification réussie. Bonjour {name}."
    tts = gTTS(text=text, lang='fr')
    tts.save("scan.mp3")
    with open("scan.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
        st.markdown(md, unsafe_allow_html=True)

# --- BASE DE DONNÉES ---
def log_event(name):
    now = datetime.now()
    date_today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")
    if not os.path.exists(DB_FILE):
        pd.DataFrame(columns=["Nom", "Date", "Heure"]).to_csv(DB_FILE, index=False)
    
    df = pd.read_csv(DB_FILE)
    # Eviter les doublons pour la même journée
    if not ((df['Nom'] == name) & (df['Date'] == date_today)).any():
        new_entry = pd.DataFrame([[name, date_today, time_now]], columns=["Nom", "Date", "Heure"])
        new_entry.to_csv(DB_FILE, mode='a', header=False, index=False)
        return True
    return False

# --- TRAITEMENT VIDÉO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Optimisation pour HP ProBook (redimensionnement)
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_img)
    face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "INCONNU"
        color = (0, 0, 255) # Rouge pour inconnu

        if len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                color = (0, 255, 0) # Vert pour reconnu
                
                if name not in st.session_state.get('already_noted', []):
                    if log_event(name):
                        st.session_state['detected_name'] = name

        # Coordonnées réelles
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame.from_ndarray(img, format="bgr24")

# --- INTERFACE PRINCIPALE ---
st.sidebar.markdown(f"## 🛠️ Développeur\n**Julien Banze Kandolo**")
st.sidebar.info("Système d'Identification Biométrique Julien-Scan v1.0")

st.title("🔍 Julien-Scan")
st.write("Analyse faciale et émargement automatique.")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="julien-scan-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
    # Déclenchement de la voix
    if 'detected_name' in st.session_state:
        play_audio(st.session_state['detected_name'])
        del st.session_state['detected_name']

with col2:
    st.subheader("📊 Registre des scans")
    if os.path.exists(DB_FILE):
        st.dataframe(pd.read_csv(DB_FILE).tail(10), use_container_width=True)
    else:
        st.info("En attente du premier scan...")