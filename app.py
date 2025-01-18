import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from google.cloud import vision
from googletrans import Translator
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialisation des modèles et données
cnn_model = load_model('best_cnn_model.h5')  # Modèle CNN
rf_model = joblib.load('random_forest_model.pkl')  # Random Forest
word2vec_model = Word2Vec.load(r'ADEME\word2vec_ademe.model')  # Modèle Word2Vec
ademe_data = pd.read_csv(r'ADEME\ademe_cleaned.csv')  # Données ADEME
data_sud = pd.read_csv(r'Data Sud\data_sud_clusters.csv')  # Données avec clusters
translator = Translator()

# Configurer le client Google Vision
client = vision.ImageAnnotatorClient.from_service_account_json(
    'C:/Users/thoma/Desktop/Project/API_Keys/ML-for-de/practical-bebop-447416-s8-fb3d7fbdb629.json'
)

# Prétraitement des données
data_sud['insee'] = data_sud['insee'].astype(str).str.replace('.0', '', regex=False)

# Colonnes pour les modèles
columns_features = [
    'omr_en_tonnes', 'cs_en_tonnes', 'verre_seul_en_tonnes', 
    'latitude', 'longitude', 'kmeans_cluster', 'gmm_cluster', 
    'hierarchical_cluster', 'PCA_1', 'PCA_2', 'tSNE_1', 'tSNE_2'
]
columns_target = [
    'omr_en_kg_hab_an', 'cs_en_tonnes', 'cs_en_kg_hab_an', 'verre_seul_en_kg_hab_an'
]

# Fonction de classification d'image
def classify_image(image_path, model):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Recyclable" if prediction[0][0] > 0.5 else "Organique"

# Fonction pour détecter et traduire les labels
def detect_labels_with_translation(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
        image = vision.Image(content=content)
    response = client.label_detection(image=image)
    english_labels = [label.description for label in response.label_annotations]
    french_labels = [
        translator.translate(label, src='en', dest='fr').text for label in english_labels
    ]
    return french_labels

# Fonction pour recommander des services
def recommend_services(labels, ademe_data, word2vec_model, code_postal):
    matched_services = []
    for label in labels:
        if label in word2vec_model.wv.key_to_index:
            similar_words = word2vec_model.wv.most_similar(label, topn=5)
            matched_services.extend([word for word, _ in similar_words])

    matched_services = list(set(matched_services))
    recommendations = ademe_data[
        ademe_data['sous_categories'].str.contains('|'.join(matched_services), na=False)
    ]
    code_postal_prefix = str(code_postal)[:2]
    recommendations = recommendations[
        recommendations['Code postal'].astype(str).str.startswith(code_postal_prefix)
    ]
    if recommendations.empty:
        return "Aucune recommandation trouvée."
    return recommendations[['Nom', 'Adresse', 'Ville', 'Code postal', 'action']].head(3)

# Fonction pour prédire les déchets par habitant
def predict_waste_from_postal_code(code_postal, data, model):
    data_filtered = data[data['insee'] == str(code_postal)]
    if data_filtered.empty:
        return f"Aucune donnée pour le code postal {code_postal}."
    input_features = data_filtered[columns_features].iloc[0].values.reshape(1, -1)
    prediction = model.predict(input_features)
    return dict(zip(columns_target, prediction[0]))

# Interface Streamlit
st.title("Application de Gestion des Déchets")

# Image
image_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])
code_postal = st.text_input("Entrez le code postal")

if st.button("Analyser"):
    if not image_file or not code_postal:
        st.error("Veuillez fournir une image et un code postal.")
    else:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(image_file.getbuffer())
        st.image(image_file, caption="Image téléchargée", use_column_width=True)

        st.subheader("1. Classification de l'image")
        image_class = classify_image("uploaded_image.jpg", cnn_model)
        st.write(f"Résultat : {image_class}")

        if image_class == "Organique":
            st.write("L'objet est organique, aucune action supplémentaire nécessaire.")
        else:
            st.subheader("2. Identification des labels")
            labels = detect_labels_with_translation("uploaded_image.jpg")
            st.write(f"Labels détectés : {labels}")

            st.subheader("3. Recommandation de services")
            services = recommend_services(labels, ademe_data, word2vec_model, code_postal)
            st.write(services)

            st.subheader("4. Prédiction des déchets par habitant")
            waste_prediction = predict_waste_from_postal_code(code_postal, data_sud, rf_model)
            st.write(waste_prediction)
