import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io

# ===============================
# 🧠 CONFIGURATION GÉNÉRALE
# ===============================
st.set_page_config(
    page_title="💳 Détection de Fraude Bancaire",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# 🎨 STYLE PERSONNALISÉ + ANIMATIONS
# ===============================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }

        .stButton>button {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            transition: 0.3s;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #0072ff, #00c6ff);
        }

        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #00c6ff;
            margin-bottom: 20px;
            animation: fadeIn 2s ease-in-out;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .fade-in {
            animation: fadeIn 1.5s ease-in;
        }

        .sidebar-content {
            text-align: center;
        }
        .sidebar-logo {
            border-radius: 50%;
            width: 90px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 🧩 CHARGEMENT DES RESSOURCES
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("modele_fraude.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("creditcarddata_propre.csv")

modele = load_model()
data = load_data()

# ===============================
# 🌐 BARRE LATÉRALE
# ===============================
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/4727/4727258.png", width=100, caption="Smart Fraud Detector", output_format="auto")
    st.markdown("### 🧭 Navigation")
    section = st.radio("", ["🏠 Accueil","🧹 Nettoyage & Préparation", "📊 Exploration", "🧠 Prédiction", "📈 Performance", "ℹ️ À propos"], index=0)

    st.markdown("---")
    st.markdown("### 👨‍💻 Projet réalisé par :")
    st.markdown("**Cheikh Ahmadou Ka**")
    st.markdown("[📧 Contact Email](mailto:contact@example.com)")
    st.markdown("<div style='font-size:12px; color:gray;'>© 2025 - Projet IA Master DSGL</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# 🧭 NAVIGATION ENTRE SECTIONS
# ===============================

# 🏠 ACCUEIL
if section == "🏠 Accueil":
    st.markdown("<h1 class='main-title'>💳 Application de Détection de Fraude Bancaire</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='fade-in'>
    Bienvenue dans cette application intelligente développée dans le cadre d’un **projet de Master en Data Science & Génie Logiciel (DSGL)**.  
    Elle repose sur un modèle d’intelligence artificielle capable de **détecter automatiquement les transactions bancaires suspectes**.

    ### 🎯 Objectifs :
    - Identifier les transactions à haut risque de fraude  
    - Analyser et visualiser les données clients  
    - Fournir un outil interactif et intuitif pour la prise de décision  
    </div>
    """, unsafe_allow_html=True)

    st.image("f1.jpg", use_container_width=True)

# 📊 EXPLORATION

elif section == "🧹 Nettoyage & Préparation":
    st.subheader("🧹 Chargement et Nettoyage Automatique d’un Nouveau Dataset Bancaire")

    uploaded_file = st.file_uploader("📂 Téléverser un fichier CSV brut :", type=["csv"])

    if uploaded_file is not None:
        # Lecture du CSV
        data_raw = pd.read_csv(uploaded_file)
        st.success("✅ Fichier chargé avec succès !")
        st.write("Aperçu initial :")
        st.dataframe(data_raw.head())

        # --- Étape 1 : suppression des doublons ---
        n_before = len(data_raw)
        data_raw.drop_duplicates(inplace=True)
        n_after = len(data_raw)
        st.info(f"🗑️ Doublons supprimés : {n_before - n_after}")

        # --- Étape 2 : gestion des valeurs manquantes ---
        st.info("🩹 Remplissage des valeurs manquantes (NaN) :")
        missing_before = data_raw.isnull().sum().sum()

        # Remplir par la médiane pour les numériques, mode pour catégorielles
        for col in data_raw.columns:
            if data_raw[col].dtype in ['int64', 'float64']:
                data_raw[col].fillna(data_raw[col].median(), inplace=True)
            else:
                data_raw[col].fillna(data_raw[col].mode()[0], inplace=True)
        missing_after = data_raw.isnull().sum().sum()
        st.success(f"Valeurs manquantes traitées : {missing_before - missing_after}")

        # --- Étape 3 : détection et suppression des outliers (méthode IQR) ---
        numeric_cols = data_raw.select_dtypes(include=np.number).columns.tolist()
        def remove_outliers_iqr(df, col):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return df[(df[col] >= lower) & (df[col] <= upper)]

        st.info("📉 Suppression des valeurs aberrantes...")
        for col in numeric_cols:
            n_before = len(data_raw)
            data_raw = remove_outliers_iqr(data_raw, col)
            n_after = len(data_raw)
            st.write(f" - {col}: {n_before - n_after} outliers supprimés")

        # --- Étape 4 : encodage des variables catégorielles ---
        categorical_cols = data_raw.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            st.info("🔤 Encodage des variables catégorielles...")
            data_raw = pd.get_dummies(data_raw, columns=categorical_cols, drop_first=True)
            st.success(f"{len(categorical_cols)} variables catégorielles encodées")

        # --- Étape 5 : normalisation des variables numériques ---
        st.info("⚖️ Normalisation des variables numériques...")
        scaler = StandardScaler()
        data_raw[numeric_cols] = scaler.fit_transform(data_raw[numeric_cols])
        st.success("Variables numériques normalisées avec succès ✅")

        # --- Étape 6 : affichage final ---
        st.subheader("📊 Dataset propre")
        st.dataframe(data_raw.head())

        # --- Étape 7 : téléchargement du dataset nettoyé ---
        cleaned_csv = data_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Télécharger le dataset nettoyé",
            data=cleaned_csv,
            file_name="donnees_bancaires_propres.csv",
            mime="text/csv"
        )

elif section == "📊 Exploration":
    st.subheader("📈 Exploration des Données")

    st.write("Aperçu du dataset :")
    st.dataframe(data.head())

    variable = st.selectbox("Choisir une variable à visualiser :", data.columns)
    fig = px.histogram(data, x=variable, color_discrete_sequence=['#00c6ff'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de corrélation")
    corr = data.corr(numeric_only=True)
    fig_corr = px.imshow(corr, color_continuous_scale="Blues", text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

# 🧠 PRÉDICTION
elif section == "🧠 Prédiction":
    st.subheader("🤖 Prédire une Transaction")

    choix_mode = st.radio("Choisir le mode de prédiction :", ["🎯 Individuelle", "📂 Par CSV"])

    if choix_mode == "🎯 Individuelle":
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Âge du client", 18, 80, 30)
        with col2:
            transaction = st.number_input("Montant de la transaction", 0.0, 2000.0, 100.0)
        with col3:
            expiry = st.number_input("Date d’expiration (année)", 2020, 2100, 2025)

        X_new = pd.DataFrame([[age, transaction, expiry]], columns=["Age", "TransactionAmount", "CardExpiryDate"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_new)

        if st.button("🔎 Lancer la Prédiction"):
            prediction = modele.predict(X_scaled)[0]
            proba = modele.predict_proba(X_scaled)[0][1] if hasattr(modele, 'predict_proba') else None

            if prediction == 1:
                st.error(f"🚨 Alerte : Transaction suspecte (probabilité : {proba:.2%})")
            else:
                st.success(f"✅ Transaction normale (probabilité fraude : {proba:.2%})")

    else:
        uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.dataframe(df_upload.head())

            X_upload = df_upload[["Age", "TransactionAmount", "CardExpiryDate"]]
            scaler = StandardScaler()
            X_upload_scaled = scaler.fit_transform(X_upload)

            preds = modele.predict(X_upload_scaled)
            df_upload["Fraude_Predite"] = preds

            st.success("✅ Prédiction terminée ! Aperçu :")
            st.dataframe(df_upload.head())

            st.download_button(
                "⬇️ Télécharger le fichier avec prédictions",
                df_upload.to_csv(index=False).encode('utf-8'),
                "predictions_fraude.csv",
                "text/csv",
                key='download-csv'
            )

# 📈 PERFORMANCE
elif section == "📈 Performance":
    st.subheader("📊 Évaluation du Modèle")

    y_test = data["PotentialFraud"]
    X_test = data[["Age", "TransactionAmount", "CardExpiryDate"]]
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    y_pred = modele.predict(X_test_scaled)

    acc = np.mean(y_pred == y_test)
    st.metric(label="Précision globale du modèle", value=f"{acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Non Fraude", "Fraude"]).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

# ℹ️ À PROPOS
elif section == "ℹ️ À propos":
    st.subheader("ℹ️ À propos du projet")
    st.markdown("""
    <div class='fade-in'>
    Ce projet a été conçu dans le cadre du **Master en Intelligence Artificielle, Big Data et Génie Logiciel (DSGL)**.  
    Il met en œuvre des techniques d’apprentissage supervisé pour identifier les fraudes bancaires.

    ### 👨‍💻 Auteur :
    **Cheikh Ahmadou Ka**  
    [LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com)

    ### 🧰 Technologies utilisées :
    - Python (Pandas, Scikit-learn,Sklearn, Joblib, Streamlit)
    - Machine Learning supervisé (classification binaire)
    - Interface interactive via Streamlit et Plotly
    </div>
    """, unsafe_allow_html=True)
