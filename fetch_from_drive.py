import gdown
import os

def download_model():
    url = "https://drive.google.com/uc?id=1J_uBeeDSdrY0dnyp7PM25j8IfILAI4bG"
    output_path = "models/modele_maladie.pkl"

    # Crée le dossier models si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Télécharge le fichier s'il n'existe pas déjà
    if not os.path.exists(output_path):
        print("Téléchargement du modèle depuis Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print("Modèle déjà présent. Pas de téléchargement nécessaire.")

if __name__ == "__main__":
    download_model()
