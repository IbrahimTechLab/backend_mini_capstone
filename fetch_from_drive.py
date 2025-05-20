import gdown
import os
import zipfile

# Dictionnaire des fichiers à télécharger avec leurs IDs Google Drive
GDRIVE_IDS = {
    'modele_maladies.zip': '1NM5cCkqcO7dt3rwVRQ0Ndfj58dU0yNzH',  # ID mis à jour pour le ZIP
    'scaler_maladies.pkl': '1RM9iwdvyecSNuEH_cyYaBRWOAfvVkdwr',
    'label_encoder_maladies.pkl': '1rb1aUGhtHn_dmdB149qWrXDLB_wzCwPO',
    'soil_encoder.pkl': '14EqyjLzD6kp7tB5CPXWg9WbgQUen4TMS'
}

def download_model(filename):
    """Télécharge un fichier spécifique depuis Google Drive, s'il est absent."""
    if filename not in GDRIVE_IDS:
        raise ValueError(f"Fichier non reconnu ou ID manquant : {filename}")

    file_id = GDRIVE_IDS[filename]
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join("models", filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path):
        print(f"Téléchargement de {filename} depuis Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{filename} déjà présent. Pas de téléchargement nécessaire.")

    # Si c'est un fichier ZIP, on le décompresse automatiquement
    if filename.endswith(".zip"):
        print(f"Décompression de {filename}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall("models")
        print(f"{filename} décompressé avec succès.")

if __name__ == "__main__":
    for fname in GDRIVE_IDS:
        download_model(fname)
