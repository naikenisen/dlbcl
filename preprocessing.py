import pandas as pd
import os
from pathlib import Path
import openslide
from PIL import Image

# Créer le dossier dataset s'il n'existe pas
output_dir = Path('dataset')
output_dir.mkdir(exist_ok=True)

# Charger les annotations
annotations = pd.read_csv('annotations.csv')

# Filtrer uniquement les images HE
he_annotations = annotations[annotations['stain'] == 'HE'].copy()

# Grouper par patient pour gérer les numéros de séquence
patient_counts = {}

print(f"Traitement de {len(he_annotations)} régions d'intérêt...")

for idx, row in he_annotations.iterrows():
    patient_id = row['patient_id']
    tma_id = row['tma_id']
    xs, ys, xe, ye = int(row['xs']), int(row['ys']), int(row['xe']), int(row['ye'])
    
    # Déterminer le nom du fichier de sortie
    if patient_id not in patient_counts:
        patient_counts[patient_id] = 1
        output_filename = f"{patient_id}.png"
    else:
        patient_counts[patient_id] += 1
        output_filename = f"{patient_id}_{patient_counts[patient_id]}.png"
    
    # Chemin de l'image source
    image_path = Path('HE') / f"{tma_id}.tiff"
    
    if not image_path.exists():
        print(f"⚠️  Image {image_path} introuvable, passage à la suivante...")
        continue
    
    try:
        # Ouvrir l'image avec OpenSlide et extraire la région d'intérêt
        slide = openslide.OpenSlide(str(image_path))
        
        # Extraire la région d'intérêt (read_region retourne une image RGBA)
        roi = slide.read_region((xs, ys), 0, (xe - xs, ye - ys))
        
        # Convertir en RGB (enlever le canal alpha)
        roi = roi.convert('RGB')
        
        # Sauvegarder la ROI
        output_path = output_dir / output_filename
        roi.save(output_path, 'PNG')
        
        slide.close()
        
        print(f"✓ {output_filename} - Région extraite de {tma_id}.tiff ({xe-xs}x{ye-ys} pixels)")
    
    except Exception as e:
        print(f"✗ Erreur lors du traitement de {output_filename}: {e}")

print(f"\n✅ Extraction terminée! {len(list(output_dir.glob('*.png')))} images sauvegardées dans {output_dir}/")