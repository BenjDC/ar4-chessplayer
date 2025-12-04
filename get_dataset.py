import cv2
import numpy as np
from pathlib import Path
from vision import vision
import os
from datetime import datetime


# === CONFIGURATION ===
N_CAPTURES = 64          # nombre total d’images à capturer
BASE_DIR = "dataset_auto"

# === POSITION INITIALE DES PIÈCES ===
initial_board = [
    ['tour_noir', 'cavalier_noir', 'fou_noir', 'dame_noir', 'roi_noir', 'fou_noir', 'cavalier_noir', 'tour_noir'],
    ['pion_noir', 'pion_noir', 'pion_noir', 'pion_noir', 'pion_noir', 'pion_noir', 'pion_noir', 'pion_noir'], 
    ['vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide'],
    ['vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide'],
    ['vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide'],
    ['vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide', 'vide'],
    ['pion_blanc', 'pion_blanc', 'pion_blanc' ,'pion_blanc', 'pion_blanc', 'pion_blanc', 'pion_blanc', 'pion_blanc'],
    ['tour_blanc', 'cavalier_blanc', 'fou_blanc', 'dame_blanc', 'roi_blanc', 'fou_blanc', 'cavalier_blanc', 'tour_blanc']
]

def dessiner_echiquier(img, tableau):
    """
    Dessine une grille 8x8 sur l'image et superpose le nom des pièces
    selon le tableau donné.
    
    :param img: image OpenCV (numpy array)
    :param tableau: liste 8x8 contenant les noms des pièces ou 'vide'
    :return: image modifiée avec grille et noms des pièces
    """
    # Dimensions de l'image
    height, width = img.shape[:2]
    case_h = height // 8
    case_w = width // 8

    # Dessiner la grille
    for i in range(9):
        # Lignes horizontales
        cv2.line(img, (0, i * case_h), (width, i * case_h), (0, 0, 0), 2)
        # Lignes verticales
        cv2.line(img, (i * case_w, 0), (i * case_w, height), (0, 0, 0), 2)

    # Mettre les noms des pièces
    for i in range(8):
        for j in range(8):
            piece = tableau[i][j]
            if piece != 'vide':
                # Position du texte (centré approximativement)
                x = j * case_w + case_w // 10
                y = i * case_h + case_h // 2
                cv2.putText(img, piece, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

# === DÉCALAGE CYCLIQUE ===
def shift_board(board):
    flat = np.array(board).flatten()
    shifted = np.roll(flat, 1)  # décale toutes les pièces d'une case
    return shifted.reshape((8, 8))

# === DÉCOUPE ET SAUVEGARDE ===
def save_cases_with_labels(cases, board, base_dir=BASE_DIR):
    Path(base_dir).mkdir(exist_ok=True)

    for case_name, case_img in cases.items():

        print(f"[DEBUG] Type de board = {type(board)}, exemple = {board[:2] if isinstance(board, list) else board}")


        # on récupère la pièce à cet emplacement
        row = 8 - int(case_name[1])  # ligne (0 en haut)
        col = ord(case_name[0]) - ord('a')
        label = board[row][col]

        path = f"{base_dir}/dataset/{label}"
        Path(path).mkdir(parents=True, exist_ok=True)

        # vérifications robustes
        if case_img is None or not isinstance(case_img, np.ndarray):
            print(f"[WARN] Case {case_name} invalide, sautée")
            continue

        idx = len(os.listdir(path))
        filename = f"{path}/{label}_{case_name}_{idx}.jpg"
        cv2.imwrite(filename, case_img)


# === CAPTURE ===
def capture_image(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Impossible de capturer l'image.")
    return frame

# === BOUCLE PRINCIPALE ===
def main():
    board = initial_board

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"📸 Démarrage de la capture automatique ({N_CAPTURES} images)...")
    for i in range(N_CAPTURES):
        print(f"\n--- Capture {i+1}/{N_CAPTURES} ---")
        
        board_wraped, cases = vision.get_board()

        if cases == None:
            continue

                
        

        save_cases_with_labels(cases, board, BASE_DIR+"_"+timestamp)
        print("💾 Sauvegarde des cases terminée.")

        debugimg = dessiner_echiquier(board_wraped, board)
        cv2.imshow('capture', debugimg)
        cv2.waitKey()

        # Décale les pièces logiquement
        board = shift_board(board)
        print("♟️  Plateau décalé pour la prochaine capture.")

        


    print("✅ Capture terminée !")

if __name__ == "__main__":
    main()