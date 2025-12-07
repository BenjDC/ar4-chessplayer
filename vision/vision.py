import cv2
import chess
import numpy as np
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import models
import torch.nn as nn


#import matplotlib.pyplot as plt

### Détecte un échiquier à partir d'une photo. Se base sur la présence de tags aruco.

# ----------------------
# CONFIGURATION
# ----------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# IDs selon placement : les tags font 3cm de côte avec une marge blanche de 5mm
ID_BL = 19   # bas gauche (sous A1)
ID_BR = 29   # bas droite (sous H1)
ID_TR = 39   # haut droite (à droite de H8)
ID_TL = 9    # haut gauche (à gauche de A8)

WARP_SIZE = 800
CASE_SIZE = WARP_SIZE // 8


# ----------------------
# DETECTION ARUCO
# ----------------------

def detect_aruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids


# ----------------------
# RECONSTRUCTION DES COINS DU PLATEAU
# ----------------------

def get_board_corners_from_tags(tags):
    """
    tags : dict {id: corners(4x2)}
    On utilise les coins du tag qui touchent physiquement le plateau,
    selon TON placement réel.
    """

    # extraire les coins ArUco (4 coins : [0]=TL, [1]=TR, [2]=BR, [3]=BL)
    pts19 = tags[ID_BL].reshape(4,2)  # sous A1
    pts29 = tags[ID_BR].reshape(4,2)  # sous H1
    pts39 = tags[ID_TR].reshape(4,2)  # à droite de H8
    pts9  = tags[ID_TL].reshape(4,2)  # à gauche de A8

    # Taille du carré noir (en px)
    def tag_margin_size(c):
        pts = c.reshape(4,2)
        size_tag = np.mean([
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0])
        ])

        px_per_mm = size_tag / 70.0      # 30 mm = zone noire
        return 5.0 * px_per_mm    # marge extérieure

    
    
    # -----------------
    # BORDS RÉELS DU PLATEAU : haut de la camera côté blancs
    # -----------------

    # TAGS DU BAS → bord supérieur du tag = bord du plateau
    margin_px = tag_margin_size(pts19)
    BL = pts19[3] + np.array([+margin_px, +margin_px * 4])   # coin haut-gauche du tag 19
    # BL = pts19[3] 

    margin_px = tag_margin_size(pts29)
    BR = pts29[0] + np.array([-margin_px, +margin_px * 4])  # coin haut-droit  du tag 29

    # TAGS DU HAUT → bord inférieur du tag = bord du plateau
    margin_px = tag_margin_size(pts39)
    TR = pts39[0]  + np.array([-margin_px, -margin_px * 4])  # coin bas-gauche du tag 39

    margin_px = tag_margin_size(pts9)
    TL = pts9[1] + np.array([+margin_px, -margin_px * 4])   # coin bas-droit  du tag 9

    # Ordre : TL, TR, BR, BL
    return np.array([TL, TR, BR, BL], dtype=np.float32)


# ----------------------
# WARP DU PLATEAU
# ----------------------

def warp_board(image, pts_src):
    pts_dst = np.array([
        [0,0],
        [WARP_SIZE-1,0],
        [WARP_SIZE-1,WARP_SIZE-1],
        [0,WARP_SIZE-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src.astype(np.float32), pts_dst)
    warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))
    return warped, M


# ----------------------
# DÉCOUPE DES 64 CASES
# ----------------------

def slice_into_64_cases(board):
    cases = {}
    for r in range(8):
        for c in range(8):
            x1 = c * CASE_SIZE
            y1 = r * CASE_SIZE
            crop = board[y1:y1+CASE_SIZE, x1:x1+CASE_SIZE]

            case_name = chr(ord('a') + c) + str(8 - r)
            cases[case_name] = crop

    return cases


def assemble_board_from_cases(cases, CASE_SIZE):
    """
    Recompose l'image complète du plateau à partir d'un dict {case_name: crop}.
    
    cases : dict comme {"a1": img, ..., "h8": img}
    CASE_SIZE : taille (en pixels) d'une case (carrée)
    """
    # Créer une grande image vide
    board = np.zeros((8 * CASE_SIZE, 8 * CASE_SIZE, 3), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            # Nom de case inverse du slicing :
            # colonne = lettre, rang = 8 - r
            case_name = chr(ord('a') + c) + str(8 - r)

            if case_name not in cases:
                raise ValueError(f"Case manquante dans cases : {case_name}")

            crop = cases[case_name]

            # Position où coller l'image de la case
            x1 = c * CASE_SIZE
            y1 = r * CASE_SIZE

            board[y1:y1+CASE_SIZE, x1:x1+CASE_SIZE] = crop

    return board

# ----------------------
# Desssine un quadrillage 8x8 sur l'image
# ----------------------

def draw_grid(image):
    # Obtenir les dimensions de l'image
    hauteur, largeur, _ = image.shape

    # Définir le nombre de cases (64 = 8x8)
    rows, cols = 8, 8

    # Calculer la taille d'une case
    case_width = largeur // cols
    case_height = hauteur // rows

    # Couleur et épaisseur des lignes du quadrillage (BGR)
    couleur = (0, 255, 0)  # vert
    epaisseur = 2

    # Dessiner les lignes verticales
    for i in range(1, cols):
        x = i * case_width
        cv2.line(image, (x, 0), (x, hauteur), couleur, epaisseur)

    # Dessiner les lignes horizontales
    for i in range(1, rows):
        y = i * case_height
        cv2.line(image, (0, y), (largeur, y), couleur, epaisseur)

    return image

# ---------------------------------------------------------------
# RECUPERE L'ECHIQUIER ET LA LISTE DES CASES EN PRENANT UNE PHOTO
# ---------------------------------------------------------------

def get_board(img_test=None):

    if (img_test == None):

        webcam = cv2.VideoCapture(0)
        #on recupere frame par frame
        ret, img = webcam.read()    
    else:
        img = cv2.imread(img_test)

    if img is None:
        print(f"Erreur : impossible de charger l'image")
        return

    # cv2.imwrite("essai2.png", img)
    # cv2.imshow('dbg', img)
    # cv2.waitKey()

    corners, ids = detect_aruco(img)

    
    # if (ids.shape[1] < 4):
    #     print(f"Erreur : 4 tags aruco attendus, tags détectés ", ids)
    #     return None

    ids = ids.flatten()
    tags = {}
    for i, ID in enumerate(ids):
        tags[int(ID)] = corners[i]

    # Vérification que les 4 tags sont présents
    needed = [ID_BL, ID_BR, ID_TR, ID_TL]
    manquants = False
    for ID in needed:
        if ID not in tags:
            print(f"Tag {ID} manquant, il est pourtant attendu.")
            manquants = True
    
    if manquants:
        return None, None

    # Récupérer les 4 coins exacts du plateau
    pts_board = get_board_corners_from_tags(tags)

    ##### DEBUG
    # Debug affichage des coins sur l’image
    dbg = img.copy()
    for p in pts_board:
        cv2.circle(dbg, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    # cv2.imshow('dbg', dbg)
    # cv2.waitKey()
    ##### DEBUG

    # Warp
    board_warped, M = warp_board(img, pts_board)

    # Découpe en cases
    return slice_into_64_cases(board_warped)

# ordre cohérent de l'échiquier pour l'affichage (a8 -> h1)
SQUARES_ORDER = [
    chr(ord('a') + c) + str(8 - r)
    for r in range(8) for c in range(8)
]




def print_occupancy(occup):
    print("\n=== OCCUPATION DU PLATEAU ===\n")
    for r in range(8, 0, -1):
        row = ""
        for c in "abcdefgh":
            k = f"{c}{r}"
            
            if occup[k] == "vide":
                row += ". "
            elif occup[k] == "piece_blanche":
                row += "B "
            else:
                row += "N "
        print(row)
    print("\nLégende : . = vide | B = blanc | N = noir\n")


def detect_move_from_occupancy(fen_before, occupancy_after):
    """
    Déduit le coup joué en comparant :
    - fen_before  : FEN avant le coup
    - occupancy_after : dict {"a1": "piece_blanche"/"piece_noire"/"vide"}

    Retourne un objet chess.Move.
    Promotion par défaut : Dame.
    """

    board_before = chess.Board(fen_before)

    # -----------------------------
    # Construction de l'occupation avant
    # -----------------------------
    occupancy_before = {}
    for sq in chess.SQUARES:
        piece = board_before.piece_at(sq)
        name = chess.square_name(sq)
        if piece is None:
            occupancy_before[name] = "vide"
        else:
            occupancy_before[name] = "piece_blanche" if piece.color == chess.WHITE else "piece_noire"

    # -----------------------------
    # Listes des carrés modifiés
    # -----------------------------
    removed = []   # cases où une pièce a disparu
    added = []     # cases où une pièce est apparue

    print("cases devenues vides : ".join(removed))
    print("cases devenues occupées : ".join(added))

    for sq in occupancy_before:
        if occupancy_before[sq] != occupancy_after[sq]:
            if occupancy_before[sq] != "vide" and occupancy_after[sq] == "vide":
                removed.append(sq)
            if occupancy_before[sq] == "vide" and occupancy_after[sq] != "vide":
                added.append(sq)

    # -----------------------------
    # 1) Roques
    # -----------------------------
    if board_before.turn == chess.WHITE:
        # Petit roque : e1->g1
        if occupancy_before["e1"] != "vide" and occupancy_after["e1"] == "vide" and occupancy_after["g1"] == "piece_blanche":
            print("petit roque blanc")
            return chess.Move.from_uci("e1g1")
        # Grand roque : e1->c1
        if occupancy_before["e1"] != "vide" and occupancy_after["e1"] == "vide" and occupancy_after["c1"] == "piece_blanche":
            print("grand roque blanc")
            return chess.Move.from_uci("e1c1")
    else:
        # Petit roque noir
        if occupancy_before["e8"] != "vide" and occupancy_after["e8"] == "vide" and occupancy_after["g8"] == "piece_noire":
            print("petit roque noir")
            return chess.Move.from_uci("e8g8")
        # Grand roque noir
        if occupancy_before["e8"] != "vide" and occupancy_after["e8"] == "vide" and occupancy_after["c8"] == "piece_noire":
            print("grand roque noir")
            return chess.Move.from_uci("e8c8")

    # -----------------------------
    # 2) Coup normal 
    # -----------------------------
    if len(removed) == 1 and len(added) == 1:
        src = removed[0]
        dst = added[0]
        piece = board_before.piece_at(chess.parse_square(src))

        # Cas standard ou prise simple
        move = chess.Move.from_uci(src + dst)
        if move in board_before.legal_moves:
            print("Coup simple "+src+dst)
            return move

    # -----------------------------
    # 2bis) PRISE SIMPLE
    # -----------------------------
    # Une pièce disparaît de 'removed[0]' et apparaît sur une case adverse
    if len(removed) == 1 and len(added) == 0:
        src = removed[0]

        # destination = toute case où l’occupation a changé et où une pièce adverse a disparu
        # c’est-à-dire : occ_before != occ_after et occ_after != "vide"
        candidates = []
        for sq in occupancy_before:
            if occupancy_before[sq] != occupancy_after[sq]:
                # destination = case qui est occupée après
                if occupancy_after[sq] != "vide":
                    candidates.append(sq)

        if len(candidates) == 1:
            dst = candidates[0]
            move = chess.Move.from_uci(src + dst)
        
            if move in board_before.legal_moves:
                print("Prise "+src+dst)
                return move

    # -----------------------------
    # 3) Prise en passant
    # removed = 2 cases : le pion bouge + le pion capturé
    # added   = 1 case  : destination du pion
    # -----------------------------
    if len(removed) == 2 and len(added) == 1:
        dst = added[0]
        for src in removed:
            # src candidat pour le pion qui bouge
            move = chess.Move.from_uci(src + dst)
            if move in board_before.legal_moves and board_before.is_en_passant(move):
                print("Prise en passant "+src+dst)
                return move

    # -----------------------------
    # 4) Promotions
    # -----------------------------
    if len(removed) == 1 and len(added) == 1:
        src = removed[0]
        dst = added[0]
        src_sq = chess.parse_square(src)
        piece = board_before.piece_at(src_sq)

        if piece and piece.piece_type == chess.PAWN:
            # Promotion blanche
            if board_before.turn == chess.WHITE and dst[1] == "8":
                move = chess.Move.from_uci(src + dst + "q")  # promotion en reine
                if move in board_before.legal_moves:
                    print("Promotion "+src+dst)
                    return move

            # Promotion noire
            if board_before.turn == chess.BLACK and dst[1] == "1":
                move = chess.Move.from_uci(src + dst + "q")
                if move in board_before.legal_moves:
                    print("Promotion "+src+dst)
                    return move

    # -----------------------------
    # Rien de valide détecté
    # -----------------------------
    return None


def predict_board_occupancy(cases_dict, model, device="cpu", debug=False):
    """
    cases_dict : dict {'a1': image BGR numpy, ..., 'h8': image }
    model      : modèle PyTorch chargé
    device     : 'cpu' ou 'cuda'
    debug      : affiche un échiquier ASCII des résultats
    """
    
    model.to(device)
    model.eval()

    class_names = ["piece_blanche","piece_noire","vide"]

    results = {}

    with torch.no_grad():
        for sq, img in cases_dict.items():

            # --- Correction du bug des strides négatifs ---
            img_np = np.ascontiguousarray(img)

            # Convertir BGR → RGB
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # Normalisation 0–1
            img_rgb = img_rgb.astype(np.float32) / 255.0

            # Passage HWC → CHW
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

            tensor = tensor.to(device)

            # Prédiction
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).item()

            results[sq] = class_names[pred]

    # ---------------------------------------
    #            DEBUG DISPLAY
    # ---------------------------------------
    if debug:
        print_occupancy(results)

    return results


def load_chess_model(path, device="cpu"):
    # 1. Reconstruire la même archi
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)   # 3 classes

    # 2. Charger le state_dict
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model