import cv2
import chess
import numpy as np
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

def get_board():
    webcam = cv2.VideoCapture(0)
    #on recupere frame par frame
    ret, img = webcam.read()

    cv2.imwrite("essai2.png", img)
    cv2.imshow('dbg', img)
    cv2.waitKey()

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

    cv2.imshow('dbg', dbg)
    cv2.waitKey()
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

def predict_board_occupancy(cases_dict, model, debug=False, img_size=128):
    """
    cases_dict : dict {"a8": image, ..., "h1": image}
    model : modèle Keras déjà chargé
    debug : affiche un échiquier annoté si True
    img_size : taille de redimensionnement du CNN
    """

    class_names = ["piece_blanche", "piece_noire", "vide"]
    results = {}

    # Préparation batch pour meilleure rapidité
    X_batch = []

    # Conserver l'ordre des cases pour correspondance batch -> case
    ordered_case_keys = SQUARES_ORDER

    for key in ordered_case_keys:
        img = cases_dict[key]

        # Resize → Normalisation
        img_resized = cv2.resize(img, (img_size, img_size))
        img_resized = img_resized.astype("float32") / 255.0
        X_batch.append(img_resized)

    X_batch = np.array(X_batch)

    # Prédiction
    preds = model.predict(X_batch)

    for i, key in enumerate(ordered_case_keys):
        label_id = np.argmax(preds[i])
        label = class_names[label_id]
        results[key] = label

    # --- Mode debug : affichage graphique de l'échiquier ---
    if debug:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Résultats du CNN")

        # Couleurs pour visualiser rapidement
        colors = {
            "vide": "#EEEEEE",
            "piece_blanche": "#88DDFF",
            "piece_noire": "#4444AA"
        }

        # Affichage case par case
        for r in range(8):
            for c in range(8):
                square = chr(ord('a') + c) + str(8 - r)
                label = results[square]

                rect = plt.Rectangle(
                    (c, r), 1, 1,
                    facecolor=colors[label],
                    edgecolor="black"
                )
                ax.add_patch(rect)

                ax.text(
                    c + 0.5, r + 0.5,
                    label.replace("piece_", "").replace("_", " "),
                    ha="center", va="center", fontsize=8
                )

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a','b','c','d','e','f','g','h'])
        ax.set_yticklabels(['8','7','6','5','4','3','2','1'])
        ax.invert_yaxis()
        plt.show()

    return results


def detect_move_from_occupancy(fen_before, occupancy_after):
    """
    fen_before : FEN initial
    occupancy_after : dict {"a1": "white"/"black"/"empty"}

    Retourne un objet chess.Move décrivant le coup joué.
    Ambiguïté restante : promotion (renvoyée sans pièce de promo).
    """

    board_before = chess.Board(fen_before)

    # Convertit le FEN initial en occupation simplifiée
    occupancy_before = {}
    for square in chess.SQUARES:
        piece = board_before.piece_at(square)
        sq_name = chess.square_name(square)
        if piece is None:
            occupancy_before[sq_name] = "empty"
        else:
            occupancy_before[sq_name] = "white" if piece.color == chess.WHITE else "black"

    # Liste des différences
    removed = []
    added = []

    for sq in occupancy_before:
        if occupancy_before[sq] != occupancy_after[sq]:
            if occupancy_before[sq] != "empty" and occupancy_after[sq] == "empty":
                removed.append(sq)
            if occupancy_before[sq] == "empty" and occupancy_after[sq] != "empty":
                added.append(sq)

    # -----------------------------
    # 1) CAS DU ROQUE
    # -----------------------------
    # Blanc
    if board_before.turn == chess.WHITE:
        if occupancy_before["e1"] == "white":
            # Petit roque : e1→g1, h1→f1
            if occupancy_after["e1"] == "empty" and occupancy_after["g1"] == "white":
                return chess.Move.from_uci("e1g1")
            # Grand roque : e1→c1, a1→d1
            if occupancy_after["e1"] == "empty" and occupancy_after["c1"] == "white":
                return chess.Move.from_uci("e1c1")
    else:
        # Noir
        if occupancy_before["e8"] == "black":
            # Petit roque
            if occupancy_after["e8"] == "empty" and occupancy_after["g8"] == "black":
                return chess.Move.from_uci("e8g8")
            # Grand roque
            if occupancy_after["e8"] == "empty" and occupancy_after["c8"] == "black":
                return chess.Move.from_uci("e8c8")

    # -----------------------------
    # 2) CAS STANDARD
    # -----------------------------
    if len(removed) == 1 and len(added) == 1:
        move = chess.Move.from_uci(removed[0] + added[0])
        return move

    # -----------------------------
    # 3) PRISE EN PASSANT
    # -----------------------------
    if len(removed) == 2 and len(added) == 1:
        # Exemple : pion blanc joue e5xd6 en passant → d5 disparaît
        # On cherche un pion disparu sur une autre colonne
        dest = added[0]
        for r in removed:
            if r != dest:
                # On teste si c'est cohérent avec un EP autorisé dans FEN
                for move in board_before.legal_moves:
                    if move.to_square == chess.parse_square(dest) and board_before.is_en_passant(move):
                        return move

    # -----------------------------
    # 4) PROMOTION (AMBIGUË)
    # -----------------------------
    # Pion monte : removed = 1, added = 1 mais sur rangée 8/1
    if len(removed) == 1 and len(added) == 1:
        src = removed[0]
        dst = added[0]

        # Blanc promu si arrivée en rangée 8
        if dst[1] == "8" and occupancy_before[src] == "white":
            return chess.Move.from_uci(src + dst)  # promo manquante

        # Noir promu si arrivée en rangée 1
        if dst[1] == "1" and occupancy_before[src] == "black":
            return chess.Move.from_uci(src + dst)

    # -----------------------------
    # Rien trouvé
    # -----------------------------
    return None