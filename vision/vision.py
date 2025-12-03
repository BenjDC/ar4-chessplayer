import cv2
import numpy as np

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

        px_per_mm = size_tag / 30.0      # 30 mm = zone noire
        return 5.0 * px_per_mm    # marge extérieure

    
    
    # -----------------
    # BORDS RÉELS DU PLATEAU
    # -----------------

    # TAGS DU BAS → bord supérieur du tag = bord du plateau
    margin_px = tag_margin_size(pts19)
    BL = pts19[3] + np.array([-margin_px, -margin_px])   # coin haut-gauche du tag 19

    margin_px = tag_margin_size(pts29)
    BR = pts29[0] + np.array([margin_px, -margin_px])  # coin haut-droit  du tag 29

    # TAGS DU HAUT → bord inférieur du tag = bord du plateau
    margin_px = tag_margin_size(pts39)
    TR = pts39[0]  + np.array([-margin_px, -margin_px])  # coin bas-gauche du tag 39

    margin_px = tag_margin_size(pts9)
    TL = pts9[1] + np.array([margin_px, -margin_px])   # coin bas-droit  du tag 9

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

    corners, ids = detect_aruco(img)

    if (ids == None):
        print("échec vision plateau : aucun tag aruco détecté")
        return None
    
    if (ids.size() < 4):
        print(f"Erreur : 4 tags aruco attendus, tags détectés ", ids)
        return None

    ids = ids.flatten()
    tags = {}
    for i, ID in enumerate(ids):
        tags[int(ID)] = corners[i]

    # Vérification que les 4 tags sont présents
    needed = [ID_BL, ID_BR, ID_TR, ID_TL]
    for ID in needed:
        if ID not in tags:
            print(f"Tag {ID} manquant, il est pourtant attendu.")
            return None, None

    # Récupérer les 4 coins exacts du plateau
    pts_board = get_board_corners_from_tags(tags)

    # Debug affichage des coins sur l’image
    dbg = img.copy()
    for p in pts_board:
        cv2.circle(dbg, (int(p[0]), int(p[1])), 10, (0,0,255), -1)

    # Warp
    board_warped, M = warp_board(img, pts_board)
    board_warped=draw_grid(board_warped)

    # Découpe en cases
    return board_warped, slice_into_64_cases(board_warped)

def extract_case_top(c, ratio=0.2):
    h, w, _ = c.shape

    margin = ratio * ratio
    top_h = int(h * ratio)
    return c[0:top_h, :]

# def extract_case_top(case_img, ratio=0.25, margin_pct=0.05):
#     h, w, _ = case_img.shape

#     # marges en px
#     margin = int(h * margin_pct)
#     band_h = int(h * ratio)

#     # début = marge
#     y1 = margin
#     # fin = y1 + hauteur de la bande
#     y2 = y1 + band_h

#     # protection pour éviter de dépasser
#     y2 = min(y2, h)

#     return case_img[y1:y2, :]

def extract_case_top(case_img, ratio=0.4):
    h, w, _ = case_img.shape
    dh = int(h * ratio)
    dw = int(w * ratio)
    y1 = (h - dh) // 2
    x1 = (w - dw) // 2
    return case_img[y1:y1+dh, x1:x1+dw]

def piece_variance_score(case_img):
    c = extract_case_top(case_img)
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return g.var()

def piece_presence_score(case_img):
    # 1. extraire centre
    c = extract_case_top(case_img)

    # 2. HSV
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3. pixels sombres
    mask_dark = cv2.inRange(v, 0, 80)

    # 4. petite fermeture morphologique
    kernel = np.ones((3,3), np.uint8)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)

    # 5. score = nombre de pixels sombres
    score = np.sum(mask_dark // 255)

    return score, mask_dark
    

def piece_presence_fused(case_img):
    # silhouette sombre
    dark_score, mask = piece_presence_score(case_img)
    
    # variance
    var_score = piece_variance_score(case_img)

    # seuils raisonnables
    DARK_THRESHOLD = 60
    VAR_THRESHOLD  = 150

    is_dark = dark_score > DARK_THRESHOLD
    is_textured = var_score > VAR_THRESHOLD
    
    # Une pièce blanche sur case blanche est détectée par variance
    # Une pièce noire / case noire est détectée par silhouette
    presence = is_dark or is_textured

    return presence, dark_score, var_score, mask


def board_state(cases):
    state = {}
    debug = {}
    for name, img in cases.items():
        presence, dark_score, var_score, mask = piece_presence_fused(img)
        state[name] = presence
        debug[name] = {
            "mask": mask,
            "dark": dark_score,
            "var": var_score,
            "presence": presence,
        }
    return state, debug

def detect_move_from_state(state_before, state_after):
    src = None
    dst = None

    for case in state_before:
        before = state_before[case]
        after = state_after[case]

        if before and not after:
            src = case
        if not before and after:
            dst = case

    return str(src)+str(dst)

def analyze_case(case_img):
    center = extract_case_top(case_img)

    # HSV → pixels sombres
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    mask_dark = cv2.inRange(v, 0, 60)
    dark_score = np.sum(mask_dark // 255)

    # Variance (pièces blanches)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    var_score = gray.var()

    # Indicateur présence
    DARK_THR = 1000
    VAR_THR  = 150
    presence = (dark_score > DARK_THR) or (var_score > VAR_THR)

    return presence, dark_score, var_score, center, mask_dark

def enhance_contrast(case_img):
    # convert LAB (beaucoup plus robuste que RGB ou HSV)
    lab = cv2.cvtColor(case_img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # application CLAHE sur la luminance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)

    # reconstruction
    lab2 = cv2.merge([L2, A, B])
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return enhanced

############################################
# Debug grid : vue complète 8×8
############################################
def debug_grid(cases):
    CELL = 200   # taille d’affichage d'une case dans la grille debug

    grid = np.zeros((8*CELL, 8*CELL, 3), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            sq = chr(ord('a')+c) + str(8-r)
            img = enhance_contrast(cases[sq])

            presence, dark_s, var_s, center, mask_dark = analyze_case(img)

            # créer une vignette visuelle
            # zone 1 : zone centrale analysée
            center_viz = cv2.resize(center, (CELL//2, CELL//2))

            # zone 2 : silhouette sombre
            mask_viz = cv2.cvtColor(mask_dark, cv2.COLOR_GRAY2BGR)
            mask_viz = cv2.resize(mask_viz, (CELL//2, CELL//2))

            # combiner les deux (haut = center, bas = silhouette)
            top_block = center_viz
            bottom_block = mask_viz

            block = np.zeros((CELL, CELL, 3), dtype=np.uint8)
            block[0:CELL//2, 0:CELL//2] = top_block
            block[CELL//2: CELL, 0:CELL//2] = bottom_block

            # fond vert = vide, rouge = occupé
            color = (0,255,0) if not presence else (0,0,255)
            block[:, CELL//2:] = color

            # texte
            txt = f"{sq}"
            cv2.putText(block, txt, (CELL//2+10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(block, f"D:{dark_s}", (CELL//2+10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.putText(block, f"V:{int(var_s)}", (CELL//2+10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # placer dans la grille
            y1 = r * CELL
            x1 = c * CELL
            grid[y1:y1+CELL, x1:x1+CELL] = block

    cv2.imshow("DEBUG PIECES - zone analyse + scores + occupation", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()