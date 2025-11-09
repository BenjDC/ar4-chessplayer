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
    if ids is None:
        raise SystemExit("Aucun tag ArUco détecté.")

    ids = ids.flatten()
    tags = {}
    for i, ID in enumerate(ids):
        tags[int(ID)] = corners[i]

    # Vérification que les 4 tags sont présents
    needed = [ID_BL, ID_BR, ID_TR, ID_TL]
    for ID in needed:
        if ID not in tags:
            raise SystemExit(f"Tag {ID} manquant, il est pourtant attendu.")

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
    return slice_into_64_cases(board_warped)



def extract_case_top(c, ratio=0.2):
    h, w, _ = c.shape
    top_h = int(h * ratio)
    return c[0:top_h, :]

def preprocess_case_for_diff(c):
    # 1. top 20%
    h, w, _ = c.shape
    t = c[:int(h*0.2), :]

    # 2. LAB
    lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 3. Normalisation lumière
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)

    # 4. Stabilisation de la chroma (utile pour cases vertes)
    A2 = cv2.equalizeHist(A)

    # 5. Fusion
    fused = cv2.addWeighted(L2, 0.6, A2, 0.4, 0)

    # 6. Filtrage anti-bruit
    fused = cv2.GaussianBlur(fused, (5,5), 0)

    return fused

# ne compare que le haut de la case
def diff_case_top(c_before, c_after):
    p1 = preprocess_case_for_diff(c_before)
    p2 = preprocess_case_for_diff(c_after)

    diff = cv2.absdiff(p1, p2)
    score = np.mean(diff)

    return score, diff


def diff_all_cases(cases_before, cases_after):
    scores = {}
    for name in cases_before:
        scores[name], diff = diff_case_top(cases_before[name], cases_after[name])
    return scores

def detect_changed_cases(scores):
    # trier du plus grand changement au plus petit
    sorted_cases = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    
    
    # les deux cases les plus différentes
    c1, s1 = sorted_cases[0]
    c2, s2 = sorted_cases[1]

    for i in range(8):
        print(sorted_cases[i])


    return (c1, s1), (c2, s2)

def classify_move(cases_before, cases_after, c1, c2):
    # prendre la moyenne de pixel brut
    def mean_intensity(img):
        return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    b1 = mean_intensity(cases_before[c1])
    a1 = mean_intensity(cases_after[c1])
    b2 = mean_intensity(cases_before[c2])
    a2 = mean_intensity(cases_after[c2])

    delta1 = a1 - b1
    delta2 = a2 - b2

    # destination = intensité augmente
    # source      = intensité diminue
    
    if delta1 > delta2:
        dest = c1
        src = c2
    else:
        dest = c2
        src = c1

    return src, dest

def detect_move(cases_before, cases_after):
    scores = diff_all_cases(cases_before, cases_after)

    (c1, s1), (c2, s2) = detect_changed_cases(scores)

    src, dest = classify_move(cases_before, cases_after, c1, c2)

    return str(src)+str(dest)

def debug_compare_cases(cases_before, cases_after):

    
    cell_h, cell_w, _ = next(iter(cases_before.values())).shape
    top_h = int(cell_h * 0.2)

    # panneau 8x8 contenant 3 bandes : before / after / diff
    panel_h = top_h * 8
    panel_w = cell_w * 8

    before_panel = np.zeros((panel_h, panel_w), dtype=np.uint8)
    after_panel  = np.zeros((panel_h, panel_w), dtype=np.uint8)
    diff_panel   = np.zeros((panel_h, panel_w), dtype=np.uint8)

    # heatmap 8x8 pour scores
    score_grid = np.zeros((8,8), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            case_name = chr(ord("a")+c) + str(8-r)

            score, diff_img = diff_case_top(
                cases_before[case_name],
                cases_after[case_name]
            )

            score_grid[r,c] = score

            # images top 20%
            top_before = extract_case_top(cases_before[case_name])
            top_after  = extract_case_top(cases_after[case_name])

            gb = cv2.cvtColor(top_before, cv2.COLOR_BGR2GRAY)
            ga = cv2.cvtColor(top_after,  cv2.COLOR_BGR2GRAY)

            y1 = r * top_h
            y2 = y1 + top_h
            x1 = c * cell_w
            x2 = x1 + cell_w

            before_panel[y1:y2, x1:x2] = gb
            after_panel[y1:y2,  x1:x2] = ga

            # diff normalisée pour affichage
            d_norm = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
            diff_panel[y1:y2, x1:x2] = d_norm

    # heatmap 8x8 des scores
    heat = cv2.normalize(score_grid, None, 0, 255, cv2.NORM_MINMAX)
    heat = heat.astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

    # affichage des panneaux
    cv2.imshow("DEBUG - top 20% BEFORE", before_panel)
    cv2.imshow("DEBUG - top 20% AFTER",  after_panel)
    cv2.imshow("DEBUG - top 20% DIFF",   diff_panel)
    cv2.imshow("DEBUG - Scores Heatmap", heatmap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def piece_presence_score(case_img):
    # 1. extraire le centre
    c = extract_case_top(case_img)

    # 2. conversion HSV
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3. seuil sur la luminosité (pixels sombres = présence pièce)
    #    80 est très robuste pour cam faible luminosité
    mask_dark = cv2.inRange(v, 0, 80)

    # 4. fermeture pour nettoyer le bruit
    kernel = np.ones((3,3), np.uint8)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)

    # 5. score = nombre de pixels sombres
    score = np.sum(mask_dark // 255)

    return score, mask_dark


def board_state(cases):
    state = {}
    masks = {}
    for name, img in cases.items():
        score, mask = piece_presence_score(img)
        state[name] = (score > 60)       # True = pièce présente
        masks[name] = mask               # pour debug
    return state, masks

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

def debug_silhouettes(masks_before, masks_after):
    # construire un panneau 8x8 montrant les silhouettes
    rows = []
    for r in range(8):
        row_before = []
        row_after = []
        for c in range(8):
            name = chr(ord('a')+c) + str(8-r)
            row_before.append(masks_before[name])
            row_after.append(masks_after[name])
        rows.append(np.hstack(row_before))
        rows.append(np.hstack(row_after))

    panel = np.vstack(rows)
    cv2.imshow("Silhouettes BEFORE / AFTER", panel)
    cv2.waitKey(0)