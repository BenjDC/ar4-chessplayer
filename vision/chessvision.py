import cv2
import numpy as np

# ----------------------
# CONFIGURATION
# ----------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# IDs selon ton placement
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

    # -----------------
    # BORDS RÉELS DU PLATEAU
    # -----------------

    # TAGS DU BAS → bord supérieur du tag = bord du plateau
    BL = pts19[3]   # coin haut-gauche du tag 19
    BR = pts29[0]   # coin haut-droit  du tag 19

    # TAGS DU HAUT → bord inférieur du tag = bord du plateau
    TR = pts39[0]   # coin bas-gauche du tag 39
    TL = pts9[1]    # coin bas-droit  du tag 9

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
# PROGRAMME PRINCIPAL
# ----------------------

if __name__ == "__main__":
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

    cv2.imshow("Coins détectés (debug)", dbg)
    cv2.waitKey(0)

    # Warp
    board_warped, M = warp_board(img, pts_board)
    cv2.imshow("Plateau rectifié", board_warped)
    cv2.waitKey(0)

    # Découpe en cases
    cases = slice_into_64_cases(board_warped)

    # Test visuel de quelques cases
    cv2.imshow("a1", cases["a1"])
    cv2.imshow("d4", cases["d4"])
    cv2.imshow("h8", cases["h8"])
    cv2.waitKey(0)

    cv2.destroyAllWindows()