#!/usr/bin/env python3
"""
detect_board.py
Détecte un échiquier dans une photo prise du dessus, découpe en 8x8 et retourne
pour chaque case ses coordonnées (centre et bbox) dans l'image originale.
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple

def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, dst_size:int=800) -> Tuple[np.ndarray, np.ndarray]:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # destination square
    dst = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))
    return warped, M

def detect_chessboard_by_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    cv2.imshow("debug", gray)
    cv2.waitKey(0)

    # extraction des lignes verticales
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    temp1 = cv2.erode(gray, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(temp1, vertical_kernel, iterations=2)

    # extraction des lignes horizontales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    temp2 = cv2.erode(gray, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(temp2, horizontal_kernel, iterations=2)

    # combinaison
    combined = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0)

    # détection lignes Hough
    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        print("Aucune ligne détectée")
        return None

    # séparer lignes verticales et horizontales
    vertical = []
    horizontal = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:
            vertical.append((x1, y1, x2, y2))
        if abs(y1 - y2) < 10:
            horizontal.append((x1, y1, x2, y2))

    if len(vertical) < 7 or len(horizontal) < 7:
        print("Pas assez de lignes détectées pour former un échiquier.")
        return None

    # sélectionner les 8 lignes les plus espacées
    def keep_8_lines(lines, axis):
        coords = []
        for l in lines:
            x1, y1, x2, y2 = l
            coords.append(x1 if axis == "vertical" else y1)
        coords = np.array(sorted(coords))
        diffs = np.diff(coords)
        # sélectionner l'espacement médian
        median_space = np.median(diffs)
        selected = []
        for c in coords:
            if len(selected) == 0:
                selected.append(c)
            elif abs(c - selected[-1]) > median_space * 0.7:
                selected.append(c)
            if len(selected) == 8:
                break
        return selected

    vertical_x = keep_8_lines(vertical, "vertical")
    horizontal_y = keep_8_lines(horizontal, "horizontal")

    if len(vertical_x) < 8 or len(horizontal_y) < 8:
        print("Échec sélection 8 lignes verticales ou horizontales")
        return None

    # carré du plateau
    x1, x2 = vertical_x[0], vertical_x[-1]
    y1, y2 = horizontal_y[0], horizontal_y[-1]

    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    return pts

def detect_lines_debug(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 40, 120)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=80,
        maxLineGap=20
    )

    debug = image.copy()

    if lines is None:
        print("Aucune ligne détectée")
        return

    # compute angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # cluster angles into two groups
    angles_np = np.array(angles).reshape(-1,1)
    # kmeans 2 clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        angles_np.astype(np.float32),
        2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # draw lines: cluster 0 = bleu, cluster 1 = vert
    colors = [(255,0,0), (0,255,0)]

    for idx, line in enumerate(lines):
        x1,y1,x2,y2 = line[0]
        c = colors[int(labels[idx])]
        cv2.line(debug, (x1,y1), (x2,y2), c, 3)

    cv2.imshow("Debug Lignes Hough (clusterisées)", debug)
    cv2.waitKey(0)

def detect_board_contour(image_gray: np.ndarray) -> np.ndarray:


    # renforce contours et cherche le plus grand quadrilatère
    blur = cv2.GaussianBlur(image_gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 250)

    cv2.imshow("degug", edged)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            return approx.reshape(4,2).astype(np.float32)
    return None

    
def shrink_quad(pts, factor=0.10):
    # pts : array shape (4,2)
    # factor : pourcentage de contraction (0.10 = 10%)
    centroid = pts.mean(axis=0)
    new_pts = centroid + (pts - centroid) * (1.0 - factor)
    return new_pts.astype(np.float32)

def generate_grid_coords(warped_size:int=800) -> List[Tuple[int,int,int,int]]:
    s = warped_size // 8
    bboxes = []
    for r in range(8):
        for c in range(8):
            x1 = c * s
            y1 = r * s
            x2 = x1 + s
            y2 = y1 + s
            bboxes.append((x1,y1,x2,y2))
    return bboxes

def transform_point(pt: Tuple[float,float], Minv: np.ndarray) -> Tuple[int,int]:
    px = np.array([ [pt[0], pt[1], 1.0] ]).T
    map_pt = Minv @ px
    map_pt = map_pt / map_pt[2]
    return int(round(map_pt[0,0])), int(round(map_pt[1,0]))


def square_name_from_index(idx:int, flip:bool) -> str:
    # idx 0..63 in row-major top->bottom, left->right in warped image
    row = idx // 8
    col = idx % 8
    # in warped image row 0 is top; we want rank1 bottom. If flip==False then we must invert row.
    if not flip:
        rank = 8 - row
        file = chr(ord('a') + col)
    else:
        # flip means the warped image already has white at bottom, or orientation reversed
        rank = row + 1
        file = chr(ord('a') + col)
    return f"{file}{rank}"

# if __name__ == "__main__":
#     img = cv2.imread("chess_board_1.jpeg")
#     detect_lines_debug(img)

def main():
    # parser = argparse.ArgumentParser(description="Détecte 64 cases d'un échiquier depuis une photo du dessus")
    # parser.add_argument("--image", "-i", required=True, help="chemin vers l'image")
    # parser.add_argument("--output", "-o", default="squares.json", help="fichier de sortie JSON")
    # parser.add_argument("--show", action="store_true", help="afficher l'image avec annotations")
    # args = parser.parse_args()

    # img = cv2.imread(args.image)
    # if img is None:
    #     raise SystemExit("Impossible de lire l'image.")
    # orig = img.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # detect_lines_debug(img)
    
    # pts = None
    # pts = detect_chessboard_by_lines(img)


    # if pts is None:
    #     raise SystemExit("Impossible de détecter le plateau automatiquement. Assure-toi que le plateau est bien visible et qu'il occupe une grande surface.")

    # warped_size = 800
    
    # # rétrécir le quadrilatère de 10 %
    # pts = shrink_quad(pts, factor=0.0)
    # warped, M = four_point_transform(orig, pts, dst_size=warped_size)

    # #warped, M = four_point_transform(orig, pts, dst_size=warped_size)

    # Minv = np.linalg.inv(M)

    # warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # bboxes = generate_grid_coords(warped_size)


    # mapping = {}
    # for idx, (x1,y1,x2,y2) in enumerate(bboxes):
    #     cx = (x1 + x2) / 2.0
    #     cy = (y1 + y2) / 2.0
    #     # transformer centre en coordonnées de l'image originale
    #     orig_cx, orig_cy = transform_point((cx, cy), Minv)
    #     # transformer bbox corners pour obtenir bbox approximative dans image originale
    #     tl = transform_point((x1, y1), Minv)
    #     br = transform_point((x2, y2), Minv)
    #     name = square_name_from_index(idx, flip=False)
    #     mapping[name] = {
    #         "center_px": [int(orig_cx), int(orig_cy)],
    #         "bbox_tl_px": [int(tl[0]), int(tl[1])],
    #         "bbox_br_px": [int(br[0]), int(br[1])],
    #         "warp_bbox": [int(x1), int(y1), int(x2), int(y2)]
    #     }

    # # sauvegarde JSON
    # with open(args.output, "w") as f:
    #     json.dump(mapping, f, indent=2)
    # print(f"Sauvegardé {len(mapping)} cases dans {args.output}")

    # if args.show:
    #     # annoter image originale
    #     disp = orig.copy()
    #     for name, info in mapping.items():
    #         cx, cy = info["center_px"]
    #         tlx, tly = info["bbox_tl_px"]
    #         brx, bry = info["bbox_br_px"]
    #         cv2.rectangle(disp, (tlx,tly), (brx,bry), (0,255,0), 2)
    #         cv2.putText(disp, name, (cx-12, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1, cv2.LINE_AA)
    #     # dessiner contour détecté
    #     for p in pts.reshape(4,2):
    #         cv2.circle(disp, (int(p[0]), int(p[1])), 6, (255,0,0), -1)
    #     cv2.imshow("Detected board", disp)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    marker = cv2.aruco.generateImageMarker(aruco_dict, 9, 200)
    cv2.imwrite("marker1.jpg", marker)
    marker = cv2.aruco.generateImageMarker(aruco_dict, 19, 200)
    cv2.imwrite("marker2.jpg", marker)
    
    marker = cv2.aruco.generateImageMarker(aruco_dict, 29, 200)
    cv2.imwrite("marker3.jpg", marker)
    
    marker = cv2.aruco.generateImageMarker(aruco_dict, 39, 200)
    cv2.imwrite("marker4.jpg", marker)
    
    

if __name__ == "__main__":
    main()