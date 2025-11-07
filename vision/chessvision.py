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

def detect_board_contour(image_gray: np.ndarray) -> np.ndarray:


    # renforce contours et cherche le plus grand quadrilatère
    blur = cv2.GaussianBlur(image_gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            return approx.reshape(4,2).astype(np.float32)
    return None

    
    

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

def determine_orientation(warped_gray: np.ndarray) -> bool:
    # Retourne True si la case bottom-left (rank1,a-file) est sombre dans le repère transformé.
    # On suppose l'origine (0,0) en haut-left dans warped. bottom-left = (0, size-1)
    s = warped_gray.shape[0] // 8
    # indices en image (row,col), bottom-left cell = row 7, col 0
    r_bl, c_bl = 7, 0
    r_tl, c_tl = 0, 0
    # sample small patch center
    def patch_mean(r,c):
        cx = int((c + 0.5) * s)
        cy = int((r + 0.5) * s)
        half = max(3, s//6)
        patch = warped_gray[cy-half:cy+half, cx-half:cx+half]
        if patch.size == 0:
            return 255
        return float(np.mean(patch))
    mean_bl = patch_mean(r_bl, c_bl)
    mean_tl = patch_mean(r_tl, c_tl)
    # if bottom-left is darker than top-left -> likely standard orientation (a1 dark)
    return mean_bl < mean_tl

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

def main():
    parser = argparse.ArgumentParser(description="Détecte 64 cases d'un échiquier depuis une photo du dessus")
    parser.add_argument("--image", "-i", required=True, help="chemin vers l'image")
    parser.add_argument("--output", "-o", default="squares.json", help="fichier de sortie JSON")
    parser.add_argument("--show", action="store_true", help="afficher l'image avec annotations")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("Impossible de lire l'image.")
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pts = None
    pts = detect_board_contour(gray)

    if pts is None:
        raise SystemExit("Impossible de détecter le plateau automatiquement. Assure-toi que le plateau est bien visible et qu'il occupe une grande surface.")

    warped_size = 800
    warped, M = four_point_transform(orig, pts, dst_size=warped_size)
    Minv = np.linalg.inv(M)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    bboxes = generate_grid_coords(warped_size)

    # déterminer orientation : True => warped bottom-left is dark (so a1 = bottom-left)
    bottom_left_dark = determine_orientation(warped_gray)

    mapping = {}
    for idx, (x1,y1,x2,y2) in enumerate(bboxes):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # transformer centre en coordonnées de l'image originale
        orig_cx, orig_cy = transform_point((cx, cy), Minv)
        # transformer bbox corners pour obtenir bbox approximative dans image originale
        tl = transform_point((x1, y1), Minv)
        br = transform_point((x2, y2), Minv)
        name = square_name_from_index(idx, flip=bottom_left_dark)
        mapping[name] = {
            "center_px": [int(orig_cx), int(orig_cy)],
            "bbox_tl_px": [int(tl[0]), int(tl[1])],
            "bbox_br_px": [int(br[0]), int(br[1])],
            "warp_bbox": [int(x1), int(y1), int(x2), int(y2)]
        }

    # sauvegarde JSON
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Sauvegardé {len(mapping)} cases dans {args.output}")

    if args.show:
        # annoter image originale
        disp = orig.copy()
        for name, info in mapping.items():
            cx, cy = info["center_px"]
            tlx, tly = info["bbox_tl_px"]
            brx, bry = info["bbox_br_px"]
            cv2.rectangle(disp, (tlx,tly), (brx,bry), (0,255,0), 2)
            cv2.putText(disp, name, (cx-12, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        # dessiner contour détecté
        for p in pts.reshape(4,2):
            cv2.circle(disp, (int(p[0]), int(p[1])), 6, (255,0,0), -1)
        cv2.imshow("Detected board", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()