import cv2
import cv2.aruco as aruco

# Charger l'image
webcam = cv2.VideoCapture(0)
#on recupere frame par frame
ret, image = webcam.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Choisir le dictionnaire ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Créer le détecteur avec ses paramètres
detector = aruco.ArucoDetector(aruco_dict)

# Détecter les marqueurs
corners, ids, rejected = detector.detectMarkers(gray)

# Vérifier si des marqueurs ont été détectés
if ids is not None:
    aruco.drawDetectedMarkers(image, corners, ids)
    print(f"Marqueurs détectés : {ids.flatten()}")
else:
    print("Aucun marqueur détecté.")

# Afficher le résultat
cv2.imshow("Détection ArUco", image)
cv2.waitKey(0)
cv2.destroyAllWindows()