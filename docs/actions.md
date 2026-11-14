# Actions – à faire par le porteur du projet

Les actions qui demandent d'être sur place (fablab, magasin) ou de prendre une décision. Cochez au fur et à mesure. Les résultats (mesures, photos, inventaire) peuvent être transmis tels quels à Claude, qui met à jour la nomenclature, les décisions et les risques.

Dernière mise à jour : 14/11/2026

## 1. Tout de suite

- [x] Relire la pull request #4 (mémoire du projet) et la fusionner. Points à relire en priorité :
  - les exigences encore « à valider » (`exigences.md`) ;
  - les statuts de la nomenclature (`nomenclature.csv`).
- [x] Confirmer le modèle de l'imprimante 3D : Bambu Lab P2S (volume 256 × 256 × 256 mm).
- [ ] Lister les filaments disponibles au fablab (PLA confirmé).

## 2. Inventaire au fablab

But : savoir exactement ce qui est déjà là, pour ne pas acheter en double et pour dimensionner les pièces imprimées. Pour chaque ligne : quantité, référence ou mesures, et une photo si besoin.

**Mécanique**
- [ ] Profilés MakerBeam : nombre et longueur de chaque type (montés et en vrac), couleur
- [ ] Équerres, plaques de jonction, visserie MakerBeam (vis M3, écrous carrés) : quantités restantes
- [ ] Rails MGN9 : nombre et **longueur de chacun**
- [ ] Chariots MGN9H : nombre
- [ ] Poulies : nombre total, puis pour chacune :
  - dentée ou lisse ;
  - nombre de dents ;
  - alésage ;
  - diamètre extérieur ;
  - largeur totale avec flasques ;
  - roulement intégré ou non.
- [ ] Courroie 2GT de 9 mm : longueur restante
- [ ] Moteurs NEMA 17 : nombre (2 seulement, ou un de rechange ?), longueur du corps
- [ ] Tendeurs de courroie, serre-courroies : présents ou non

**Électronique**
- [ ] Carte de contrôle déjà disponible (SKR, Arduino, Raspberry Pi…) : référence
- [ ] Alimentations : 24 V ou 12 V, puissance
- [ ] Arduino Nano de la pendule : présent ou non
- [ ] Webcam du banc de vision : référence
- [ ] Fins de course, câbles moteurs, connecteurs : présents ou non
- [ ] Aimants, rondelles acier, capteurs à effet Hall : présents ou non

**Matériel de jeu**
- [x] Pièces : diamètres de base (consignés dans `mesures/pieces.md`)
- [ ] Pièces : **hauteur** et **masse** de chaque type, épaisseur du feutre éventuel
- [ ] Échiquier en vinyle : épaisseur, et dimensions exactes de la zone des cases

**Outillage**
- [ ] Pied à coulisse, multimètre, alimentation de labo réglable (tension et courant maximal)
- [ ] Dynamomètre ou peson : présent ou non (sinon, voir les courses)
- [ ] Filaments disponibles : PLA, PETG, ASA, TPU

## 3. Après l'inventaire

- [ ] Transmettre l'inventaire à Claude, qui met à jour la nomenclature, ajuste la liste des courses et lance l'étude des collisions (risque R-02) avec les diamètres réels des bases.
- [ ] Faire les courses du **lot 1** (`courses.md`).

## 4. Banc magnétique (jalon 2)

- [ ] Imprimer les supports du banc : support d'aimant et support de servo. Claude fournira les fichiers.
- [ ] Faire les manipulations : force latérale, effet sur une pièce voisine à 50 mm, comparaison des 3 matériaux de plateau, **décalage entre la position d'arrêt de l'aimant et celle de la pièce** (pour EX-14), et **essai de contact** : pousser une pièce contre une voisine pour voir si elle glisse ou se renverse (décision 013). Claude fournira la fiche de protocole et le tableau de relevé.
- [ ] En option : test d'un capteur à effet Hall sous une case (décision 012).
- [ ] Transmettre les résultats, pour trancher les décisions 002 (préhenseur), 003 (plateau) et 009 (pièces attirables par l'aimant).

## 5. Plus tard

- [ ] Dans Onshape : renommer les pièces génériques (« Part 1 », « Corps ») et donner un matériau aux pièces imprimées, puis relancer le workflow « Export Onshape ».
- [ ] Mesurer le taux de reconnaissance de la vision **par coup** sur une dizaine de parties (EX-20).
- [ ] Courses du lot 2, puis du lot 3, selon les résultats des tests.
