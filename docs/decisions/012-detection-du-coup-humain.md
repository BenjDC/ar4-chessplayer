# 012 – Détection du coup humain

- **Statut** : ouverte. En V1, on garde la vision existante, avec confirmation humaine en cas de doute.
- **Exigences liées** : EX-02, EX-20, EX-21, EX-22
- **Liée à** : 007 (vision), 009 (pièces attirables par l'aimant), 011 (caméra)

## Contexte
La première solution prototypée est la vision : caméra sur un portique au-dessus du plateau, tags ArUco, ResNet18 sur les 64 cases. Les premiers essais sont encourageants, mais le taux de reconnaissance tourne autour de 90 %, ce qui reste insuffisant pour jouer sans aide. D'autres pistes avaient été envisagées.

## Options étudiées
1. **Vision (existante)**
   - Pour : aucune modification du plateau, déjà prototypée, et marge de progrès connue (taille d'image en entrée, données d'entraînement, éclairage).
   - Contre : sensible à l'éclairage et aux reflets du plexiglas, et les pièces hautes masquent les cases voisines. Il faut aussi un support de caméra au-dessus du jeu (EX-22).
2. **Capteurs à effet Hall ou ILS (reed) sous chaque case**, qui détectent un aimant collé sous la pièce
   - Pour : fiable, insensible à la lumière, rapide. Avec des capteurs Hall linéaires ou bipolaires, **la polarité de l'aimant peut même indiquer la couleur** (nord en haut pour les blancs, sud pour les noirs). On obtient alors exactement l'occupation à 3 états qu'utilise déjà `detect_move_from_occupancy`.
   - Contre : 64 capteurs à câbler et multiplexer (matrice, registres à décalage ou expandeurs I2C), et un circuit imprimé ou un câblage sous l'échiquier. Ce dispositif s'intercale entre l'électro-aimant et les pièces, ce qui augmente l'entrefer. Et l'électro-aimant perturbe les capteurs pendant ses déplacements : il faut donc lire seulement à l'arrêt.
   - **Couplage fort avec 009** : la solution suppose des aimants permanents sous les pièces, pas de simples rondelles d'acier.
3. **Tag RFID ou NFC sous chaque pièce et plateau instrumenté**
   - Pour : identifie chaque pièce, pas seulement sa couleur. Le système reconnaît alors une position quelconque, une promotion, ou une pièce remise au mauvais endroit.
   - Contre : le plus complexe (64 antennes multiplexées ou lecteurs multiples). Le lest en plomb et l'électro-aimant risquent de perturber la lecture, et les antennes ajoutent de l'épaisseur entre l'aimant et les pièces.
4. **Saisie ou confirmation humaine** (clavier, appui sur la pendule)
   - Pour : immédiat et fiable.
   - Contre : contraire à l'objectif final d'EX-02. C'est un repli acceptable en V1.

## Démarche proposée
- **V1** : vision et confirmation humaine en cas de doute (EX-21). On corrige la dette connue de la vision (taille des images) et on mesure le taux réel.
- **En parallèle**, quand on tranchera 009 : tester un capteur Hall sous une case avec un aimant de pièce, à travers l'empilement réel (échiquier vinyle, plexiglas). Si le test est concluant et que 009 retient des aimants néodyme, l'option 2 devient la candidate naturelle pour la V2.
- RFID : à garder en réserve, si l'identification des pièces devient un besoin (V3).
