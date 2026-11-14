# Liste des courses

Organisée par lots. On n'achète un lot que lorsque le précédent a donné ses résultats, pour ne pas payer du matériel qu'un test rendrait inutile (EX-35 : maîtrise des coûts).

- **Leroy Merlin** (magasin le plus proche) : pour les plaques, les panneaux et la quincaillerie volumineuse. Vérifiez le stock sur leroymerlin.fr en choisissant son magasin. La découpe des panneaux de bois se fait sur place.
- **En ligne** : pour les petits composants, faciles à envoyer.

Les prix marqués « constaté » ont été relevés sur leroymerlin.fr le 14/11/2026. Les autres sont des **ordres de grandeur à vérifier**.

**À ajuster après l'inventaire du fablab** (`actions.md`) : rayer ce qui est déjà disponible.

## Lot 1 – Dérisquer le préhenseur et le plateau (à acheter maintenant)

Il sert au banc magnétique du jalon 2 et répond aux risques R-01, R-05 et R-09, ainsi qu'aux décisions 002, 003, 009 et 012. Budget estimé : 50 à 80 €.

**À Leroy Merlin**

| Article | Qté | Pour quoi | Prix |
|---|---|---|---|
| Plaque Plexiglas PMMA transparent 3 mm, format A4 | 1 | Échantillon de plateau PMMA | 9,80 € (constaté) |
| Chute de MDF 3 mm (demander à la découpe), sinon panneau MDF 3,2 mm 244 × 122 cm | 1 | Échantillon de plateau MDF. Le panneau entier servira plus tard pour le plateau et le fond | 21,90 € le panneau (constaté) |
| Chute de composite aluminium 3 mm (à demander, sinon à reporter) | 1 | Échantillon de plateau rigide | à voir sur place |
| Aimants néodyme en disques (rayon quincaillerie) | 1 à 2 lots | Aimant de préhenseur (option 2) et essai d'aimant sous une pièce | ≈ 8 € le lot (constaté) |
| Rondelles plates en acier, M8 à M12 | 1 sachet | Essai « rondelle sous la pièce » | ≈ 3 € |
| Patins de feutre adhésifs | 1 sachet | Reproduire le dessous des pièces et de l'aimant | ≈ 3 € |

**En ligne**

| Article | Qté | Pour quoi | Prix indicatif |
|---|---|---|---|
| Électro-aimant de maintien 24 V, Ø 20-25 mm (par exemple RS Pro 785-8512, ou un modèle générique) | 1 | Option 1 du préhenseur | 5 à 25 € |
| Micro-servo MG90S (pignons métal) | 1 à 2 | Option 2 : aimant permanent levé par un servo | ≈ 5 € pièce |
| Assortiment de petits aimants néodyme en disques (Ø 6 à 12 mm, 2 à 3 mm d'épaisseur) | 1 lot | Aimants à coller sous les pièces (options 3 et capteurs Hall) | ≈ 10 € |
| Dynamomètre à ressort 0-10 N, ou peson numérique 0-5 kg | 1 | **Instrument clé** : mesurer la force de décrochage | 10 à 15 € |
| Capteurs à effet Hall linéaires SS49E | 1 lot de 5 à 10 | Option : test de détection d'un aimant à travers le plateau (012) | ≈ 3 € |

À imprimer sur la Bambu Lab, sans achat : support d'aimant et de servo pour le banc. Claude fournira les fichiers.

## Lot 2 – Monter le CoreXY nu (après le banc magnétique)

Il sert au jalon 3 et répond aux risques R-06 et R-07, ainsi qu'à la décision 004. Les quantités seront précisées après l'inventaire. Budget estimé : 40 à 80 €.

| Article | Où | Pour quoi | Prix indicatif |
|---|---|---|---|
| Poulies folles manquantes : dentées ou lisses, alésage selon l'inventaire, pour courroie de 9 mm | en ligne | Compléter les 8 poulies folles | ≈ 2-3 € pièce |
| Visserie M3 et M5 (vis, écrous, rondelles) pour les axes de poulies et les pièces imprimées | Leroy Merlin | Montage | ≈ 10 € |
| Écrous carrés M3 ou T-nuts pour MakerBeam (si l'inventaire en manque) | en ligne | Fixer les pièces au cadre | ≈ 10 € |
| Filament PETG (optionnel) | en ligne | Blocs moteurs (chaleur) ; supports de poulies si le PLA flue | ≈ 20 € la bobine |
| 2 micro-switchs de fin de course | en ligne | Repli si le homing sans capteur est peu fiable (R-07) | ≈ 3 € |
| Plateau final, découpé à 600 × 600 mm, dans le matériau retenu après le banc | Leroy Merlin (découpe) | Plateau supérieur | selon le matériau |

## Lot 3 – Électronique de pilotage (après le CoreXY nu)

Il sert au jalon 4 et à la décision 005, qui est encore au statut « proposée » : à confirmer avant l'achat. Budget estimé : 60 à 100 €.

| Article | Pour quoi | Prix indicatif |
|---|---|---|
| Carte BTT SKR Mini E3 V3 | Contrôleur des moteurs et du préhenseur | ≈ 40 € |
| Alimentation 24 V, 100 à 150 W (sauf si l'inventaire en trouve une) | Carte, moteurs, préhenseur | ≈ 20-30 € |
| Câble USB adapté à la carte, câbles moteurs (si absents) | Liaison avec le PC et câblage | ≈ 10 € |
| Diode 1N4007 (seulement si l'électro-aimant est retenu) | Protection de la sortie MOSFET | < 1 € |

## À ne pas acheter pour l'instant

- **Le préhenseur définitif** (électro-aimant ou servo en série) : à choisir d'après le banc magnétique.
- **32 aimants ou rondelles pour toutes les pièces** : à acheter seulement une fois la solution validée sur 2 ou 3 pièces.
- **Des capteurs Hall pour les 64 cases** : seulement si le test d'un capteur est concluant et que la décision 012 les retient.
- **Le grand plateau en PMMA** : environ 109 € pour une plaque de 3 mm en 200 × 100 cm chez Leroy Merlin, alors que le plateau n'a pas besoin d'être transparent.
