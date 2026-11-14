# Plan

Principe : avancer par incréments (V1, V2, V3, voir `exigences.md`). Valider physiquement chaque étape avant d'ajouter la suivante. L'intervention humaine est acceptée tant qu'elle évite une complexité prématurée.

Tableau de suivi (Whiteboard) : https://claude.ai/artifact/VLvAx6wNvFf2uxrUXrjHff

## Jalon 1 – Spécification et dépôt propre *(en cours)*
- [x] README : architecture logicielle et mécanique
- [x] Accès à la CAO Onshape (export REST + MCP FeatureScript)
- [x] `CLAUDE.md`, exigences, fiches de décision, nomenclature, plan
- [x] Retrait du vocabulaire « bras AR4 »
- [x] Exigences revues : démarche incrémentale V1 / V2 / V3
- [ ] Valider les exigences encore marquées « à valider »

## Jalon 2 – Test magnétique
- [ ] Trancher la décision 009 (pièces attirables par l'aimant)
- [ ] Électro-aimant tenu à la main sous le plexiglas, sur 3 ou 4 cases avec de vraies pièces
- [ ] Mesurer l'entrefer, la force minimale de glissement et l'effet sur les cases voisines à 50 mm
- [ ] Choisir l'électro-aimant et son réglage de PWM

## Jalon 3 – CoreXY nu
- [x] Cadre MakerBeam assemblé, rails Y posés (photo du 14/11)
- [x] Moteurs, courroie 2GT 9 mm et poulies GT2 en stock
- [ ] Poser le rail X sur son profilé en pont
- [ ] Modéliser et imprimer les pièces manquantes : blocs moteurs, supports de poulies en sandwich, gantry blocks, chariot central
- [ ] Monter le châssis, les rails, les courroies et les poulies
- [ ] Prise d'origine et répétabilité sur les 64 cases (EX-14)

## Jalon 4 – Firmware et pilotage
- [ ] Configurer et flasher Marlin (CoreXY, sans chauffe, sortie de l'aimant)
- [ ] Module Python de pilotage G-code (`ok`, erreurs, `G28`, `M106`)
- [ ] Planificateur de trajectoires : cases → mm, gouttières, roque, prises vers le parc (décision 010)

## Jalon 5 – Intégration
- [ ] Brancher le pilotage dans `chessplayer.py`
- [ ] Corriger la dette de la vision et mesurer EX-20 (≈ 90 % aujourd'hui)
- [ ] Décision 012 : tester un capteur Hall sous une case (si 009 retient des aimants)
- [ ] Monter la caméra (décision 011)
- [ ] Option : pendule instrumentée
