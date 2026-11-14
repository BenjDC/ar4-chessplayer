# Mécanique – plateau CoreXY

Retour au [README principal](../README.md) · Nomenclature : [docs/nomenclature.csv](../docs/nomenclature.csv) · Risques : [docs/risques.md](../docs/risques.md)

Statuts, comme dans les [fiches de décision](../docs/decisions/README.md) : **décidé** (acheté ou monté), **proposé** (hypothèse à confirmer), **ouvert** (à choisir).

## Châssis – *décidé* ([003](../docs/decisions/003-chassis-makerbeam-et-rails-mgn9.md))
- Caisson de 600 × 600 × 100 mm, ossature en profilés aluminium **MakerBeam 10 × 10 mm**. Le cadre est assemblé.
- Échiquier taille 5 (400 × 400 mm, cases de 50 mm), centré, avec une marge de 100 mm sur le pourtour pour la mécanique.
- Plaque de fond rigide (MDF ou contreplaqué) qui raidit l'ensemble et porte les moteurs.
- **Plateau supérieur : ouvert.** Il n'a pas besoin d'être transparent. Trois candidats : PMMA de 3 mm, MDF de 3 mm, composite aluminium de 3 mm. Les critères sont un entrefer minimal, une flèche faible sur 600 mm, le coût, et la disponibilité chez Leroy Merlin (EX-35).

## Guidage linéaire – *décidé* ([003](../docs/decisions/003-chassis-makerbeam-et-rails-mgn9.md))
- 3 rails **MGN9H** (9 mm, quasiment au ras des profilés de 10 mm).
- **Axe Y** : 2 rails parallèles sur les profilés latéraux inférieurs, déjà posés.
- **Axe X** : 1 rail sur un profilé MakerBeam en pont entre les deux chariots Y, fixé par des *gantry blocks* imprimés en 3D. Il porte le chariot central. Ce rail n'est pas encore posé.

## Cinématique et transmission – CoreXY ([004](../docs/decisions/004-cinematique-corexy-courroies-2gt.md))
- Deux boucles de courroie indépendantes. Les moteurs restent fixes à l'arrière, ce qui garde l'axe X mobile léger.
- 2 moteurs **NEMA 17** Motech MT-1704HS168A (1,8°/pas, 1,68 A, axe 5 mm), en stock.
- Courroie **2GT renforcée fibre de verre, 9 mm** (Trianglelab, rouleau de 8 m), en stock.
- 10 poulies (hypothèse à confirmer selon le stock) :
  - 2 motrices Trianglelab GT2 de 20 dents (alésage 5 mm) ;
  - 8 folles à roulements, en paires superposées, aux coins avant et sur les chariots Y.
- Les deux boucles circulent sur deux étages décalés d'environ 11-12 mm, avec des supports de poulies « en sandwich ». À dimensionner selon les poulies folles réelles.
- Résolution théorique : 0,2 mm par pas complet, 0,0125 mm en 1/16 de micropas.

## Préhenseur – *ouvert* ([002](../docs/decisions/002-electro-aimant-fixe-sans-axe-z.md), [009](../docs/decisions/009-pieces-attirables-par-aimant.md))
Trois options, à départager au banc magnétique du jalon 2 :
1. Électro-aimant fixe de 24 V sous le plateau (proposition initiale).
2. Aimant néodyme permanent levé par un micro-servo.
3. Aimant néodyme collé sous chaque pièce, avec l'une ou l'autre des options ci-dessus.

Les pièces sont lestées au plomb, donc non magnétiques. Il faudra y ajouter de l'acier ou un aimant.

## Déplacement des pièces
Le principe initial est de faire passer les pièces le long des lignes entre les cases. Ce n'est **pas suffisant** avec des pièces de taille 5, dont les bases font 30 à 40 mm pour 25 mm de dégagement (voir le [risque R-02](../docs/risques.md)). Une étude géométrique et un planificateur de trajets sont à prévoir.

## CAO
- **Onshape** (référence) : document « Caisson ». L'export automatique, en lecture seule, est dans [`onshape/RESUME.md`](onshape/RESUME.md) ; on le relance par le workflow GitHub « Export Onshape ».
- **FreeCAD** (anciens fichiers) : `caisson.FCStd`, `Joint_rail.FCStd`, `../equerre_rail.FCStd`, et `pieces/` (rail, chariot, profilés MakerBeam 100 et 600, équerre ; STEP pour les profilés).

## Pièces imprimées à concevoir
Imprimées au fablab sur la Bambu Lab P2S. On utilise du PETG pour les pièces sous tension de courroie (supports de poulies, blocs moteurs, gantry blocks), parce que le PLA flue sous charge permanente.

Blocs moteurs, supports de poulies en sandwich aux coins, gantry blocks, chariot central avec support du préhenseur. Elles seront conçues par features paramétriques dans Onshape (MCP FeatureScript), une fois les poulies mesurées et le préhenseur choisi.
