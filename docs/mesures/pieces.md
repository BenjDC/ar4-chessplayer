# Mesures des pièces d'échecs

Pièces Damier Opéra en résine plombée, taille 5, sur un échiquier à cases de 50 mm. Diamètres de base mesurés au pied à coulisse.

| Pièce | Ø de base (mm) | Rayon (mm) | Hauteur (mm) | Masse (g) | Feutre (mm) |
|---|---|---|---|---|---|
| Roi | 38,8 | 19,4 | à mesurer | à mesurer | à mesurer |
| Dame | 36,2 | 18,1 | à mesurer | à mesurer | à mesurer |
| Cavalier | 32,8 | 16,4 | à mesurer | à mesurer | à mesurer |
| Fou | 31,0 | 15,5 | à mesurer | à mesurer | à mesurer |
| Tour | 31,0 | 15,5 | à mesurer | à mesurer | à mesurer |
| Pion | 27,5 | 13,75 | à mesurer | à mesurer | à mesurer |

## Dégagements entre une pièce en mouvement et une pièce immobile

Le dégagement est la distance libre entre les bords des deux bases. Il vaut la distance entre leurs centres moins la somme des deux rayons. Une valeur négative signifie que les bases se touchent ou se chevauchent. On suppose ici les pièces parfaitement centrées : toute erreur de positionnement se retranche du dégagement.

Trois trajets types, avec la distance entre le centre de la pièce en mouvement et celui de la pièce voisine :
- **le long d'une ligne entre deux cases** (« gouttière ») : 25 mm de la case voisine ;
- **par un coin**, à l'intersection de quatre cases : 35,4 mm des cases en diagonale ;
- **par le centre des cases** : 50 mm des cases voisines.

| Pièce en mouvement / voisine | Gouttière (25 mm) | Coin (35,4 mm) | Centre de case (50 mm) |
|---|---|---|---|
| Pion / Pion | **−2,5** | +7,9 | +22,5 |
| Pion / Tour | **−4,2** | +6,1 | +20,8 |
| Cavalier / Pion | **−5,1** | +5,2 | +19,9 |
| Cavalier / Tour | **−6,9** | +3,5 | +18,1 |
| Dame / Pion | **−6,9** | +3,5 | +18,1 |
| Roi / Pion | **−8,1** | +2,2 | +16,9 |
| Dame / Tour | **−8,6** | +1,8 | +16,4 |
| Roi / Tour | **−9,9** | +0,5 | +15,1 |
| Cavalier / Roi | **−10,8** | **−0,4** | +14,2 |
| Dame / Roi | **−12,5** | **−2,1** | +12,5 |
| Roi / Roi | **−13,8** | **−3,4** | +11,2 |

## Ce qu'on en tire

1. **Une pièce ne peut jamais passer dans une gouttière entre deux cases occupées** sans toucher une voisine, même entre deux pions. L'idée initiale de faire passer les pièces par les gouttières ne fonctionne pas avec ces pièces.
2. **Passer par un coin est possible**, mais avec seulement 0,5 à 8 mm de marge, et il y a contact quand un roi ou une dame est concerné. La précision de positionnement doit donc être nettement meilleure que ±3 mm sur ces passages : il faut viser environ ±1 mm (EX-14).
3. **Passer par le centre de cases libres** laisse au moins 11 mm de marge.
4. Conséquence immédiate : un cavalier ne peut pas sortir de sa position initiale sans frôler ou pousser un pion de la deuxième rangée.

Ces constats sont repris dans le risque R-02 et la décision 013.
