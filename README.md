# ar4-chessplayer
Une joueur d'échecs robotique à partir du bras AR4.

## Architecture : 

              ┌───────┐             
              │       │             
              │ Chess │             
              │       │             
              └─▲───┬─┘             
                │   │               
┌────────┐    ┌─┴───▼─┐    ┌───────┐
│        ◄────┤       │    │       │
│ Vision │    │ Main  ├────►  Arm  │
│        ├────►       │    │       │
└────────┘    └─▲───┬─┘    └───────┘
                │   │               
              ┌─┴───▼─┐             
              │       │             
              │ Clock │             
              │       │             
              └───────┘             

Le code **main* orchestre le joueur d'échecs sur les fonctions suivantes : 

### Vision : 

Prend une photo du plateau, fait les opérations nécessaires pour détecter et discriminer les 64 cases et fournit les fonctions nécéssaires pour comparer deux photos successives et en déduire le coup joué. 

#### Chess : 

à partir d'une position initiale, et de coups joués par un humain reportés par **vision**, propose des coup en utilisant stockfish. Maintien l'état de la partie. 

Clock : 

Code embarqué dans une pendule d'échecs instrumentée, indique à **main** via port série qu'un coup a été  joué. 

Arm : 

reçoit un coup au format UCI et le joue avec le bras AR4. Une fois le coup joué, le bras actionne la pendule.

## Matériel

* Plateau d'échecs : https://damieropera.com/produit/echiquier-en-vinyle-souple-roulable-4/ (vert)
* Pièces : https://damieropera.com/produit/pieces-en-resine-plombees-taille-5-3/
* Horloge : https://www.amazon.fr/RNPKZ-Professionnelle-chronomètre-pourMaison-etTournois/dp/B0FDQJTFJ2/ (bouton large pour que le robot l'utilise). L'horloge est instrumentée pour renvoyer par port série un indicateur lorsqu'un coup est joue (utiliser un arduino nano)

## libraires