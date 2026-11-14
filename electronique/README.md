# Électronique et firmware

Retour au [README principal](../README.md) · Décisions : [005](../docs/decisions/005-electronique-skr-mini-e3-v3.md), [006](../docs/decisions/006-firmware-marlin-et-hote-gcode.md)

> **Statut : proposé.** Cette architecture vient d'une première exploration. Rien n'est encore acheté ni testé. Ce dossier accueillera la configuration du firmware (`Configuration.h`, `Configuration_adv.h`) et les schémas de câblage.

## Architecture proposée

| Élément | Choix proposé | Rôle |
|---|---|---|
| Carte | BTT SKR Mini E3 V3 (STM32) | Contrôleur bas niveau, avec drivers et sorties de puissance intégrés |
| Drivers | TMC2209 intégrés (2 sur 4 utilisés) | StealthChop2 pour le silence, StallGuard pour le homing sans capteur |
| Alimentation | Bloc 24 V unique | Carte, moteurs, préhenseur |
| Préhenseur | Sortie MOSFET (FAN ou HE0) en PWM, avec diode de roue libre 1N4007 ; ou servo sur le connecteur de sonde, selon le préhenseur retenu | Saisir et relâcher les pièces |
| Firmware | Marlin, `#define COREXY`, compilé avec PlatformIO | Cinématique, accélérations, G-code |
| Liaison hôte | USB série à 115200 bauds, protocole « attendre `ok` » | Le programme Python envoie le G-code |

## Points de vigilance
- Marlin s'attend à piloter une imprimante 3D : il faut le configurer sans extrudeur ni chauffe (`EXTRUDERS 0`, sondes et protections thermiques désactivées), et vérifier que la sortie du préhenseur n'est pas liée au refroidissement de buse.
- Le homing sans capteur est plus délicat en CoreXY : prévoir l'emplacement de deux micro-switchs de secours (risque R-07).
- Le courant du préhenseur détermine la sortie à utiliser : une sortie FAN convient en dessous de 1 A environ.

## Pendule instrumentée (en option)
Un Arduino Nano détecte l'appui sur la pendule et envoie `clic` sur son port série, à 9600 bauds. Le code côté PC est prêt (`wait_for_click` dans `chessplayer.py`). Le firmware Arduino n'est pas encore dans le dépôt : il viendra dans ce dossier.
