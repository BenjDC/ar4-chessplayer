# 006 – Firmware Marlin et pilotage G-code depuis un hôte Python

- **Statut** : décidée
- **Date** : 2026-11-13
- **Exigences liées** : EX-10, EX-15, EX-33

## Décision
- Marlin, compilé avec PlatformIO et flashé par carte microSD, avec `#define COREXY`.
- Le programme Python (PC ou Raspberry Pi) transforme un coup UCI en trajectoire par les lignes entre les cases, puis en G-code.
- Liaison USB série à 115200 bauds, avec le protocole « attendre `ok` » : chaque commande attend la réponse `ok` de Marlin.

## Conséquences
- Marlin doit être configuré pour une machine sans chauffe : `EXTRUDERS 0`, sondes de température et protections thermiques désactivées, sortie de l'aimant non liée au refroidissement de buse.
- À écrire côté Python : un module de pilotage (connexion, `G28`, `G0`/`G1`, `M106`/`M107`, attente de `ok`, gestion des erreurs) et un planificateur de trajectoires (cases → millimètres, passage par les gouttières, roque, prise vers la zone de rangement).
