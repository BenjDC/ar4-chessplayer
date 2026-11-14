# 005 – Électronique : BTT SKR Mini E3 V3, TMC2209, 24 V

- **Statut** : proposée (issue d'une exploration avec une IA, à confirmer)
- **Date** : 2026-11-13
- **Exigences liées** : EX-16, EX-32, EX-33, EX-34

## Options étudiées
1. **BTT SKR Mini E3 V3 (STM32)** : tout-en-un, drivers TMC2209 intégrés, sorties MOSFET, fonctionne en autonomie.
2. **BTT SKR Pico (RP2040)** : compacte, mais pensée pour Klipper, avec un hôte permanent.
3. **Teensy et drivers séparés** : souple, mais beaucoup de câblage et de firmware à écrire.

## Décision
- SKR Mini E3 V3, avec 2 de ses 4 TMC2209 utilisés (X et Y), en StealthChop2 pour le silence.
- Prise d'origine sans capteur (StallGuard).
- Bloc d'alimentation 24 V unique pour la carte, les moteurs et l'aimant.
- Électro-aimant sur une sortie MOSFET (FAN ou HE0), avec une diode de roue libre 1N4007 en parallèle inverse, et un PWM réglé par `M106`.

## Conséquences et points de vigilance
- Le homing sans capteur est plus délicat en CoreXY, où les deux moteurs tournent pour chaque axe. Il faut prévoir l'emplacement de deux micro-switchs en secours.
- Vérifier le courant réel de l'aimant. Une sortie FAN convient en dessous de 1 A environ, sinon utiliser HE0.
