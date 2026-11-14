# 008 – Accès à la CAO Onshape : export REST et MCP FeatureScript

- **Statut** : décidée
- **Date** : 2026-11-13

## Contexte
La CAO principale est sur Onshape. L'IA doit pouvoir lire le modèle et proposer des pièces, sans que les clés API passent par une conversation.

## Options étudiées
1. **MCP FeatureScript officiel (Onshape Labs)** : écrit et teste des features paramétriques. Il ne lit pas les Part Studios ni les assemblages.
2. **Serveur MCP communautaire sur l'API REST** : accès large, mais il tourne en local avec les clés, et c'est un projet tiers.
3. **Export REST par GitHub Actions** : les clés restent dans les secrets GitHub et le résultat est versionné dans le dépôt.

## Décision
Options 1 et 3 combinées : MCP FeatureScript pour l'écriture, export REST par GitHub Actions pour la lecture (`tools/onshape_export.py` → `mecanique/onshape/`). L'option 2 reste possible plus tard, dans Claude Code en local.

## Conséquences
- Quota partagé de 2 500 appels par an. Un export coûte environ 7 appels aujourd'hui.
- Pour des exports lisibles : nommer les pièces et leur donner un matériau dans Onshape.
