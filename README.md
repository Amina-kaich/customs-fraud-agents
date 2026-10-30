# Détection de la fraude douanière par système multi-agents

Projet 2026-2027 · ENIT, Ingénierie Télécommunications (Data & ML) · Sujet IA 06, catalogue DESIGNET (niveau Avancé)

Auteure : Amina Kaich

## Contexte

Les administrations douanières reçoivent plus de déclarations qu'elles ne peuvent en inspecter. Les fraudes courantes visent à réduire indûment les droits et taxes : sous-évaluation de la valeur, mauvais classement tarifaire (code SH), fausse origine, quantités ou poids incohérents. Les modèles d'apprentissage automatique actuels donnent un score, mais n'expliquent pas leur décision, ne vérifient pas la conformité réglementaire et ne croisent pas la déclaration avec des informations externes.

## Objectif

Concevoir et évaluer un prototype qui combine un modèle de scoring supervisé et un système multi-agents (CrewAI). Chaque agent a une tâche spécialisée, afin de produire des alertes explicables, appuyées par des éléments probants et soumises à la validation d'un agent humain.

Principe central : le système signale, l'humain décide. Aucune décision automatique de sanction ou de blocage.

## Types de fraude ciblés (noyau du projet)

- Sous-évaluation : valeur unitaire déclarée très inférieure à la référence du même code SH.
- Mauvais classement : description de la marchandise incompatible avec le code SH déclaré.
- Incohérences : poids, quantité et valeur discordants, origine incompatible avec le produit.

## Architecture multi-agents

| Agent | Rôle |
|---|---|
| Analyste de déclarations | Nettoyage, variables, score du modèle supervisé |
| Détection d'anomalies | Écarts de prix, poids ou quantités incohérents |
| Vérification réglementaire (RAG) | Cohérence code SH et description, règles applicables, citations des sources |
| Collecte d'informations externes | Prix de référence, informations publiques ciblées |
| Rédacteur de rapport | Synthèse de l'alerte, justification, recommandation d'inspection |

Flux : analyse, anomalies, vérification réglementaire et collecte externe, score agrégé, rapport, validation humaine.

## Données

Aucune donnée douanière réelle : uniquement des jeux synthétiques publiés et des statistiques publiques.

- Customs Import Declaration Datasets (Kim et al.) : jeu principal, déclarations synthétiques étiquetées.
- Jeu synthétique de l'OMD (modèle LITE DATE) : comparaison de référence.
- Référentiels : codes SH à 6 chiffres, UN Comtrade (valeurs unitaires de référence), code des douanes tunisien, TARIC UE, HTS américain, CBP CROSS.

## Évaluation

- Détection : précision, rappel, F1, PR-AUC, ROC-AUC, métriques au top-k (budget d'inspection), intervalles de confiance.
- Ablation : modèle seul, puis plus anomalies, plus vérification réglementaire, plus collecte externe.
- Qualité des agents : fidélité des citations, hallucinations, coût par déclaration, latence.
- Robustesse et sécurité : tests d'injection de prompts dans les contenus externes.

## Structure du dépôt

    data/         données (non versionnées : raw, processed, external)
    notebooks/    exploration et expériences (Google Colab)
    configs/      configuration des agents et des prompts
    docs/         fiche de données, rapport d'évaluation
    tests/        tests automatiques (pytest)
    src/data      chargement et nettoyage des données
    src/models    modèle de scoring supervisé
    src/anomalies détection d'anomalies
    src/rag       base de connaissances réglementaire
    src/agents    agents CrewAI
    src/api       backend FastAPI

## Démarrage

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Limites

Données synthétiques d'origine étrangère, descriptions de marchandises générées, couverture partielle des fraudes réelles, dépendance au modèle de langage.