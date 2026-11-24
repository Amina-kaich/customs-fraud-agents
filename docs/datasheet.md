# Fiche de données : jeu BACUDA (synthetic_data2.csv)

## 1. Provenance
- Jeu synthétique du projet BACUDA de l'Organisation mondiale des douanes (OMD), diffusé dans le dépôt GitHub `YSCHOI-github/Customs_Fraud_Detection` (fichier `synthetic_data2.csv`).
- Données **synthétiques et anonymisées** (codes de bureaux, d'importateurs et de pays), à usage pédagogique.
- Licence : **non précisée** dans le dépôt. Source à citer dans le rapport ; usage à valider avec l'encadrant.
- Modèle de référence associé : DATE (Kim et al., KDD 2020).

## 2. Contenu
- 200 000 déclarations, 13 colonnes utiles (après suppression d'un index sans intérêt), 17,6 Mo.
- Cible : `illicit` (1 = fraude). **14 370 fraudes, soit 7,18 %.**
- Période : du 1er janvier 2013 au 31 décembre 2014.
- Aucun doublon, aucune valeur manquante.

## 3. Colonnes
| Colonne | Type | Valeurs distinctes | Remarque |
|---|---|---|---|
| year, month, day | entier | | 2 970 lignes (1,5 %) ont un jour impossible : on n'utilise que l'année et le mois |
| OFFICE | catégorie | 73 | 10 bureaux = 66 % des lignes |
| IMPORTER.TIN | catégorie | 15 348 | 22,1 % des importateurs n'apparaissent qu'une fois |
| TARIFF.CODE | texte | 6 333 | 1 974 codes avaient 9 chiffres (zéro initial perdu) : complétés à 10 chiffres |
| ORIGIN.CODE | catégorie | 130 | 10 pays = 72,7 % des lignes |
| CIF_USD_EQUIVALENT | numérique (USD) | | médiane 5 595, moyenne 57 959 |
| QUANTITY | numérique | | médiane 84 |
| GROSS.WEIGHT | numérique | | médiane 1 711 |
| TOTAL.TAXES.USD | numérique (USD) | | médiane 518 |
| RAISED_TAX_AMOUNT_USD | numérique | | **fuite de données, voir section 5** |
| illicit | binaire | 2 | cible |

Colonnes dérivées : `HS6`, `HS4`, `HS2` (6, 4 et 2 premiers chiffres du code tarifaire). Le chapitre SH 87 (véhicules) regroupe 38,8 % des lignes (77 601).

## 4. Qualité
- Asymétrie très forte des quatre variables numériques (de 15,7 à 47,9) : passage en logarithme.
- Valeurs planchers : 51 737 lignes (25,9 %) ont exactement 5 $ de taxes ; 24 538 (12,3 %) un poids de 1 ; 48 843 (24,4 %) une quantité de 1 ; 2 780 (1,4 %) 0 $ de taxes.
- 33 471 lignes (16,74 %) ont des taxes supérieures à la valeur CIF : anomalie du jeu synthétique, le taux effectif est plafonné à 1,5.

## 5. Fuite de données
`RAISED_TAX_AMOUNT_USD` est strictement positive pour **les 14 370 fraudes** et nulle pour les 185 630 autres (moyenne 749 $, médiane 635 $, maximum 5 700 $). C'est le supplément de taxe récupéré après inspection : elle contient la réponse. **Elle n'est jamais utilisée comme variable explicative.** Elle ne pourrait servir que comme second objectif (revenu récupéré).

## 6. Découpage temporel
Aucune partition n'est fournie. Découpage défini par année et mois :
| Partition | Période | Lignes | Fraudes | Taux |
|---|---|---|---|---|
| train | 2013 | 113 223 | 7 797 | 6,89 % |
| valid | janvier à juin 2014 | 46 348 | 3 366 | 7,26 % |
| test | juillet à décembre 2014 | 40 429 | 3 207 | 7,93 % |

Toute statistique (fréquences, médianes) est apprise sur le train uniquement. Le test n'est utilisé qu'une seule fois, à la fin.

## 7. Observations
- **Le taux de fraude varie fortement selon le bureau** (de 2,3 % à 33,1 %, modalités de plus de 300 lignes), **le pays d'origine** (de 1,6 % à 31,2 %) et **le chapitre SH** (de 0,4 % pour le chapitre 86 à 25,1 % pour le chapitre 25).
- **Effet de rareté** : plus une modalité est rare, plus son taux de fraude est élevé (corrélation de Spearman de -0,88 pour les bureaux, -0,64 pour les pays, -0,70 pour les chapitres SH, -0,66 pour les importateurs). Vérifié dans le futur : en validation, les importateurs inconnus du train ont 23,8 % de fraude, ceux de plus de 100 déclarations 1,7 %.
- Les fraudes portent sur des valeurs un peu plus faibles (CIF médian de 3 771 $ contre 5 805 $).
- Les valeurs planchers sont presque toujours non frauduleuses : 2,1 % de fraude pour 5 $ de taxes, contre 26,3 % pour 0 $ de taxes.
- Le volume mensuel varie de 5 028 à 14 614 déclarations et le taux de fraude de 5,0 % à 10,1 % ; leur corrélation est de -0,89, car le nombre de fraudes par mois varie peu (476 à 735).
- Dérive entre partitions faible (statistique KS au maximum de 0,108, sur les taxes). Entités du test inconnues du train : 5,0 % des importateurs, 1,3 % des codes tarifaires, 0,5 % des codes HS6.

## 8. Limites
- Données synthétiques : les relations observées (en particulier l'effet de rareté) peuvent refléter la génération du jeu plus que la réalité douanière.
- Aucune description de marchandise : l'enrichissement prévu au cahier des charges reste nécessaire pour l'agent RAG.
- Taux de fraude de 7 % propre à ce jeu, pas un taux réel du flux douanier.
- Licence non précisée.
- Jours impossibles et taxes supérieures à la valeur CIF : artefacts de génération.

## 9. Références
- DATE : Kim et al., *DATE: Dual Attentive Tree-aware Embedding for Customs Fraud Detection*, KDD 2020.
- Projet BACUDA : https://bacuda.wcoomd.org
- Dépôt des données : https://github.com/YSCHOI-github/Customs_Fraud_Detection