# Identifikácia stupňa kvality vody na základe metód exploratívnej a prediktívnej dátovej analýzy

Bakalárska práca zameraná na identifikáciu stupňa kvality vody prostredníctvom exploratívnej a prediktívnej dátovej analýzy. Práca sa zameriava na hlavné ukazovatele kvality vodných zdrojov a chemické a fyzikálne charakteristiky vody, ktoré na ne vplývajú.

## Analyzované datasety

- **Water Potability** (Kaggle) – 3 277 vzoriek pitnej vody, 9 chemických atribútov + binárny atribút *Potability*
- **Groundwater Quality Monitoring** (program GAMA, California) – viac ako 1 milión záznamov o kvalite podzemných vôd
- **GEMStat Water Quality Africa** (UNEP GEMS/Water) – 188 záznamov o fyzikálnych a geografických charakteristikách vodných útvarov v Afrike

## Obsah repozitára

| Súbor | Popis |
|---|---|
| `heatmaps.py` | Korelačná analýza a tepelné mapy pre všetky tri datasety |
| `linear_regression_loess.py` | Regresná analýza (lineárna regresia a LOESS) vzťahu chemických atribútov a pitnosti |
| `random_forest.py` | Klasifikácia pitnosti vody pomocou náhodného lesa |
| `decision_tree.py` | Klasifikácia pitnosti vody pomocou rozhodovacieho stromu |
| `bayesian_network.py` | Klasifikácia pitnosti vody pomocou Bayesovej siete |
| `lime_shap.py` | Viacvrstvová neurónová sieť (MLP) a interpretácia výsledkov pomocou metód SHAP a LIME |
| `dotaznik_llm_odpovede.docx` | Dotazník predložený štyrom veľkým jazykovým modelom (ChatGPT, Gemini, Claude, Copilot) v rámci diagnostickej analýzy výsledkov, vrátane ich odpovedí |

## Použité knižnice

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` (Keras), `shap`, `lime`, `pgmpy`

## Poznámka k dátam

Samotné datasety nie sú súčasťou repozitára z dôvodu ich veľkosti. Odkazy na ich voľne dostupné zdroje sú uvedené v texte práce (kapitola 2) a v zozname bibliografických odkazov.
