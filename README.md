# QPP-Fusion
- [RRF & WRRF](#RRF)
- [CombMNZ & WCombMNZ](#CombMNZ)
- [CombSUM & WCombSUM](#CombSUM)
- [Oracle](#Oracle)
- [ProbFuse](#ProbFuse)

## RRF
```
python3 QPPrrf_argmax.py --res_path data/RL/2019 --qpp_path data/predictors/2019 --qrels data/qrels/2019.qrels --topics data/qrels/2019.queries --strategy adaptive
```
## CombMNZ
```
python3 CombMNZ.py --res_path data/RL/2019 --qpp_path data/predictors/2019 --qrels data/qrels/2019.qrels --topics data/qrels/2019.queries --strategy combmnz
```

## CombSUM
```
python3 Combsum.py --res_path data/RL/2019 --qpp_path data/predictors/2019 --qrels data/qrels/2019.qrels --topics data/qrels/2019.queries --strategy CombSUM
```

## Oracle
```
python3 Combsum.py --res_path data/RL/2019 --qrels data/qrels/2019.qrels --topics data/qrels/2019.queries
```

## ProbFuse
```
python3 ProbFuse.py --train_res_path data/RL/2019 --test_res_path data/RL/2020 --train_qrels data/qrels/2019.qrels --test_qrels data/qrels/2020.qrels --topics data/qrels/2020.queries --variant judged --x 50 --L 100 --eps 1e-6
```
