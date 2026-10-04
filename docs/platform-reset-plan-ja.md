# 回路解析・プラズマ等価回路・最適化・機械学習の再設計計画

更新日: 2026-10-04
状態: 回路計算・最適化・端子同定基盤は完成（R0〜R5完了）。次はR6のMLパッケージ比較準備だが、導入gateは未成立

## 1. 結論

現行の機械学習実装は、今後の製品機能の前提にしない。`ml-prepare`、`ml-evaluate`、
`ml-corpus`、`ml-corpus-evaluate`、現在のNumPy MLP/GNN、固定候補ランキングをいったん
公開基盤から撤去した。実験結果は「このデータと問題設定では有用性を示せなかった」という履歴だけを
残し、コードやスキーマを互換維持しない。

理由は実装品質が低いからではない。Ruff、Pyrefly、Import Linterは通過し、依存方向も守られている。
しかし、次の製品価値を一つも実証していない。

- 未評価候補を提案してngspice評価数を減らすこと
- 新しいチャンバー、運転条件、回路トポロジへ一般化すること
- 測定波形からプラズマ等価回路を一意に同定すること
- 学習結果を利用して工学制約を満たす回路を選ぶこと

現在のML関連実装は11ファイル、3,235行、専用テストは5ファイル、1,277行である。一方、108件の
ngspice ACデータによる比較では定数予測がridge、MLP、GNNより良く、固定候補ランキングも事前に定めた
評価数30%削減基準を満たさなかった。これは「さらにモデルを複雑にする」根拠ではなく、問題定義と
データ設計からやり直す根拠である。

今後の開発順序を次に固定する。

1. ネットリストまたは明示回路をngspiceで計算し、波形と回路指標を返す。
2. 装置で固定される量、運転時に変更できる量、測定から推定する量を区別する。
3. 測定波形からプラズマ端子モデルを同定する。
4. 同定済みモデルを使い、実際に変更できる回路素子だけを最適化する。
5. 検証済み回路テンプレートを個別に最適化し、結果を比較してトポロジを選ぶ。
6. 上記の反復計算が十分に蓄積され、学習による削減効果を測れる場合だけMLを再導入する。

この順序なら、MLが再び不合格でも回路解析、同定、最適化の基盤は完成したまま残る。

## 2. 本コード基盤の目的

PCDの目的は、半導体製造装置のRF電気系について、装置の電気的境界を明示したうえで次を行うことである。

- ngspiceによるAC、過渡、外部ネットリスト計算
- 電圧、電流、複素インピーダンス、反射、電力、素子ストレス、波形誤差の算出
- 測定または外部モデルで与えられたプラズマ端子特性の回路への接続
- 測定波形に整合する低次元の等価回路パラメータ同定
- 設置可能な素子値、チューナ設定、検証済み回路構成の最適化
- 保存済み結果から、判断に必要な表と図を生成

PCD単体の対象外は次である。

- プラズマ化学、粒子輸送、電子温度、密度、シース形状の自己整合計算
- エッチング速度、均一性、選択比、イオンエネルギー分布の直接予測
- チャンバーの3次元電磁界、熱、機械配置の最適化
- 未検証の任意ネットリストを生成する機械学習

これらの量は、測定値または別の物理モデルの出力としてPCDへ渡すことはできる。ただし、その場合も
PCDが証明するのは端子回路の応答であり、プラズマ物理やプロセス性能そのものではない。

## 3. 変更できる量と変更できない量

二周波RFと下部DCパルスを持つCCP装置を想定し、全ての量を次の役割のどれか一つへ割り当てる。
同じ値を「設計変数」と「プラズマ推定値」の両方にしない。

| 役割 | 代表例 | PCDでの扱い | 最適化で変更 |
|---|---|---|---|
| 固定ハードウェア | 電極面積・間隔、チャンバー壁、feedthrough、ケーブル、設置済みfilter/choke、測定面 | 装置IDと校正値として固定 | 不可。変更時は別装置または別構成 |
| 校正パラメータ | ケーブル遅延、壁・ESC・feedthroughの寄生C/L/R、VI probe補正、基準面変換 | プラズマ点火前または既知負荷で先に同定し固定 | 通常不可 |
| 運転条件 | 上下RFの周波数・位相・電力、DC pulse幅・周期・立上り、圧力、ガス、温度 | scenario/recipeとして入力し、条件間比較に使う | 装置許容範囲でのみ可。回路最適化とは別目的 |
| 回路設計変数 | matching C/L、可変コンデンサ位置、許可されたtap、switch状態、交換可能素子 | 明示した範囲・離散候補で探索 | 可 |
| 潜在プラズマ量 | 上下シース容量、シース損失、bulk R/L、時間依存端子インピーダンス | 測定から同定する未知量 | 設計最適化では不可。同定問題でのみ更新 |
| 観測量 | matcher/chamber面のV/I、反射波、wafer/chuck電圧、電極電流 | 目的波形、同定データ、検証用holdout | 不可 |
| 外部プロセス量 | 密度、電子温度、IED、エッチング速度・均一性 | 外部モデルまたは測定のラベル | PCD単体では不可 |

重要な区別は次の三点である。

1. 圧力やガス流量は操作できても、PCDの回路式だけではそれがシース容量やbulk抵抗へどう影響するかを
   計算できない。対応する測定または外部プラズマモデルがない限り、単なるscenarioラベルである。
2. `C_sheath(t)` や `R_bulk(t)` は装置が直接設定できるノブではない。測定波形を説明するために推定する量である。
3. 寄生素子とプラズマ素子を一つの波形から同時に自由推定すると、異なる組合せが同じ端子応答を作り得る。
   寄生素子を別測定で固定し、複数のV/I、周波数、運転条件を使ってプラズマ量を推定する。

## 4. プラズマ等価回路の対応範囲

### 4.1 四つのモデル段階

| 段階 | 入力 | 計算 | 主な用途 |
|---|---|---|---|
| E0 測定端子モデル | 基準面付き `Z(f)` または複素V/I | 測定値を一ポートとして接続 | 実装・検証の基準。最も主張が小さい |
| E1 静的等価回路 | `R_bulk`, `L_bulk`, 上下`C_sheath`、必要な損失 | 一条件のAC/周期定常回路 | 整合、反射、素子stress |
| E2 準静的時系列 | `Z(f,t)` または低次元係数から得る時刻別R/L/C | 各時刻を独立したsnapshotとして解く | pulsed plasmaの時間分解インピーダンス |
| E3 動的端子モデル | 電荷・磁束・内部状態を持つ構成式 | 外部モデルと回路を時間積分 | 非線形シースや状態遷移。別adapterとして実装 |

E2は「各時刻のAC回路」であり、過去状態やエネルギー保存を表さない。E3を実装せずに、ngspiceの
コンデンサ値やインダクタ値へ任意の `C(t)`、`L(t)` を代入して真の過渡プラズマを表したとは扱わない。
電荷 `q(v,state)` または磁束 `phi(i,state)` の構成式とエネルギー規約が必要である。

### 4.2 同定する時間変化の表現

時系列の各サンプルを独立変数にしない。未知波形は次の少数係数で表す。

- pulseのoff、立上り、定常、立下り、afterglowを表す固定節点の値
- 正値を保証する対数パラメータ
- 必要な場合だけ、節点間の単調または滑らかな補間
- 物理的に必要な最小・最大値と時定数の範囲

例えば `R_bulk(t)`、`L_bulk(t)`、`C_upper(t)`、`C_lower(t)` を全て自由な波形にせず、最初は
「off値は校正済み、on値4個と共通立上り時定数1個」のように5次元程度から始める。パラメータを増やすのは、
残差に再現性のある構造が残り、追加観測によって新しいパラメータを識別できる場合だけである。

## 5. 解く問題を分離する

### 5.1 順計算 `simulate`

入力:

- 回路または外部netlist
- 固定ハードウェア・校正値
- 運転条件
- 既知のプラズマ端子モデル
- 解析条件と観測点

出力:

- canonicalな時間波形または複素周波数応答
- solver成否と最低限の再現情報

順計算は推定、最適化、MLを知らない。全ての上位機能は同じ順計算を呼ぶ。

### 5.2 計測 `analyze`

入力はcanonical結果だけである。インピーダンス、反射、電力、調波、波形誤差、素子stressを計算する。
合否閾値や候補探索を持たず、同じ入力には同じ値を返す純粋処理にする。

### 5.3 プラズマ端子同定 `identify`

目的は「測定波形を説明する潜在等価回路パラメータ」を求めることであり、装置設計ではない。

入力:

- 校正済み固定回路
- 一つ以上の運転条件
- 電圧・電流波形と測定面
- 推定対象、固定値、範囲、時間基底

出力:

- 同定パラメータと再構成波形
- 学習に使わなかった観測点・条件での誤差
- 多点初期化の結果、感度、識別困難なパラメータ

最小二乗値だけを成功条件にしない。異なる初期値から同じ範囲へ収束し、未使用の波形または運転条件を
再現し、パラメータを変えたときに出力が十分変わることを確認する。

### 5.4 回路設計最適化 `optimize`

目的は「実際に変更できる回路変数」を選ぶことである。

入力:

- 一つの検証済み回路トポロジ
- design variableの範囲
- 複数scenarioの同定済みプラズマモデル
- 目的、制約、solver予算

出力:

- 最良候補と全scenarioのngspice再評価
- 目的履歴、制約余裕、素子値、波形
- 探索済み範囲と未探索範囲

推定対象のプラズマ値を設計変数へ混ぜない。最適候補は必ず通常のngspice経路で独立に再計算する。

### 5.5 トポロジ選択 `select-topology`

最初の実装ではトポロジを学習生成しない。L、pi、高調波filter、bias teeなど、エンジニアが接続と許容素子を
確認したテンプレートcatalogを作る。各テンプレートを同じscenario、目的、制約、予算で個別最適化し、最終結果を
比較する。候補数が小さい間は、この方法が最も説明可能で再現可能である。

任意graphのedge追加・削除、極性変更、groundの自動生成は対象外とする。物理的に組めない回路や危険な共振を
大量に生成する問題を、MLの妥当性問題と同時に持ち込まない。

## 6. 手法の選択

### 6.1 同定

最初の標準手法は、物理範囲付きの非線形最小二乗と多点初期化でよい。非滑らかまたは局所解が強い場合は、
既存のDifferential Evolutionを初期探索に使い、その後局所最小二乗で仕上げる。GPや深層学習を使わなくても、
同定対象が低次元ならこの方が結果を解釈しやすい。

識別性は、少なくとも次で確認する。

- パラメータ感度または有限差分Jacobian
- 複数初期値の解の広がり
- profile lossまたは各パラメータ固定時のloss増加
- 未使用観測・未使用条件の再現

### 6.2 固定トポロジの素子値最適化

| 問題 | 第一選択 | 理由 |
|---|---|---|
| 小さい離散候補 | 完全列挙 | 最適性とcoverageが明確 |
| 低次元連続、ngspiceが高価 | 制約付きGP Bayesian optimization | 少ない評価で探索と活用を両立しやすい |
| 非滑らか・中次元・評価予算が比較的大きい | Differential EvolutionまたはCMA-ES | GP仮定に依存せず頑健なbaselineになる |
| 離散トポロジ数が少ない | トポロジごとに上記を実行 | 混合探索より結果を説明しやすい |
| 多数のカテゴリと連続値 | 実証後に階層型またはmixed-variable BO | カテゴリと連続値を同じ距離で扱わない |

手法名を増やすこと自体を目的にしない。問題ごとに一つの単純baselineと一つの候補手法を、同じsolver予算、
複数seed、同じ制約で比較する。波形一致だけでなく、最終候補の工学制約と独立再計算を合否に使う。

### 6.3 機械学習を再導入する条件

GNNは回路トポロジが複数あり、トポロジ間で共有できるデータが十分ある場合の候補である。現行の3トポロジ、
108 sampleはこれを実証する規模ではない。既存研究でも、回路をdevice/net graphにし、数千から一万規模の
simulation instanceを使う例が多い。必要量は固定件数では決めず、学習曲線と独立holdoutで判断する。

再導入時の用途は、まずforward surrogateだけに限定する。

- 入力: 検証済みトポロジgraph、素子値、運転条件、プラズマ端子条件
- 出力: 複素port応答、調波、または低次元化した波形
- 使用法: 次にngspiceへ渡す候補のshortlistまたはwarm start
- 禁止: surrogateだけによる最終合否、直接の自由graph生成、非一意な逆写像の単発予測

graphはdevice nodeとnet nodeを持つ二部heterogeneous graphとし、端子種別、port role、素子種別、値、損失を
表す。現在の `circuit_graph.v1` は二端子p/n、三port、構造化builderだけに限定され、外部netlist、source、
磁気結合、時間依存モデルを表せないため、将来スキーマとして温存しない。必要なデータが揃った時点で、対象回路を
先に固定してv2を設計する。

モデル比較の順序は次とする。

1. training meanと物理式
2. topology別ridge/GP/小型MLP
3. topologyをまたぐGNN
4. 必要性が確認された場合だけ時間・周波数decoder

評価分割は行単位にせず、装置、測定campaign、トポロジ、recipeのgroup単位にする。RMSEだけでなく、候補選択時の
regret、制約違反率、ngspice呼出し削減、最終候補の再検証成否を測る。単純baselineに勝てない場合はMLを製品へ入れない。

データgate通過後の第一候補は、次の小型forward modelである。

- encoder: component nodeとnet nodeをtyped terminal edgeで結ぶheterogeneous message-passing GNN
- condition: 周波数、source設定、recipe、既知または同定済みの端子条件を別vectorとして与える
- decoder: source/load/wafer面の複素V/Iと必要な調波、または検証済み波形基底係数を予測する
- uncertainty: 同一構造のseed ensembleで予測分散を出し、学習範囲外ではngspice評価を優先する
- optimization接続: 平均予測だけで決定せず、予測値と不確かさからshortlistを作り、通常evaluationへ戻す

直接「目標波形から素子値」を一回で出すinverse networkは採用しない。同じ波形を作る複数回路があり得るため、
学習するのは順方向応答とし、逆設計は制約付き探索として解く。

## 7. 目標アーキテクチャ

```text
YAML / netlist / measurement files
                |
                v
          resolve input once
                |
                v
    CircuitProblem (typed, role-explicit)
       |             |               |
       |             |               +----> identify ----> fitted terminal model
       |             |
       |             +--------------------> optimize -----> candidate
       |                                      |                |
       v                                      v                |
    simulate ---------------------------> evaluate <------------+
       |                                      |
       v                                      v
 canonical response ----------------------> report

completed evaluations --(future optional)--> ML dataset --> shortlist --> simulate/verify
```

依存方向は一方向にする。

- `simulate` はanalysis、同定、最適化、MLをimportしない。
- `analyze` はcanonical responseだけを受け、Case、ngspice、保存先をimportしない。
- `identify` と `optimize` は共通の `evaluate(problem, parameters)` を呼ぶ。
- `identify` はlatent parameterだけ、`optimize` はdesign variableだけを変更できる。
- core runtimeはMLをimportしない。MLを再導入する場合はoptional packageまたはextraにする。
- reportは保存済み結果を読み、solverを再実行しない。

### 7.1 実装単位と責務

既存ファイルをさらに細かく増やさず、use case単位へ寄せる。

| 目標単位 | 所有するもの | 所有しないもの |
|---|---|---|
| `input` | YAML/netlist/測定ファイル読込、一度だけの型変換、役割付与 | solver、指標計算、探索 |
| `circuit` | 素子、接続、検証済みtopology catalog、netlist生成 | 最適化、学習 |
| `plasma` | E0〜E3の端子モデルinterfaceと電気的適用範囲 | 化学、密度計算、プロセス予測 |
| `simulation` | 解析要求、ngspice実行、canonical response | 指標、合否、候補選択 |
| `analysis` | 波形・ACからの純粋な電気量計算 | ファイル探索、solver、閾値 |
| `evaluation` | 一候補×全scenarioの計算、目的・制約 | 次候補の提案 |
| `identification` | latent parameterの推定、識別性、holdout再現 | 回路設計変数 |
| `optimization` | design variableの提案と履歴 | プラズマ推定、solver詳細 |
| `reporting` | 保存済み結果の表・標準図 | 再計算、合否規則の再定義 |
| optional `ml` | dataset、forward surrogate、shortlist | 真値、最終合否、solver代替 |

公開CLIもuse caseへ合わせ、最終的に `simulate`、`analyze`、`identify`、`optimize`、`report` を基本とする。
同じ内部処理に複数の入口を作らない。低水準の診断コマンドは通常helpから分離する。

### 7.2 最小限のアーキテクチャ規則

Import Linterには次の依存方向だけを置く。物理範囲や全フィールドをimport契約で表現しない。

1. simulationはanalysis/evaluation/optimization/identification/MLへ依存しない。
2. analysisはinput、solver、保存、最適化へ依存しない。
3. optimizationとidentificationはsolver実装ではなくevaluation interfaceへ依存する。
4. core runtimeはoptional MLへ依存しない。

Ruff、Pyrefly、Radonは局所品質の監視に使うが、ツール合格をアーキテクチャ有用性の合格と同一視しない。

## 8. 現行コードの扱い

### 8.1 削除するもの

R0で次を公開基盤から削除した。

- `pcd/ml/`
- `pcd/corpus.py`
- `pcd/corpus_evaluation.py`
- `pcd/circuit_graph.py` と `Circuit` に混在していたgraph専用API
- `ml-prepare`、`ml-evaluate`、`ml-corpus`、`ml-corpus-evaluate`
- 対応するML/corpus専用テストとREADME/input-formatの使用説明
- `.importlinter` のML/corpus専用規則
- `bench/ml/` の実行コードと、通常利用を示す説明
- release suiteのML結果添付・合否判定
- RFコンパイル結果とベンチマークcaseに残っていた `topology_family`

`bench/ml` の不合格値は[短い履歴文書](ml-experiment-history-ja.md)へ移した。詳細はGit履歴から再現できるため、
失敗した製品面を保つ目的で数千行を維持しない。

影響調査の結果、`optuna-benchmark` optional dependency groupはML runtimeではなく、現在のエッチングCCP
最適化図を再現するcase-localベンチマークだけが利用していた。このためR0では削除せず、通常installとruntime
importから隔離したまま残す。R4で採用optimizerを決める際に、再現性を失わない移行手順とともに依存を最小化する。

### 8.2 再設計まで採用しないもの

- 削除済み `circuit_graph.v1` と現行graph encodingを互換復活すること
- 現行NumPy GNN/MLPの重み・構造
- KNN固定poolランキング
- benchmark-localなOptuna sampler比較を製品optimizerへ昇格すること

### 8.3 残すもの

- ngspice実行、canonical波形・AC表、純粋analysis
- 外部netlist入力
- grid、random、Differential Evolutionの単純baseline
- 選択候補を全scenarioで再計算するstudy/evaluation経路
- 静的端子R+jX、CCP/ICP lumped model
- 正値の外生 `R(t)`。ただしプラズマ自己整合モデルとは呼ばない
- 現在の図とrun artifact。過去の問題設定・数値証拠としてのみ扱う

## 9. 優先順位付き実装計画

### R0: ML公開面の撤去と文書整合 — 完了（2026-10-04）

実装:

- 8.1のML/corpusコード、CLI、テスト、文書を削除
- 不合格結果を一ページの履歴へ要約
- READMEを回路計算・同定・最適化の目的へ戻す
- import規則と品質実行を現行runtimeへ同期
- graph都合で増えていた`Circuit`の二重素子表現を一つへ戻す

完了条件:

- 通常installにML framework・OptunaHubが入らない
- `pcd --help` に未成立のML機能がない
- ngspice計算、解析、studyの既存受入試験が通る
- 削除後の孤立import、未使用依存、古い手順がない

実績:

- 公開CLIから4コマンド、runtimeからML/corpus/graph実装を撤去した
- release gateを回路受入だけへ戻し、schemaを`pcd.release_closure.v2`へ更新した
- `Circuit.add`と`Circuit.raw`をSPICE生成だけの小さい契約へ戻した
- 旧基盤計画、ML専用ベンチマーク、専用テストを削除した
- OptunaHub等は`optuna-benchmark`を明示した場合だけ解決され、通常runtimeには入らない

R0後の監査:

- Import Linterの解析対象は72 module/206 dependencyから60 module/176 dependencyへ減った。削除対象だけを
  守っていた二つの契約も消し、残る5契約は全て成立している
- Ruff complexity違反は0、Radon全体平均はA（4.12）である。一方、`resolve_study_case`はD（27）、
  `_print_run_summary`はD（26）であり、前者はR1の入力役割統合、後者はresult shape確定後の表示整理で扱う
- Pyreflyは0 error、64 warningである。大半は既存の防御的な数値型変換であり、別の型checkerやsuppressionを
  増やさず、R1/R2で該当境界を触る時に減らす
- `plan.py`と`core/models.py`のmaintainabilityはCである。点数のために型を細分化せず、R1で辞書再解釈を
  減らした結果として改善するかを再測定する

### R1: パラメータ役割を持つ一つの内部問題表現 — 完了（2026-10-04）

実装:

- 既存の`Case`を問題本体、既存の`ProbePlan`を観測要求として再利用し、重複wrapperを作らず
  `ParameterSet`だけを追加
- fixed hardware、calibration、operating condition、design、latentを入力時に一度だけ分類
- 現行YAML/netlistをadapterで同じ内部型へ変換
- `plan.py`、`study.py`、validationで同じ辞書を再解釈している箇所を移行

完了条件:

- 全変数が一つの役割だけを持つ
- optimizerへ渡るのはdesign variableだけ
- identifierへ渡るのはlatent parameterだけ
- simulatorは両者を単なる解決済み数値として受ける

実績:

- `pcd/problem.py`を入力役割の唯一の所有者とし、fixed、calibration、operating、control、design、latentを
  `ParameterRole`で相互排他的に解決するようにした
- 公開RF入力の`network.fixed`はfixed、`network.search`だけをdesign、`network.tuning`はcontrol、
  table/condition列はoperatingへ変換するように修正した
- 高度入力はbounds・複数choices・既定値なしをdesignとして推論し、`role: calibration`と`role: latent`を
  明示できる。scenario/controlとの重複や複数箇所での再宣言は入力エラーにした
- builtin/extension optimizerへ渡すCaseをdesignだけへ射影し、grid、random、DEの候補から固定値とlatent値を
  除外した。固定・校正・latentの既定値はsolver入力で従来どおり解決する
- `study_result.json`へ役割一覧と定数値を残し、`evaluations.csv`と準静的snapshotでは`fixed.*`、
  `calibration.*`、`latent.*`をdesign列と分離した

設計変更と派生影響:

- 当初は新しい`CircuitProblem`と`ObservationSet`も作る予定だったが、既存`Case`と`ProbePlan`と責務が重なる。
  型を増やすこと自体が複雑化になるため追加せず、R2でもこの二つを拡張せずに一評価経路へ渡す
- 従来は公開入力の固定素子もCandidate値として保存していた。今後Candidateは変更可能なdesignだけを表し、
  固定素子は結果表の`fixed.*`と保存済みcaseで確認する。これにより候補とBOM定数の意味を混ぜない
- latentはこの段階では既定値を使う既知のforward入力であり、更新する処理はまだない。R3のidentifierだけが
  `ParameterRole.LATENT`を更新する

R1後の監査:

- `quality-pr`は665件、branch coverage 93.01%で合格し、Ruff、Pyrefly（0 error/59 warning）、
  Import Linter 5/5、Bandit、pip-audit、Vulture、detect-secretsも合格した
- Import Linterの解析対象は、新しい役割所有者1 moduleを加えて61 module/182 dependencyである。
  役割を追加してもsolver・analysis・CLIの依存方向は増やしていない
- `resolve_study_case`のRadon D（27）は、exact-grid override検査と保存plan更新を分けてC未満へ下げた。
  新しい`resolve_parameter_set`はC（12）、残るstudy側の主要hotspotは`run_case_study`のC（19）であり、
  R2の一評価経路整理で扱う
- ngspice 46で3候補の公開離散探索を再実行し、固定L1/C2は全行の`fixed.*`、探索C1だけが`design.*`に
  出力され、既知の最良C1と`|Gamma|=0.001022...`を再現した

### R2: 一評価経路の確定 — 完了（2026-10-04）

実装:

- 一回のforward solveを `simulate(problem, values)` に統一
- canonical responseから `analyze(response)` を呼ぶ
- `evaluate` は全scenarioを実行し、目的と制約を返す
- 保存・reportを計算から分離

完了条件:

- 外部netlistと構造化回路が同じresponse型を返す
- 一候補評価をCLI、optimizer、identifierから同じ方法で呼べる
- 失敗時に空波形や大きな架空lossを生成しない

実績:

- `sim_core.execute_case`を一回のforward solveとし、solverのcanonical `SimulationResult`と保存用`SimRecord`を
  一つの`SimulationRun`で返すようにした。既存`simulate_case`は同じ呼出しからrecordだけを返す薄い入口である
- `metrics.measure_response`をcanonical応答からの唯一の指標入口にした。保存済みrunは`records.load_simulation_result`
  で同じ型へ戻してから測るため、直接計算と再読込で別の数式を持たない
- `evaluation.CaseEvaluationBackend`へrequestからsolve、raw result、metricまでのadapterを移し、`study.py`から
  solver実行とartifact解釈を除いた。新規評価はin-memory応答を使い、cache hit時だけcanonical CSVを復元する
- 構造化回路と外部netlistをngspice 46で実行し、どちらも`SimulationResult`と`simulation_record.v2`を返すことを
  確認した。3候補の離散整合studyも失敗0、既知の最良反射`|Gamma|=0.001022...`を再現した
- solver失敗は従来どおりstatusとartifactとして残り、metricと架空lossは生成しない

設計変更と派生影響:

- 新しいresponse wrapperは追加せず、全solverがすでに返している`SimulationResult`をcanonical型として再利用した
- metric pluginの入力をartifact pathではなく、隔離したCase、解決済みparameter、`SimulationResult`へ変更した。
  pluginがファイル配置やsolverに依存しなくなり、同定と最適化から同じ測定処理を呼べる
- study実行をtrial収集とgeneration確定へ分け、`run_case_study`のRadon C（19）をC未満へ下げた。型や契約を
  増やすための分割ではなく、計算と出力確定の二責務に合わせた分割である

R2後の監査:

- `quality-pr`は670件、branch coverage 93.07%で合格し、Ruff、Pyrefly（0 error/59 warning）、Bandit、
  pip-audit、Vulture、detect-secretsも合格した
- Import Linterは62 module/193 dependencyを解析し、core、signals、CLI、simulation、analysisの5契約が成立した。
  新しいevaluation adapterを追加してもsimulationから上位層への逆依存はない
- architecture auditでは`run_case_study`と新しい`load_simulation_result`はC以上のhotspot一覧から消えた。
  残るDはCLI表示の`_print_run_summary`だけであり、電気計算・探索経路ではない

### R3: プラズマ端子同定MVP

状態: 完了。

実装:

- `pcd identify`は校正済み回路、複素V/Iのscenario CSV、boundedな`latent`だけを受ける。
  `run`の`design`候補と混ぜず、fixed/calibration/operating/controlも変更しない
- fit scenarioだけを既存のbounded DEと通常ngspice評価へ渡し、選択値を固定してoptimizerから見えなかった
  holdout scenarioを同じ計算・metric・cache-free replay経路で評価する
- 各latentを宣言boundsの正規化座標で摂動し、指定した電気量から局所感度行列、特異値、rank、条件数を出す。
  lossが小さくてもrank不足または条件数超過なら`not_identified`にする
- `terminal_vi_fit`は校正基準面の複素V/Iから目標インピーダンスを作り、予測R/Xとの差を明示scaleで評価する。
  fit/holdoutの予測・目標・残差CSVとsolver artifactを保存する
- profile係数も通常のbounded latentとして同じ経路を利用できる。専用plasma solverや別optimizer契約は追加しない

ベンチマーク:

- 校正済みseries R/L fixtureと有効CCP series R-L-Cを使い、10、13.56、27.12 MHzの複素V/Iで
  `R_eff`と`C_sheath_eq`をfitし、20、40.68 MHzをholdoutにした
- 64 trialで`R_eff=17.2353 ohm`（既知18 ohmに対し4.25%）、`C_sheath_eq=121.135 pF`
  （既知120 pFに対し0.95%）を回収した
- fit worst normalized impedance errorは0.01459、holdoutは0.00985。感度rankは2/2、条件数3.823である

完了条件と実績:

- 事前に固定した8%許容誤差内で既知パラメータを回収した
- 未使用2周波数をfitへ混ぜず、宣言したholdout loss 0.03以内で再現した
- fit/holdout solver失敗、loss超過、感度rank不足、条件数超過のいずれも成功扱いしない
- 実ngspice-46の公開CLI release suiteは同定を含む8入力、353 fresh solve、cache hit 0でGOとなった

### R4: 回路素子最適化MVP

状態: 完了。

実装:

- 完全列挙grid、単純random、bounded `DE/rand/1/bin`はすべて同じ`CaseEvaluationBackend`と
  Candidate x Scenario x Control評価を使う。optimizerはsolverやmetricをimportしない
- 公開RF入力は有限候補の完全列挙だけを許可し、連続DEは明示的なadvanced caseに限定した
- 探索順位はsolver完了、全scenario feasibility、制約違反、宣言objectiveの順であり、工学制約をlossへ
  隠していない
- 探索後の最終候補はraw cacheを読まず書かず、同じ通常solver経路で各scenarioの選択controlを再計算する。
  `best_candidate.json`と最終合否はこの再計算から作り、探索表・履歴と分離した
- 結果には探索parameter、objective履歴、物理単位付き制約余裕、canonical波形/AC artifactへの参照、
  最終再計算の件数と成否が残る

計画変更と理由:

- benchmark-localなOptuna/OptunaHub GPはruntime optimizerへ昇格しない。既存のcase限定図は再現できるが、
  coreに重いoptional依存と別の探索契約を加えるだけの汎用的優位はまだ示していない
- 「GPを必ず追加する」を完了条件から外し、小さい離散問題は完全列挙、低次元連続問題は依存なしDEを
  baselineとする。実運用でngspice費用が律速になり、同じ予算・複数seedの独立問題で改善した時だけ
  model-based optimizerを昇格する。この変更はR6のデータgateとも整合する

完了条件と実績:

- design以外のfixed/calibration/operating/control/latentをoptimizer候補へ混ぜない
- 探索では全scenario/controlを同じ評価経路で計算し、満たさない制約を最適値として隠さない
- 最終確認では選択済みcontrolだけをscenarioごとに1点再計算し、探索grid全体を無駄に繰り返さない
- 最終候補をcache-freeで独立再計算し、その結果だけを採否証拠として保存する
- 完全列挙は探索範囲全体、DEは観測した範囲だけの結論と明示し、未探索の大域最適性を主張しない

### R5: 検証済みトポロジcatalog

状態: 完了。

実装:

- `topology_catalog.py`をL、pi、高調波piの接続と部品roleの唯一の定義元にした。公開入力compilerと
  ngspice circuit builderが同じcatalogを読むため、追加時に接続記述が二重化しない
- catalogが所有するのは検証済み接続と部品roleだけである。素子値、損失、探索範囲、適用周波数、
  工学制約は装置・部品・運転条件で変わるため、誤った暗黙保証を避けて各caseに明示する
- 公開RF入力はcatalogにない接続を生成せず、各topologyを独立caseとして完全列挙または固定評価する。
  任意graphのedge追加・削除は行わない
- A1〜A3の独立複素インピーダンスgolden、netlist builder test、実ngspice caseが3 topologyを個別に検証し、
  共通の評価表で反射、電力、stress、制約余裕を比較できる

完了条件と実績:

- 各templateは同じcatalogからnetlist化され、独立goldenと実solverで接続・応答を確認できる
- 選択理由はcaseに宣言した周波数、負荷、素子候補、損失、制約と保存済み電気量から説明する
- topology名だけから適用周波数や許容stressを推測せず、装置にない任意接続も候補にしない

R4/R5後の実行監査:

- ngspice 46を使う7ケースの公開CLI release suiteは142回のfresh solve、cache hit 0で全件合格した。
  これには探索に加え、各scenarioの選択controlを使うcache-free最終再計算を含む
- 3候補のpi整合gridは既知値`C1=2.58221757073e-10 F`を選び、再計算後も
  `|Gamma|=0.001022055288...`、正の制約余裕で合格した
- RC連続設計は単一電圧からRとCを同時に決める非識別な例をやめ、設置済みRを固定してCだけを探索した。
  24点DEでnormalized RMSEを0.340510から0.180607へ下げ、peak 4.81899 V < 4.9 Vを満たして再計算も成功した
- DEの範囲外変異はclipではなく反射し、それでも同一点になる場合はseeded resampleするため、24 trialを
  重複cache hitで消費しない
- quality gateは675件、branch coverage 93.04%で合格した。Ruff、Pyrefly（0 error/59 warning）、
  Import Linter（63 module/195 dependency、5契約）、Bandit、pip-audit、Vulture、detect-secretsも合格した
- architecture auditはRuff C90と5依存契約に合格した。今回追加した最終再計算とcatalogに新しいC以上の
  hotspotはなく、残るDはCLI表示だけである

R3完了後の最終監査:

- 同定を追加した公開CLI release suiteは8入力をstrict validationし、ngspice 46で353回をfresh実行した。
  fit探索、fit最終再計算、holdout計算、holdout最終再計算、局所感度計算を含み、cache hit 0、全件合格、
  circuit-analysis / deterministic-sizing / effective-terminal-identification判定はGOである
- `quality-pr`は706件、branch coverage 93.01%で合格した。Ruff、Pyrefly（0 error/59 warning）、
  Bandit、pip-audit、Vulture、detect-secretsも合格した
- Import Linterは64 module/206 dependencyを解析し、core、signals、CLI、simulation、analysisの5契約が
  すべて成立した。`identification.py`はsolver実装へ迂回せず、通常study/evaluation経路だけを使う
- architecture auditはRuff C90に合格した。同定処理と`terminal_vi_fit`にC以上の新規hotspotはなく、
  Dは既存のCLI表示だけである

### R6: ML再導入のデータgate

現在位置:

- R0〜R5の完了条件は満たしたため、候補パッケージの技術調査と比較基準の策定は開始できる
- 一方、複数装置・独立campaignを含む学習データ量と、ngspice計算が運用上の律速であるという実測は
  まだ成立していない。したがって現時点ではML依存の追加、graph schema実装、学習API公開は行わない
- 次の作業は、対象タスク、必要データ、group holdout、単純baseline、solver削減率を先に固定し、その条件に
  合うパッケージを比較することである。パッケージ名を先に決めて現行基盤へ依存を追加しない

開始条件:

- R0〜R5が完了
- 複数topology、複数装置または独立campaign、複数recipeの十分な完了評価がある
- ngspice費用が実際の運用上の律速になっている

実装順:

1. 対象回路を表せるgraph schema v2
2. group splitとbaseline
3. 小型GNN forward surrogate
4. 不確かさまたはensembleによるshortlist
5. shortlistを通常ngspiceへ戻すclosed-loop評価

合格条件:

- 独立topology/campaign holdoutで単純baselineを上回る
- 同一solver予算で最終設計regretまたはfeasible到達率を改善
- ngspice呼出しを事前閾値以上削減
- 最終候補の全scenario再計算が成功

一つでも満たさない場合、MLを再び削除できる構造にする。

### R7: 利用者向け完成

実装:

- 三つの短い例: forward、plasma identification、circuit optimization
- topology追加方法を一ページで説明
- 標準reportを回路図、入力、波形、loss履歴、最終素子値へ限定
- 古いCLI、スキーマ、重複文書を削除

完了条件:

- 初見の回路エンジニアが例から入力・出力・変更可能範囲を説明できる
- プラズマエンジニアが測定面、固定寄生、推定対象を区別できる
- 新しいtopologyまたは端子モデルの追加場所が一か所で分かる

## 10. 計画変更時の整合規則

計画変更時に更新するのは次の四点だけとする。

1. 対象範囲と対象外
2. パラメータ役割表
3. データフローと依存方向
4. その能力を証明するbenchmarkと合格条件

新しいライブラリ、診断、スキーマ、抽象interfaceを先に追加しない。既存の一つの問題で必要性が示され、
責務の所有場所が決まった後に追加する。オプション追加時も、既存処理へ分岐を散らさず、入力境界で解決して
一つの内部表現へ渡す。

## 11. 調査根拠

- Schmidt, Mussenbrock, Trieschmann, “Consistent simulation of capacitive radio-frequency discharges and external matching networks,”
  *Plasma Sources Science and Technology* 27 (2018), 105017,
  https://doi.org/10.1088/1361-6595/aae429 。plasma、matcher、generator、stray lossの相互作用を一つの
  系として扱う必要性を示す。
- Lee et al., “Development of a high-speed impedance measurement system for dual-frequency capacitive-coupled pulsed-plasma,”
  *Review of Scientific Instruments* 86 (2015), 083505,
  https://doi.org/10.1063/1.4928121 。matcherとchamberの間で時間分解インピーダンスを測定し、pulsed plasmaの
  transient intervalを観測している。
- Kuhfeld, Sakiyama, Czarnetzki, “Analytical plasma impedance model of dual frequency capacitive discharges with ion dynamics,”
  *Plasma Sources Science and Technology* 28 (2019), 035004,
  https://doi.org/10.1088/1361-6595/ab034f 。HFとLFで等価回路の応答が異なり、単純な共通series R-Cでは
  ion dynamicsを表せないことを示す。
- Sobolewski, “Current and Voltage Measurements in the Gaseous Electronics Conference RF Reference Cell,”
  *Journal of Research of NIST* 100 (1995), 341–351,
  https://doi.org/10.6028/jres.100.026 。電極、壁、絶縁体、外部回路の寄生と測定差を明示しており、
  寄生校正とplasma同定を分ける根拠になる。
- Hakhamaneshi et al., “Pretraining Graph Neural Networks for Few-shot Analog Circuit Modeling and Design,” 2022,
  https://arxiv.org/abs/2203.15913 。device/netとterminal種別を持つgraph表現、未見topology評価、大規模な
  simulation pretrainingの必要性を示す。これは手法候補の根拠であり、PCDの現データでの有効性証明ではない。
- Dong et al., “CktGNN: Circuit Graph Neural Network for Electronic Design Automation,” ICLR 2023,
  https://openreview.net/forum?id=NE2911Kq1sp 。既知subgraph basisと約10k回路のbenchmarkを用いており、
  小さな検証済みtopology grammarから始める方針と整合する。
- Zhang, He, Katabi, “Circuit-GNN: Graph Neural Networks for Distributed Circuit Design,” ICML 2019,
  https://proceedings.mlr.press/v97/zhang19e.html 。GNNによる異なるtopologyのforward modelと逆設計の可能性を
  示すが、分布定数EM回路の成果であり、plasma等価回路へ直接移植できる証拠ではない。
- Lyu et al., “Batch Bayesian Optimization via Multi-objective Acquisition Ensemble for Automated Analog Circuit Design,”
  ICML 2018, https://proceedings.mlr.press/v80/lyu18a.html 。高価な回路simulationに対するBOの適用根拠になる。
- Gardner et al., “Bayesian Optimization with Inequality Constraints,” ICML 2014,
  https://proceedings.mlr.press/v32/gardner14.html 。目的と同様に高価な工学制約を別にモデル化する根拠になる。
- Ru et al., “Bayesian Optimisation over Multiple Continuous and Categorical Inputs,” ICML 2020,
  https://proceedings.mlr.press/v119/ru20a.html 。topology/categoryと連続素子値を単純one-hot距離で扱わず、
  階層的またはmixed-variableな探索へ分ける根拠になる。
