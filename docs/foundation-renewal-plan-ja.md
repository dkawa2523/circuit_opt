# 回路解析基盤の再構成計画

## 1. 結論

PCD は実 ngspice を使う RF 整合回路スタディとしては動作している。しかし、現在の実装は
「一回の回路計算」「測定値の算出」「複数条件の評価」「探索」「保存・再実行」を一つの
設計スタディ経路へ集めすぎている。結果として、単純な一評価でも内部保存物が多く、
利用者が最初に知りたい回路結果より、識別子、世代、キャッシュ、診断情報が前面に出る。

今後の基盤は、次の五つを独立した処理として組み直す。

1. `simulate`: ネットリストと解析条件を ngspice へ渡し、生の波形・周波数応答を返す。
2. `measure`: 波形・周波数応答から回路指標を計算する。
3. `study`: 設計候補と外部条件を列挙し、評価表を作る。
4. `optimize`: `study` の一評価関数だけを呼び、次候補を提案する。
5. `report`: 保存済み結果だけを読み、意思決定用の表と図を作る。

プラズマモデルは ngspice 層へ直接混ぜず、静的端子モデル、時間系列モデル、外部状態モデルの
三段階に分ける。機械学習はシミュレータの代替真値にせず、候補提案を高速化する任意層とし、
採用候補は必ず ngspice で再評価する。

### 1.1 完成版で達成すること

本計画の完成対象を **PCD v1 回路設計基盤** と固定する。完成とは、機能や診断を増やし続けることではなく、
次の四つの利用目的を、同じngspice実行・計測・評価経路で再現できる状態をいう。

1. SPICE netlistまたは明示caseを一回解析し、波形・周波数応答と標準的な回路指標・図を得る。
2. chamber/plasmaを、測定または外部計算された端子R+jX、準静的時系列、外生R(t)として回路へ接続する。
3. 素子値と、宣言済みの検証済み回路テンプレートを、全Scenario/Control・工学制約込みで比較・最適化する。
4. MLは固定トポロジの数値sizingと宣言済みトポロジ間の応答学習を分けて評価し、未見トポロジ予測と
   実ngspice評価数削減をそれぞれ実証した場合だけ候補提案へ使う。

完成対象外も固定する。plasma chemistry・sheath・密度の自己整合計算、PCB/三次元配置、EM/熱設計、
任意グラフ生成はPCD v1へ追加しない。「配置最適化」は宣言済み回路テンプレートの選択を意味し、
物理配置や任意接続生成を意味しない。これらが必要なら、PCDへ機能を継ぎ足さず別モデルまたは別段階として設計する。

段階0〜4の回路計算能力と従来のrelease closureは完了している。固定KNNによる旧P3は不合格の履歴として
保持する。その結果を深層学習全般の不合格とは扱わず、宣言済み回路を対象にしたグラフ表現、PCD専用AC
コーパス、同一分割のbaseline/MLP/GNN比較まで実装した。最初のM2実ngspice corpusではconstant baselineが
三protocolすべてで最良だったため、model保存へ進めない。既に見たtestへ合わせてmodelを調整せず、次は
独立したtopology/frequency/load corpusを固定して再検証する。任意グラフ生成を追加せず、製品全体の目的を
ML実験へ置き換えない。

## 2. 今回確認した事実

### 2.1 実行と品質

- ngspice 46 のコンソール実行を検出し、バッチ解析が成功した。
- 全703テストが成功した。したがって現状は「動かない」のではなく「動くが理解・運用しにくい」。
- Python 実装は `pcd/` 72 モジュール、RadonのSLOCで 12,654 行である。
- 主な集中箇所はpublic入力変換、study、CLI adapterである。MLは保存済みstudyを扱うapplication adapter、
  純粋なcorpus整形、既存dataset、評価、rankingへ分離した。新しいcorpus export/evaluation、graph encoding、
  NumPy model、比較protocolの各モジュールはRadon MI A、C以上の関数0を維持している。
- 調査開始時の `bench/` のコード・入力・説明は約 9,500 行あり、実行基盤より大きかった。
  特に個別報告書生成は2,295行を占めていた。P2で既存suite summaryだけを読む372 SLOCの投影へ置換し、
  図生成1,718行は成果物固有処理として `bench/figures` に隔離されていることを確認した。

### 2.2 一評価の出力

`examples/rf_component_stress.yaml` を設計studyとして実 ngspice で一回評価したところ、計算は約 0.07 秒で
成功し、反射係数、入力 R/X、電力、効率、各素子 V/I/損失を正しく得た。一方、一評価だけで
17 ファイルが `raw/`、`evaluations/`、`artifacts/`、`generations/` に分散し、
`evaluations.csv` は 98 列になった。

これは監査・キャッシュ用としては説明できるが、日常の回路設計出力としては過剰である。
利用者向けの一次結果と、再現・デバッグ用の二次情報を分離する必要がある。

段階2bの最初の変更で、直接 `sim-run` はルートを `summary.json`、`data/`、`debug/` の3項目へ整理した。
canonical波形・AC表は `data/`、case・netlist・solver log・詳細manifestは `debug/` に置き、重複していた
`params.json` は削除した。この時点では旧ルート `sim_manifest.json` をreaderだけで読み取れた。上記17ファイル問題のうち、
直接simulationは解消したが、study側のcache・世代・評価保存の整理は引き続き段階2bの対象であった。
P2の参照調査で旧ルート形式を生成・利用する現行入力と成果物がないことを確認し、このreader分岐も削除した。

段階2bの第三変更では、1評価の同一ケースで旧形式と新形式を実測した。読み戻し箇所が存在しなかった
`evaluations/<key>/result.json` を廃止し、raw cacheからmetric/constraintを毎回再計算する既存動作を
正式な責務にした。この時点ではcandidate詳細を全trialについて一度だけ保存し、選択trialをindexで参照した。さらに物理入力と
重複していた `observation.params/case/solver` と未使用の `artifact.run_dir` を表から除いた。その結果、
ファイル数は16から15、総量は26,630 byteから19,248 byte（27.7%減）、candidate JSONは
6,745 byteから3,236 byte（52.0%減）、root summaryは4,139 byteから3,378 byte、
`evaluations.csv` は55列から46列になった。raw cache、公開generation、完全評価表は用途が異なるため残す。

### 2.3 実装済み能力と名称のずれ

| 項目 | 現在の実能力 | v1判定 |
|---|---|---|
| ngspice 回路計算 | AC、tran、外部ネットリスト、R/L/C、組込みRF topology | 実装済み。P1利用者経路を実ngspiceで確認済み |
| RF 指標 | R/X、反射、VSWR、電力、効率、素子stress、高調波、共振・帯域・Q | 実装済み。P1で出力を統一確認済み |
| plasma等価回路 | 静的CCP/ICP端子一ポート、供給されたR+jX | v1範囲は実装済み。自己整合plasmaは対象外 |
| 時間変化 | 準静的R+jX snapshot、外生R(t) transient | v1範囲は実装済み。状態伝播と未定義C(t)/L(t)は対象外 |
| 素子値最適化 | 離散完全列挙、seed固定DE、制約優先、Pareto | 実装済み。P1目的波形workflowを実ngspiceで確認済み |
| 機械学習 | role固定dataset、固定pool回顧ranking、物理回路graph、複数study AC corpus、同一splitのridge/MLP/GNN比較 | **旧P3は不合格。** M0〜M2は実装済みだがM2も進級不合格、model保存・逐次提案は未実装 |
| 配置・接続最適化 | 宣言済みL/π/高調波templateの選択 | v1範囲は実装済み。任意接続・物理配置は対象外 |
| 標準図 | 保存runから波形、impedance、Smith、電力・stress、高調波比較図 | 実装済み。個別論文再現はbenchmark責務 |

「ML 対応」「動的素子対応」「配置最適化」と記載する場合は、上表の未実装部分を完成するまで
「予定」「準静的snapshotのみ」または「データ受け渡しのみ」と明記する。

### 2.4 アーキテクチャ品質ツールによる実測

2026-10-01 時点の固定済み開発環境で、Ruff 0.16.3、Pyrefly 1.2.0、Import Linter 2.13、
Radon 6.0.1 を実行した。四つは同じものを測っていないため、単一の合否点数へ統合しない。
同じ監査は `uv run --frozen python -m nox -s architecture-audit` で再実行できる。

| ツール | 実測 | 解釈 |
|---|---|---|
| Ruff | 違反 0、McCabe `max-complexity=10` も通過 | 局所的な構文・既知バグ・Ruff方式の複雑度は管理されている |
| Pyrefly | 既定エラー 0、warning レベルでは 62 件、新しいM2実装は0件 | 残る旧変換warningは移行単位で減らす |
| Import Linter | 72モジュール、206依存、7規則すべて成功 | simulationからmeasurement/reportへの逆依存、analysisからCase/solverへの依存、ML/corpusからsolver・study実行・保存への依存がない |
| Radon CC | 891ブロック、平均 A（4.09） | C以上は34関数。新しいM1/M2モジュールにはC以上がない |
| Radon MI | `analysis/`、`reporting/`、`corpus.py` 21.88、`ml/corpus.py` 27.30はA。`validation.py` はB、`plan.py` と `core/models.py` がC | 次の主要ホットスポットはpublic入力変換と結果モデル |

Radon のD判定は `study.resolve_study_case` 27、`cli._print_run_summary` 26 の2関数で、E/Fはなかった。
旧 `analysis._final_periodic_cycles` 21は、周期窓検査、根拠作成、再標本化を同じowner内で分け、
`analysis.transient.final_periodic_cycles` 9まで下げた。C判定には solver実行、入力変換、
検証、結果要約、周期処理が集中している。複雑度を下げるためだけに物理式を細切れにはせず、
「入力の分岐」「保存・表示の分岐」「数値計算」を分離する対象として使う。

内部import graphに循環はなかった。fan-outは `cli` 19、`study/reporting.run` 15、`api/sim_methods` 12、
`metrics` 11、fan-inは `case` 17、`core.models/simulation_input` 13、`simulation` 12である。
`simulation_input` のfan-inは、Case解釈を一か所へ寄せた結果であり、
fan-outは4に限定されている。計測側で最も参照される `analysis.ac` のfan-inは5で、
solver/netlist/sim_coreはanalysis packageを参照しない。
最大の残課題は、study、circuit、load側に残る `Case.data: dict[str, Any]` の再解釈である。

## 3. 現在の主な構造課題

### 3.1 二つの入力表現と手書き変換が大きい

`pcd.rf.v1` は利用者にとって比較的短いが、`plan.py` が高度形式 `case_yaml.v1` へ変換し、
`validation.py` と `study_config.py` が再び辞書を解釈する。各層が `dict[str, Any]` を読むため、
どこで既定値が決まり、どこから必須かを追いにくい。

改良は「スキーマを増やす」ことではなく、入力境界で用途別の小さな型へ一度変換し、以後の層が
YAML の辞書を再解釈しないことである。全機能を持つ巨大なCase dataclassは別の密結合を作るため
採用しない。短い RF YAML と高度なネットリスト入力はフロントエンドとして残し、同じuse-case内では
一つの解決済み表現を共有する。

### 3.2 simulationとmeasurement、AC/tran/stressの分離は完了

第一段階で旧 `analysis.py` から解析要求、probe計画、ngspice制御文、solver出力読込みを外した。
836行から506行に減らした後、段階2の第一分割で旧file自体を削除した。現在のownerは次である。

- `simulation.py`: solver非依存の `AnalysisRequest`/`SimulationResult` と標準列。
- `probes.py`: Caseを知らない `ProbePlan`、`ComponentObservation` とprobe命名規則を所有。
- `component_models.py`: Case上のobserve/ESR/DCR宣言を純粋な観測型へ解決。
- `simulation_input.py`: solver設定、解析要求、probe、測定基準面、netlist optionをCaseから一度だけ解決し、
  `ResolvedSimulationCase` を生成。
- `netlist.py`: `AnalysisRequest` からngspice制御文を描画。
- `ngspice_io.py`: `wrdata` の版差を標準DataFrame/arrayへ変換。
- `solver.py`: `SolverRunRequest` だけを受けてngspiceを実行し、Case辞書を読まない。
- `analysis/__init__.py`: 従来のimportを保つ37行の公開facade。計算を持たない。
- `analysis/ac.py`: Z、Γ、VSWR、AC電力、周波数点選択。
- `analysis/transient.py`: 周期窓、整定根拠、時間平均電力、高調波、RF端子指標。
- `analysis/stress.py`: AC/過渡の素子端子V/I、ESR/DCR損失、損失収支。

`metrics.py`、`reporting/`、CLI、validationは公開facadeを横断せず、必要なownerを直接importする。
一方、既存利用者とbenchmarkは `from pcd.analysis import ...` を継続できる。保存前の
`SimulationResult` と保存後のcanonical waveformから同じRF指標を再生成する境界testも追加した。

保存済みrunを入力とする `analyze` use caseを追加し、ACではR/X、|Z|/位相、Smith、反射損失、
VSWR、電力・stressを、過渡ではV/I/瞬時電力、周期整定、高調波、平均電力を再生成する。
旧 `visualize-response` と `response_plot.py` は機能を包含したため削除した。`analyze` は計測だけを
所有し、工学的合否はstudyに残す。複数周波数点では、load端子電力が保存されていればその実電力、
なければ `1-|Gamma|^2` を応答量として、最大点、両側の半電力点、-3 dB帯域、fractional bandwidth、
loaded Qを算出する。単一点、片側しか含まない掃引、unloaded component Qには値を作らない。
高調波はH1基準の電圧・電流dBc比較図にした。これにより段階2の完了条件を満たした。
Sパラメータや群遅延は対応するsolver出力と基準面を定義してから追加し、名前だけの機能を先行させない。

### 3.3 `study.py` が評価、キャッシュ、探索履歴、表出力をまとめて所有する

最適化器は「候補を提案し、評価値を受け取る」だけでよい。保存世代、ファイルハッシュ、
ngspice 実行ディレクトリを知るべきではない。次の境界に変える。

```text
DesignPoint -> Evaluator.evaluate() -> Evaluation
                    |
                    +-> Simulator -> SimulationResult
                    +-> Measurement -> MetricSet

SearchStrategy.ask() -> DesignPoint
SearchStrategy.tell(Evaluation)
```

### 3.4 保存形式が内部都合を利用者へ露出している

既定保存物は次で十分である。

```text
run/
  summary.json          利用者が最初に読む結果、失敗理由、主要指標
  data/ac.csv           AC の場合
  data/transient.csv    過渡の場合
  figures/              標準図
  debug/netlist.cir     再現・調査用
  debug/solver.log
  debug/manifest.json   バージョン、ハッシュ、詳細診断
```

スタディはrootの `study_result.json` を最初に読む要約兼commit pointerとし、そこからactive generationの
`evaluations.csv`、`candidates/<selected>.json`、必要時の `pareto_front.csv` / `snapshot_response.csv` へ
直接到達できる形に固定した。選択candidate JSONが各conditionのmanifest、canonical表、netlist、solver logを
指す。標準図が必要な一評価だけ、そのmanifestを `analyze` へ渡す。大量のsolver出力はrootの `artifacts/`、
再利用cacheは `raw/`、中断時にも混ざらない公開単位は `generations/` が所有し、同じ内容を `best/` へ
複製しない。

### 3.5 拡張機構が実装数に対して先行している

現在は回路、負荷、ソルバー、指標、探索の登録機構があるが、組込みソルバーは ngspice 一つ、
探索は grid/random 二つである。外部利用実績が確認できるまで、プラグインは `pcd.extensions`
一か所へ集め、通常経路からは見せない。汎用化のための抽象層を、具体機能より先に増やさない。

### 3.6 simulation→measurementの逆依存は解消した

`pcd.signals` と `pcd.rf_loads` は、配列または物理パラメータだけを受ける独立した計算単位であり、
現状でも再利用しやすい。第一段階で `solver.py`、`netlist.py`、`sim_core.py` から
analysis packageへのimportを除去し、Import Linterで再導入を禁止した。

次の四つの境界は実装済みである。

- analysis request/probe plan: AC/tran要求と必要probeを別の小さな型で表し、netlist生成が使用する。
- solver adapter: netlistを実行し、未解釈の軸付き配列を返す。回路指標を知らない。
- simulation result I/O: solver形式を標準配列へ変換する。目的関数や合否を知らない。
- measurement: 標準配列からZ、Γ、電力、stress、高調波を計算する純粋関数。

第一段階ではImport Linterに「simulation準備・実行から `analysis/metrics` へ依存しない」規則を
追加し、段階2で「analysisは標準data、core、signals、probe型以外のCase/solver/persistenceを参照しない」
規則も追加した。これは利用者向け契約ではなく、解消済みの依存方向だけを守る保守用静的検査である。
さらに `simulation_input.py` へsolver・probe・測定基準面のCase解釈を集約し、`simulation.py`、`probes.py`、
`solver.py` はCase/YAML辞書を読まない。netlist生成とsolver実行は同じ `ResolvedSimulationCase` を使う。
すべてのsolverは `SolverRunRequest` を受ける。P2で旧4引数plugin分岐を削除し、内蔵ngspice、外部adapter、
test doubleが同じ一引数境界を使うようにした。

### 3.7 失敗処理と診断の三重化は解消済み

旧実装では `solver.ngspice_cli`、`sim_core.simulate_case`、`core.StudyRunner` がそれぞれ広い
`Exception` を捕捉し、別形式の `status/error/diagnostics` へ変換していた。この方式は連続探索を
止めない一方、プログラム不具合まで「評価失敗」という正常データに見せていた。

段階2bで、失敗の意味を判定できるownerだけが一度変換する境界へ変更した。ngspice adapterは
実行ファイル不在、timeout、非zero終了、要求出力不在、solver出力の読取り不成立を
`SimulationResult(status="failed")` にする。Evaluatorが認識する既知失敗は明示的な
`RawResult(status="failed")` で返す。Measurementでは `UnsettledMeasurementError` だけを
`not_settled` にする。`sim_core.simulate_case` と `StudyRunner` は任意例外を捕捉しない。

したがって、未知のmodel/method、metric名重複、目的指標欠落、plugin実装不具合はPython APIへ
そのまま送出される。solver呼出し前まで完了していれば正直な `prepared` recordだけが残り、空波形を
失敗観測として作らない。CLIの入力不備はexit 2の `invalid` とし、`--allow-failure` はsolverが
明示した失敗結果だけに適用する。これにより新しい例外契約階層を増やさず、診断の重複を削除した。

同様に、有限値、shape、必須fieldの検査が `plan.py`、`validation.py`、dataclassの
`__post_init__`、rendererで重複している。外部入力を型付き内部モデルへ変換する地点で一度検査し、
内部の純粋計算はその型を前提にする。物理的受動性、基準面、単位、探索role重複の検査は
計算の意味を守るため残す。

検査責務の棚卸し結果は次のとおりである。単に検査数を減らすのではなく、同じ意味の再解釈だけを
削除する。

| 入力・不変条件 | owner | 方針 |
|---|---|---|
| `pcd.rf.v1` の必須field、候補role、負荷・定格 | `plan.py` | authored入力を実行Caseへ変換するため維持 |
| solver名・timeout、AC/tranのshape・有限値・範囲 | `simulation_input.py` / `simulation.py` | 型付きsolver入力への変換時に一度だけ検査 |
| source container/name、probe列、measurement/load mapping、測定基準面 | `simulation_input.py` | 型付きsimulation入力への変換へ集約済み |
| circuit/load section、builder選択、model必須値・値域、builder出力型 | `netlist.py` / `sim_methods.py` / `rf_loads.py` | 実際のin-memory model構築へ集約済み |
| Candidate/Scenario/Control、評価結果 | `core/models.py` | generic studyの不変条件として維持 |
| 複数sectionにまたがる適用範囲・周期長・物理根拠 | `validation.py` | typed adapterを呼ぶ薄いreportと意味検査に限定 |
| ngspice文字列生成 | `netlist.py` | 型付き入力を前提とし、同じ数値範囲を再検査しない |

段階2bの第四変更では、`validation.py` に複製されていたAC sweep、AC一点、tran、timeoutの
個別parserと13種類の細分化診断を削除した。`AnalysisRequest`、`TransientAnalysis`、`AcSweep`、
`SolverSettings` が実行・`validate-case` 共通のownerになり、不正な明示timeoutを300秒へ黙って
置換する処理も廃止した。設定解決をrun directory確保より先に移し、入力不備は成果物を残さない。
物理的に必要な正値、有限値、解析区間、AC一点とsweepの排他は型付き境界で維持している。

第五変更では、source単体/listのshapeと排他規則、構造化source名、測定対象source、probe列、
`measurement` / `load` / `load.ports` mappingを `simulation_input.py` に集約した。netlist生成も同じ
source resolverを使い、`validation.py` の並行したsource/probe parserとsection別のshape診断を削除した。
`validate-case` は実行と同じ `ResolvedSimulationCase` を一度構築し、入力不備を一つの
`simulation.invalid_input` と具体的な原因で返す。周期窓、基準impedance、測定器挿入とload有無、
load modelの適用範囲など複数sectionにまたがる工学的検査は維持した。

第六変更では、circuit/load section、YAML component、外部netlist方針、RF loadの必須parameterと
受動性・値域を実際のbuilderへ集約した。`validation.py` は同じ `NetlistInputs` を構築して一つの
`model.invalid_input` と具体的原因を返し、load別field表、RF式の再呼出し、外部netlist用の並行parserを
削除した。reference plane、characterization、一点impedanceとAC解析の整合だけを工学的適用範囲として
残した。simulationはsimulation入力とin-memory modelの両方が成立してからrun directoryを確保する。

### 3.8 テスト量は多いが、内部保存形式への固定が強い

`tests/` は32個の `test_*.py`、合計9,146物理行、485個のtest関数からなり、parameterize後の全実行は
658件である。数値符号、電力収支、基準面、ngspice回帰を守るtestは価値が高い。一方、細かな
診断code、例外文言、内部schema、cache配置の全分岐を固定するtestは、不要機能の削除と
責務移動を難しくする。

testを次の三層に整理する。

1. 物理・数値test: 解析解、独立式、論文table、単位、符号、受動性。最優先で残す。
2. use-case test: `simulate/analyze/study/optimize` の公開入出力と実ngspice golden。
3. 実装test: parser、adapter、保存の少数の境界例。private helperや完全なdebug辞書は固定しない。

coverage 93%は退行検知の参考値であり、分岐を増やしてtestを増やす目標にはしない。不要機能を
削除した結果としてtestも削除し、公開動作と工学的根拠のcoverageを優先する。

## 4. 目標アーキテクチャ

```text
case.py / plan.py / study_config.py     外部入力を一度だけ実行用の型へ変換
simulation*.py / netlist.py / solver.py ngspice準備・実行。指標や順位を知らない
analysis/ / signals/                     保存可能な数値dataから純粋に回路量を計算
core/ / study.py                         Scenario/Control評価、制約、集約、世代公開
search.py / continuous_search.py         grid、random、DEの候補生成だけ
ml/                                      dataset、固定評価、条件付き候補順序
results/ / reporting/                    study表と保存runのsummary・標準図
cli.py                                   上記use caseを呼ぶ薄い入口
```

完成のためにpackageを全面rename・移動しない。現在の配置で「入力を型に変える」「解く」「測る」
「並べる」「探す」の依存方向は成立している。P2では実際の逆依存・重複ownerだけを直す。

### 4.1 依存方向と Import Linter

以下で `A -> B` は「AがBをimportする」を意味する。`core`、`simulation`、`signals` が基底であり、
具体solver・保存・CLIへの不要な逆依存を持たない。

```text
input adapters       -> core, simulation
solver execution     -> simulation, netlist
analysis             -> core, simulation, signals
study                -> core, simulation, analysis, search
ml                   -> pandas/numpyだけ（solver、study、保存へ依存しない）
reporting/results    -> 保存済み標準結果
cli                  -> public use-case functions
```

`Evaluator` は候補から評価への小さな入出力だけを持ち、optimizerからngspice、保存先、
manifestを隠す。ユースケース連携は薄い関数で行い、循環を避けるためだけの空の
service/interface層は作らない。

Import Linterは、分割後に次の意味ある少数規則だけを追加する。

- simulation準備・実行から `analysis/study/search/reporting/ml` へのimport禁止は実装済み。
- `analysis` からsolver adapter、保存、CLIへのimportを禁止する。
- `ml` から具体solver、study orchestration、保存形式へのimportを禁止する。
- CLIは引き続きleafとし、`core/signals` はorchestrationをimportしない。

現在の配置を追認する契約を増やすと移行を妨げる。したがって、残る規則も先にimport反転を
解消したchange setで追加する。依存境界検査は利用者向け「契約」ではなく保守用の静的検査である。

### 4.2 拡張時の編集箇所

「新規モデルに五ファイル以上の分岐追加が必要」という現状を解消する。一つの機能の定義、
適用範囲、実装、登録は可能な限り同じpackageに置き、他の層は公開型だけを見る。

| 追加したいもの | 主な編集先 | 隣接して必要なもの |
|---|---|---|
| 入力option | `case.py` / `plan.py` と値の実owner一か所 | 利用者例、境界test |
| 回路topology | `sim_methods.py` のbuilderまたはplugin一か所 | 登録、回路golden |
| プラズマ端子モデル | `rf_loads.py` の端子model | 式、単位、適用域、SPICE実現、文献fixture |
| solver | `solver.py` と `sim_registry.py` のadapter境界 | 能力表とadapter test |
| 指標 | `analysis/ac`, `transient`, `stress` のいずれか | 単位、符号、独立式test |
| 探索法 | `search.py` または `continuous_search.py` | ask/tell test、seed再現 |
| 図・表 | `reporting/` | 保存済み標準結果だけを入力にする |

`pcd.api` は安定ユースケース、Request/Result、拡張に必要な最小プロトコルに絞る。
内部dataclassを公開面へ並べない。新しい分岐が `plan.py`、`validation.py`、`sim_methods.py`、
`rf_loads.py` に同時追加される場合は、その機能のownerがまだ定まっていないと判定する。

## 5. 利用者向け操作

PCD v1の日常操作は、現在実装済みの次の三経路に固定する。名称変更のためだけのaliasや
並行service層は増やさない。

```text
pcd sim-run case.yaml              一回のngspice解析
pcd analyze run/                   保存済み結果の指標・標準図を再生成
pcd run study.yaml                 sweep、Scenario、素子値・既定template探索を完全評価
```

`run` はcase内のoptimizer設定を読む一つのstudy経路であり、`study` と `optimize` を別コマンドへ分けない。
`result-summary`、`result-prune`、`solver-diagnose`、`validate-case`、`sim-netlist` は運用・診断用、
`ml-prepare` と `ml-evaluate` は高度な検証用である。標準出力は結論、主要値、失敗制約、
結果ディレクトリだけにし、完全JSONは `--json` で得る。等価回路parameter同定は、測定series identity、
損失関数、identifiability、保留測定の要件が揃うまで、存在しない `model fit` コマンドとして約束しない。

### 5.1 回路・プラズマ端子解析の共通フロー

ngspiceの一般的な「netlist読込み→回路行列構築→解析→データ処理」の外側に、
実測と等価回路に必要な根拠・同定・検証を配置する。標準フローは次とする。

1. 設計問い、観測端子、基準面、周波数・時間域、必要精度を固定する。
2. 測定、論文表、図読取り、合成fixtureを分け、出典と不確かさの種類を記録する。
3. 校正・de-embeddingを外部で済ませ、同じ基準面の V/I/Z または素子パラメータへ正規化する。
4. 識別可能な最小端子モデルを選び、fit用とheld-out用を分け、残差と適用域を報告する。
5. 電源、matcher、cable、fixture、寄生を明示して回路を構成し、ACまたはtranを実行する。
6. 保存済み波形から Z、Γ、電力収支、効率、素子stress、周期整定を独立に測定する。
7. 操作可能な調整範囲、corner、感度、時間変化に対し、固定設計と調整可能設計を別々に評価する。
8. 探索はこの一評価を再利用し、選択候補を全scenarioでngspice再評価して図・表・適用限界を出す。

| 工程 | 現行コード | 不足と対応 |
|---|---|---|
| 基準面・文献根拠 | 強い | `bench/literature` のsource fidelityとdesign challengeの分離を維持 |
| 静的一ポートモデル | CCP/ICPとR+jXがある | 式・適用域・SPICE実現を同じownerへ移す |
| AC整合・stress | 主要指標はある | 標準Resultと共通figureへ集約 |
| 周期tran | 整定・電力・高調波がある | 保存と計測をsolverから分離 |
| モデル同定 | 汎用機能なし | fit/held-out残差を小さな独立use caseとして追加 |
| 感度・公差 | benchmark個別の例が中心 | 確率と決定論cornerを混同しない汎用studyへ整理 |
| 時間変化 | 準静的R+jX snapshotと外生R(t)を実装済み | C/Lは構成式と外部エネルギー授受を定義できる実案件が出た場合だけ追加 |
| 自己整合plasma | なし | コアの責務外。必要時のみ別adapter/co-simulationとする |

## 6. 回路解析で標準化する指標と図

### 6.1 AC・周波数領域

標準指標:

- 入力・負荷の複素インピーダンス `R`, `X`, `|Z|`, 位相。
- 反射係数 `Γ`、`|Γ|²`、return loss、VSWR。
- 電圧・電流利得、実電力・無効電力・皮相電力、伝送効率。
- 各素子の peak/RMS V/I、実効損失、制約余裕。
- solverが生成した電力応答の共振周波数、-3 dB帯域、loaded Q。unloaded Qや群遅延は対応する
  測定量・基準面・抽出法を定義してから扱う。
- 正規化感度 `S(y,x)=(x/y) dy/dx` と有限差分条件。

標準図:

1. Bode: 利得・位相対周波数。
2. インピーダンス: R/X と |Z|/位相対周波数。
3. Smith: 実計算点、受入境界、設計点を別記号で表示。
4. 電力・効率対周波数。
5. 素子ストレスと定格線。
6. 感度 tornado または周波数別感度。

ngspice は AC、noise、pole-zero、sensitivity、S パラメータを備える。ただし、派生量 Γ や
制約余裕の感度は、ngspice の素子パラメータ感度だけで完結しないため、基盤側の有限差分も必要である。

### 6.2 過渡・周期定常

標準指標:

- peak、RMS、DC、crest factor、整定時間、overshoot。
- 基本波振幅・位相、各高調波、THD。
- 瞬時電力、周期平均電力、パルス当たりエネルギー。
- 周期間残差と使用周期数。未整定値は合否に使わない。
- 時間窓ごとの基本波 R/X、|Γ|、反射電力比。

標準図:

1. 電源・負荷 V/I 波形。
2. V-I Lissajous と周期平均電力。
3. 高調波棒グラフ。
4. R(t)、X(t)、|Γ(t)|、供給・受電・損失電力の時間推移。
5. パルスごとのエネルギーと最大ストレス。

## 7. 時間変動素子とプラズマ等価回路

一つの「時間変動対応」という名前で異なる物理を混ぜない。

### 7.1 準静的スナップショット

測定した `time_s, resistance_ohm, reactance_ohm` を各時刻で独立した AC 条件として解く方式を
`impedance_profile` として実装した。RF 周期より十分遅い包絡変化に適用し、時間点間の動的状態は
主張しない。

出力は load/input R/X、Γ、進行・反射・受電電力、選択control、制約違反の時間表とした。高速度インピーダンス測定で
サブマイクロ秒級の端子インピーダンスが得られる例があり、入力データ形式とも整合する。

### 7.2 外生的な時間変動 R/L/C

advanced `from_yaml` componentに `profile` を実装した。

```yaml
value:
  profile: plasma_load.csv
  time_column: time_s
  value_column: resistance_ohm
  interpolation: linear   # linear | hold
  repeat_period_s: 75e-6  # optional
```

`component_profiles.py` がCSV時刻・正抵抗、linear/hold補間、明示的な周期終端、最短区間に対する
solver刻みを一度だけ検査し、ngspice behavioral resistorへ変換する。非周期linear profileは最終傾きの
外挿を止め、`.tran` は出力刻みを明示的な最大内部刻み `tmax` としても出力する。入力CSVは通常の
content-addressed archiveとfingerprintに含め、`observe: true` なら端子V/Iをcanonical transient表へ保存する。

実装範囲は散逸関係が明確な外生 `R(t)` に限定した。`C(t)` と `L(t)` は単なる値置換では電荷・磁束と
エネルギーの扱いが曖昧になり得るため、構成式と外部から注入・回収されるエネルギーの定義を持つ
実案件が出るまで段階3の完了条件から外す。この計画変更はcomponent入力だけに閉じ、study/core/optimizer
のAPIや保存形式を増やさない。任意の複素 `Z(t)` を瞬時 R と C に機械変換する機能も追加しない。

### 7.3 自己整合プラズマ―回路連成

回路から得た電力・端子量で外部プラズマ状態を更新し、更新した等価素子を回路へ戻す方式である。
これは ngspice の単独機能ではなく、状態更新器、収束条件、時間刻み、エネルギー収支を持つ
co-simulation である。別パッケージ／アダプタとして実装し、単純な端子一ポートと同じ名称で
扱わない。

Qu らの pulsed ICP 研究では、パルス中の R/X、整合 C、反射、電力経路が時間で変わり、機械式
可変容量の応答がプラズマ過渡より桁違いに遅い。したがって、時間変動解析の主要出力は
「一つの最適値」ではなく、時間別 Γ、電力、必要調整量、調整速度限界である。

## 8. 最適化と機械学習

### 8.1 素子値最適化

順序は次の通りとする。

1. 小さい離散空間は全列挙。完全性が明確で最も信頼できる。
2. v1の連続設計選択はseed固定 Differential Evolutionを使う。
3. MLは既存結果を使うoffline評価に限定する。P3の固定基準が不合格だったため、逐次候補提案や
   Bayesian optimizationは追加しない。

目的関数と制約を混ぜた不透明な一数値を標準出力にしない。まず可否、次に制約違反、最後に
目的指標という現在の優先順位は残す。探索履歴には候補、指標、制約、失敗、seed、実行時間を残す。

段階4の第一変更では、独立乱数だけだったadvanced `random`を基準に、追加依存のない
`differential_evolution`を実装した。候補生成だけを `continuous_search.py` が所有し、有限・増加boundを持つ
連続変数をlinear/log正規化空間の `DE/rand/1/bin` で探索する。母集団は連続変数数の4倍（最小4）とし、
宣言defaultを最初に評価した後、層化した初期点と少なくとも一世代を評価する。複数choice、整数、bool、
unbounded軸は拒否し、トポロジ選択を数値軸へ暗黙変換しない。

子候補の採否は新しいpenalty係数ではなく、既存のsolver証拠、全scenario可否、正規化制約違反、目的指標の
辞書式rankをそのまま使う。`study_history.json` は候補、世代、親index、採否、目的集約、失敗制約・評価、
scenario/evaluation数、seed、実行時間を別fieldで残し、旧互換optimizerへ渡す圧縮lossは利用者履歴へ出さない。
公開 `pcd.rf.v1` の完全離散列挙と全control評価は変更していない。

監査の結果、現行studyは探索中の全候補を既に全scenario・全controlで実ngspice評価しており、選択後に同じ
入力をもう一度solverへ渡す処理は証拠を増やさずraw cacheの再読込みになる。この重複再実行は段階4へ追加せず、
最良候補のcoverageと失敗制約を既存の完全評価から公開する。段階5で代理モデルが未計算候補を提案する場合だけ、
その候補をこの完全評価経路へ戻してngspiceで検証する。

段階4の第二変更では、2目的以上を宣言したstudyだけに `pareto_front.csv` を追加した。支配判定は
`core.aggregation` の純粋計算、表への射影は `results/report.py`、原子的な世代公開は `study.py` が所有する。
全controlのsolver成功、全scenario可行、有限な目的集約を満たす候補だけを対象にし、minimize/maximizeを
方向変換して非支配集合を求める。同じ目的座標でも素子値が違う候補は工学的選択肢として残す。従来の
可否優先・辞書式bestは変更せず、表の `selected` で対応を示す。

連続探索の表は探索済み候補内の `observed_candidates` であり、未探索の連続空間に対する完全性を主張しない。
`declared_grid` は完全列挙された離散候補だけに使う。これにより、多目的表示の追加が探索アルゴリズムや
物理計算の意味を暗黙に変更しない。

### 8.2 目的波形

L2 誤差だけでなく、用途に応じて次を合成できるようにする。

- 正規化 RMSE、相関、位相ずれ。
- rise/fall、overshoot、settling、DC offset。
- 高調波ごとの重み付き誤差、THD、パルスエネルギー。
- 必ず満たす peak/RMS/電力/素子制約。

### 8.3 ML 代理モデル

#### 8.3.1 旧固定トポロジbaseline（履歴）

`pcd/ml` は任意依存とし、コア実行に scikit-learn 等を必須化しない。最低限の流れは次である。

1. `evaluations.csv` の行粒度と失敗行を検査。
2. 入力特徴、目的・制約label、追跡列、除外列をmanifestで固定。
3. 同じ固定設計の全Scenario/Control行が跨らない group split。
4. 指標ごとの誤差と制約分類の識別性能を保留設計で評価。確率校正は両classの十分な件数がある場合だけ行う。
5. P3を通過した場合だけ、予測値による候補順序を逐次評価へ使う。不確かさ取得はv1に含めない。
6. 上位候補を対象版 ngspice で全条件再評価。

P3ではこの順序を事前固定した二つの81候補poolへ適用した。RC目的波形では12評価対random中央値14評価
（14.3%削減）、13.56 MHz π整合では19評価対27評価（29.6%削減）で、両方とも固定した30%基準を
満たさなかった。162件のpool評価と、選択した上位候補2件の別run rootでのngspice再評価はすべて成功した。
したがって、応答近似や回顧評価を「逐次最適化」と呼ばず、候補提案loopは追加しない。結果確認後のseed、
候補集合、閾値、モデル変更も行わない。

`selected_control` を同じ結果の予測特徴に使わない。学習時間が節約したシミュレーション時間を
上回る場合は ML を使わない。研究例も、固定トポロジの sizing と任意トポロジ生成を分けている。

#### 8.3.2 トポロジ認識ML更新

上記P3は固定された3近傍モデルと二つの候補poolに対する履歴であり、後から閾値や結果を変更しない。一方、
回路構造を失った平坦特徴だけではトポロジ学習の可否を判定できないため、2026-10-01から次の独立した
ML更新系列を開始する。

1. 構造化回路を `component-terminal-net` とsource/load/ground portで表す。SPICE文字列を推測して変換しない。
2. ngspice結果から、複素ノード電圧・枝電流・ポート量と工学指標を役割付きコーパスへ保存する。
3. 同じ分割で定数、木/MLP、関係別小型GNNを比較する。モデルの複雑さではなく保留性能で選ぶ。
4. 素子値保留、負荷/周波数保留、トポロジfamily保留、独立測定系列を別々に評価する。
5. in-domain不確かさと分布外判定を表示し、上位候補は必ず通常のngspice経路で再評価する。

GNNの主targetは合否だけでなく、再利用可能な複素電圧・電流・インピーダンスとする。反射、電力、stress、
制約余裕は既存analysis/metricsが同じ式で算出し、学習層へ物理判定を複製しない。Graph Transformer、
DeepONet、生成モデルは、小型GNNまたは波形latent baselineで不足が測定された後の比較対象とする。

第一変更の評価対象は「一つの確定済みcase内で未見の固定設計を予測する」に限定する。完全な
Candidate x Scenario x Control表では、同じ固定設計と同じscenario系列の両方をtrain/test間で禁止すると
全行が一つの連結groupになり、保留評価が成立しない。以前の「同一case・測定系列を跨がない」という
記述はcross-case汎化とwithin-case sizingを混同していたため修正した。cross-caseまたは独立測定系列への
汎化は、明示的な系列IDとcase固有の物理特徴を表へ追加し、別の保留設計として検証するまで主張しない。
この変更により、現行のsolver、評価表、候補選択には分岐を追加せず、ML側の妥当性主張だけを狭くした。

### 8.4 配置・接続最適化

「配置」を二つに分ける。

- 回路図上の接続・トポロジ選択: 本基盤の将来範囲。
- 物理レイアウト、配線寄生、EM・熱配置: ngspice 集中定数基盤だけでは完結しない。

PCD v1ではL/π/T、多段、harmonic trapなど、宣言済みで検証済みのテンプレート集合を離散候補として
比較するところまでを「配置・接続最適化」とする。小さい候補集合はMLを使わず完全列挙する。
`CircuitGraph` は生成器ではなく、宣言済み回路を学習データへ渡す共通表現として導入する。端子規則、浮遊ノード禁止、
短絡禁止、受動性を満たす新規トポロジの生成や、ML/RLによる任意グラフ生成は引き続き完成条件から外す。
配線指紋は宣言済みfamily内の変更検知に使い、一般のグラフ同型判定とは主張しない。

## 9. クリーンアップ方針

### 今回削除・簡素化するもの

- 廃止済み `--strict-exit`。失敗時非ゼロが既定なので意味がない。
- `simulation_record.v2` に重複していた旧フラット成果物キー。
- v1 保存形式を暗黙探索するフォールバック。
- 旧 `analysis.py` の単一file実装。責務別packageと薄いfacadeへ置換し、重複本体を残さない。
- 旧 `visualize-response` と `response_plot.py`。保存済みrunの計測と標準図を一つの `analyze` に集約した。
- 古いコミット固定の巨大仕様書と古いレビュー deck 版。
- ローカル図・deck 生成一時ディレクトリが Git 状態へ出る問題。

### 残すもの

- ngspice バージョン差を吸収する `wrdata` 読込み。
- 基準面、単位、peak/RMS、電流方向。
- 周期整定、電力収支、素子ストレス。
- Candidate/Scenario/Control の役割分離。
- 入力・ソルバー版・ネットリストの再現情報。ただし一つの debug manifest へ集約する。
- 実 ngspice と独立式に基づく小さな回帰ベンチ。

### P2の参照調査による判定

- 選択candidate詳細は、P1で条件別manifest・波形・netlist・solver logへ到達する単一証拠として
  `best_candidate.json` に残した。全候補の比較は既存 `study_history.json`、全Candidate x Scenario x Control詳細は
  `evaluations.csv` へreaderを移行できたため、非選択candidate JSONと候補directory探索を削除した。
  raw cacheと公開generationは引き続き混同しない。
- `pcd.figures` は確認時点ですでに再利用する回路geometry/visual grammarだけを所有し、benchmark固有の解釈は
  `bench/figures` に分離済みだった。現行figure生成と検査が利用しているため削除対象から外す。
- 2,295行の個別JSON報告書生成器は、core/literature suiteの既存summaryだけを読む372 SLOCの投影へ置換した。
  独自UI widget/schema、B5だけのcandidate読取り、質問別の重複説明を削除し、Markdown、compact JSON、
  core/literature CSVへ整理した。benchmark再現と設計可否は別列のまま維持した。
- 回路・metric pluginはadvanced利用例から到達し、型付きsolver登録もngspiceとtest doubleが使用しているため
  残す。一方、旧4引数solver呼出しだけは現行例に不要だったため削除した。

削除判定は、CLI/API 到達性、リポジトリ内参照、テスト、実 ngspice の四点で行う。

### 診断・契約・テストの簡素化

- 利用者向け状態は `RunOutcome` 相当の一種に集約し、成功、入力誤り、solver失敗、計測不成立を
  短いcodeとmessageで示す。solver実行状態と設計制約の `evaluation_success` を同じ概念にしない。
- 入力manifest、simulation record、solver diagnostic、evaluation resultに重複するフィールドを正規化し、
  利用者summaryと一つのdebug manifestに分ける。内部key一覧は既定出力しない。
- deep-freeze、`to_dict/from_dict`、有限値の再帰検査は、永続化境界と外部入力境界だけに限る。
  内部の全dataclassへ同じ防御コードを持たせない。
- `plan.py`、`validation.py`、`__post_init__`、rendererの同一チェックを一つに統合し、変換後の内部型を
  再検査しない。基準面、単位、受動性、素子定格など工学的意味のある検査は残す。
- 削除したschema分岐、診断文言、cache内部配置のtestも同時に削除する。物理不変条件、公開入出力、
  実ngspice回帰を残し、private helper単位のtest一対一固定は避ける。

## 10. 完成までのクリティカルパス

これ以降は次の順序を変えない。前のgateが完了するまで、後段の新機能を追加しない。

| 順位 | 完了gate | 実施内容 | 完了判定 |
|---:|---|---|---|
| P0 | 範囲固定 | 本節、利用者コマンド、対象外、ML停止条件を文書とREADMEで一致させる | 実装済み・未実装・対象外が一つの表で矛盾しない |
| P1 | 利用者経路の受入 | netlist/明示case解析、RF study、準静的R+jX、外生R(t)、目的波形最適化を実ngspiceで一巡する | 各workflowが一つの入口から完了し、主要結果・単位・基準面・合否・再現pathを短いsummaryで得られる |
| P2 | 基盤の収束・削除 | P1で使われない旧互換、重複成果物、到達不能plugin面、重複diagnostic/testを参照調査後に削除する | 通常出力とdebug証拠が分かれ、追加機能のownerと編集箇所がarchitecture文書と一致する |
| P3 | ML数値sizingの実証 | 既評価候補だけのretrospective rankingを先に行い、通過時だけ一つの逐次提案loopを実装する | 下記の固定基準を満たし、選択候補を全Scenario/Controlのngspiceで再評価できる |
| P4 | release closure | README/input/architecture/例/変更履歴を実挙動へ同期し、全品質gateとrelease受入suiteを実行する | 完成条件ごとのYES/NOと、回路基盤・ML・当初全範囲のGO/NO-GOを証拠付きで確定する |

P3不合格後にP4の完了条件を「全項目をYESにする」から「YES/NOを隠さず確定する」へ変更した。旧条件のままでは、
固定済みML停止条件を破ってモデルを追加するか、P4を永久に未完了とするしかないためである。この変更は製品範囲を
広げない。派生影響としてrelease結果を、回路基盤、ML候補提案、当初全範囲の三判定へ分離し、MLのNOを回路基盤の
PASSで上書きしない。

### 10.1 P1で固定する五つの受入workflow

1. 外部SPICE deckまたは明示caseのAC/tran解析から、canonical表と標準図を再生成する。
2. R+jXの複数条件に対し、既定match-network候補と全tuner controlを評価し、失敗制約とcoverageを示す。
3. 周波数sweepから共振、-3 dB帯域、loaded Q、Smith/impedance/powerを、成立する場合だけ示す。
4. 準静的R+jX profileと外生R(t)を、それぞれ独立snapshot・一方向時間変動として再現する。
5. 目的波形に対する素子値探索を行い、可否優先、目的値、Pareto、seed、最終ngspice証拠を残す。

P1では新しい指標やmodelを増やさない。既存機能を利用者の開始点から最終成果物まで通し、欠落・重複・
分かりにくい出力だけを修正する。回路解析基盤としての完成を、ML実験より先に確認するためである。

### 10.2 旧P3の固定合否基準（履歴）

旧P3のMLは固定トポロジの連続素子値候補順序だけを扱った。実装前に二つ以上のcase、初期観測、候補pool、seed、
品質閾値を固定し、後から都合よく変更しない。完全評価で得た可行候補の上位10%以内を発見するまでの
実ngspice評価数について、固定乱数順baselineの中央値より30%以上少ないことを合格基準とする。
最終候補は必ず通常の全Scenario/Control経路で再評価する。

基準を満たさない場合は、モデルを複雑化したりcaseを選び直したりしない。`ml-prepare` / `ml-evaluate` を
offline評価機能として残し、「MLによる評価削減・候補提案は未提供」と明記してP3を未完了にする。
通過した場合だけ、`fit -> rank -> ngspice evaluate -> append` の小さな逐次loopを追加する。
不確かさ取得関数、Bayesian optimization、任意topology生成はPCD v1の完了条件に含めない。

### 10.3 現在位置と次の一手

| gate | 状態 | 判断 |
|---|---|---|
| P0 範囲固定 | **完了** | 本文、README、architecture、変更履歴でv1範囲・実コマンド・対象外・ML停止条件を同期した |
| P1 利用者経路の受入 | **完了** | 五workflowを `bench/release/README.md` に固定し、7入力の厳格検証と実ngspice-46 E2Eを通した |
| P2 基盤の収束・削除 | **完了** | 旧互換、巨大report UI、非選択candidate JSONを削除し、summary/history/evaluation table/選択証拠の責務を固定した |
| P3 ML数値sizing | **判定完了・不合格** | 固定RC/RF回顧rankingは14.3%/29.6%削減で30%基準未達。逐次proposalを追加しない停止条件を適用した |
| P4 release closure | **完了** | 公開CLI release suiteの131実ngspice評価・0 cache hitと品質gateを通し、回路基盤GO、ML候補提案NO-GO、当初全範囲NO-GOを確定した |

P1で利用者経路を確定した後、P2の収束・削除を完了した。第一change setでは旧run manifest、candidateの
旧selected複製、旧4引数solver署名を削除し、第二change setでは巨大report UI契約を既存suite summaryの
単純な投影へ置換した。第三change setでは探索中のcandidate永続化を実行pipelineから外し、非選択candidateの
詳細を完全評価表へ、候補単位の比較を既存historyへ移した。P1/P0の入口、標準表・図、選択候補から実ngspice
証拠へ至るpathは維持し、参照・Import Linter・test・実ngspice 1候補/3候補benchmarkで確認した。
P3の停止基準を維持したままP4を完了した。`bench/release/run_suite.py` は公開CLIだけを使い、7入力をstrict検証し、
新規run rootで五workflowを再生する。2026-09-30の実行は131件すべてsolver成功、cache hit 0で、意図的な
不適合caseもnamed constraintとcoverageを再現した。P3結果のprotocol hashと独立再検証も確認し、回路解析・
決定論的最適化基盤をGO、本依頼の「ML候補提案」と当初全範囲をNO-GOとした。

P0からP4は履歴として閉じる。以後は新しい探索手法を増やす前に、次の一方向の責務を維持する。

1. **計算**: `simulation` / `solver` が入力をngspiceへ渡し、canonical波形を返す。
2. **測定**: `analysis` / `metrics` が波形から工学指標と符号付き制約余裕を一度だけ求める。
3. **学習**: `pcd/ml` が保存済み評価表だけを読み、予測品質をofflineで検証する。solverやstudyを呼ばない。
4. **最適化**: `search` が候補だけを提案し、`study` が通常の全Scenario/Control計算で候補を評価する。

現時点ではGP、Bayesian optimization、新しい空間充填optimizerを追加しない。符号付き制約余裕は、可否だけでは失われる
「成立側の残り余裕」を結果・学習・探索へ同じ物理単位で渡すため残す。これは新しい計算経路ではなく、既存計算結果の
共通fieldである。新しいML更新もこの四境界を変更せず、まず保存可能な回路グラフとコーパス契約を作る。

### 10.4 ML更新系列の現在位置

旧P0〜P4は履歴として閉じたままにし、番号を再利用しない。追加計画は次の順序で進める。

| 順位 | 実施内容 | 完了判定 | 状態 |
|---:|---|---|---|
| M0 | 正規回路グラフ | 構造化テンプレートをcomponent-terminal-netとportへ決定論的に変換し、配線と値の指紋を分離する | **完了** |
| M1 | PCD専用コーパス | ngspiceの複素電圧・電流・ポート量を回路、周波数、負荷、scenarioと対応付け、リークのない分割を作る | **完了** |
| M2 | 同一条件モデル比較 | 定数/表形式baseline、MLP、小型関係別GNNを同じsplit・指標で評価する | **完了・進級不合格** |
| M2R | 独立corpus再検証 | 未使用のtopology/frequency/load群を固定し、学習modelがconstantを集計と過半数foldで上回る | **次** |
| M3 | 推論・不確かさ | 保存model、in-domain区間、OOD判定を実装し、予測だけで合否を確定しない | M2Rまで保留 |
| M4 | ngspice削減実証 | 固定protocolで上位候補を通常studyへ戻し、品質とsolver call削減をbaseline比較する | 未着手 |
| M5 | 過渡波形 | AC経路成立後に、GNN encoderと波形latent baselineから評価する | 未着手 |

M0はML frameworkへ依存しない。M2まで公開optimizerや候補提案を変更していない。M2ではGNNが単純baselineを
保留トポロジ集計で上回らず、最良はconstantだったため、保存対象modelなしと判定した。このtestを見た後の
hyperparameter変更は同じ証拠へ採用しない。M2Rは新しい固定corpusで行い、再び未達ならML候補提案を終了する。

M2Rは、(1) 既存の構造化回路入力だけで独立topology・frequency・load群とprotocol hashを結果を見る前に固定し、
(2) 現M2を開発dataとしてtraining/internal-validationだけでmodel容量を抑え、(3) 新しいholdoutを一度だけ評価する。
合格は、topology集計でconstantを上回り、topology foldの過半数で勝ち、design/condition集計でもconstantより悪化
しないこととする。失敗時は終了、成功時だけM3を解禁する。M2 corpusはcomponent観測を持たないため、現結果を
素子stress予測の根拠にはしない。派生変更は `bench/ml` のdata/protocolと純粋評価gateへ限定し、solver、study、
search、公開回路計算経路は変更しない。

### 10.5 実装履歴（完了判定の根拠であり、次作業一覧ではない）

2026-10-01時点:

- 装置用途に近い追加証拠として、40 MHz上部励振と800 kHz下部矩形biasを持つ2周波CCP問題を固定した。
  装置と等価回路を同じ上部電極→上部sheath→plasma bulk→下部sheath→waferの縦方向で対応させ、出力を
  `V_W(t)`（下部電極／wafer chuck対chamber ground）と、上部50 ohm面の反射電圧波
  `V_-(t)=(V_P-Z_0 I_P)/2` に分離した。独立charge ODE（62.5 ps RK4）とngspice 46（最大250 ps）の順問題は、
  `V_W` でRMSE 0.05274 V / nRMSE `1.7167e-4`、`V_-` で0.01948 V / `3.9890e-4` だった。
  逆問題は自由関数推定へ広げず、節点時刻を固定した正値piecewise-linear `R_p(t)` の4値だけを各3候補、
  計81点で完全列挙した。`V_W` 単独と `V_-` 単独の二studyはいずれもsolver失敗・cache hit 0で
  `70/24/30/65 ohm` を回収し、順位付けに使わない他方の波形もheld-outとして一致した。装置図、順問題、
  二つの逆問題を4枚のSVG/PNG/PDFへ集約した。これは既存のtransient回路・deterministic study境界を利用する
  benchmarkであり、production schema、solver、search、MLの責務とM2R優先順位を変更しない。
- 基本計算の第三者向け証拠を4枚へ集約した。外部 `.cir` からcanonical過渡波形までの公開経路、外生R(t)の
  分圧電圧・素子電流、CCP/ICP端子等価回路の解析式対ngspice、既知RC時定数を回収する9点完全列挙の
  目標波形・loss履歴・素子値gridを、保存済み成果物だけからSVG/300 dpi PNG/PDFへ生成する。
  `figure_data.json` は入力hashと直接比較誤差を保持し、図生成器はsolver・study・searchを呼ばない。
  plasma図は端子等価回路conformanceに限定し、plasma状態検証とは明記して分離した。これは可視化証拠の
  追加であり、計算・学習・最適化の境界やM2R以降の優先順位は変更しない。
- M2を完了した。`ml-corpus-evaluate` はcorpus hashを確認し、訓練平均、flat graph/context ridge、2 hidden layer
  MLP、component/netとp/n方向を持つ2 step relational GNNを、corpusが宣言した同一splitで比較する。
  filesystem adapter、graph encoding、NumPy neural計算、fold/metric判断を分離し、solver・study・optimizer・
  model保存への依存を追加していない。source電圧の定数targetと欠損load-current targetはfold評価から除外し、
  複素source電流とload電圧の4 targetを単位差のないtrain標準偏差基準RMSEで比較した。
- M2 evidenceとして、ngspice 46で `l_match`、`pi_match`、`pi_match_harmonic` を各12設計×共通3負荷条件、
  合計108件すべてsolver成功で実行した。36 graph/108 sampleでdesign、condition、topology splitはいずれもready。
  mean normalized RMSEはdesignで constant 0.534 / ridge 0.592 / MLP 0.763 / GNN 0.801、conditionで
  0.289 / 0.460 / 1.479 / 1.535、leave-one-topology集計で 1.273 / 1.317 / 1.638 / 1.463だった。
  GNNはL-match保留foldだけ最良だったが集計ではconstantが最良で、`advance_to_model_persistence=false` とした。
  topology foldのtrainは72行に対してMLP 940〜1,012 parameter、GNN 4,556 parameterで、train lossは小さい一方
  保留誤差が大きい。主因をsample/topology多様性不足と過剰capacityと整理し、既に見たtestで調整せずM2Rへ進む。
- M1を完了した。`ml-corpus` は複数の完了済みstudyだけを読み、ngspiceを再実行せず、物理回路
  `graphs.jsonl`、周波数・source/load/scenario条件と複素port応答の `samples.csv`、論理素子ごとの
  複素V/Iをlong形式で持つ `component_responses.csv`、役割・hash・coverage・split状態を持つ
  `manifest.json` を出力する。設計splitはtopologyごとに二つ以上の設計groupを要求し、条件splitは
  train/test双方に同じtopology familyが存在しなければ利用不可にする。小さいdataを行単位で混ぜて
  見かけの精度を作らない。`pcd/corpus.py` はstudy探索・同一性・provenance・保存だけを担当し、
  応答検証、target整形、group/split規則はfilesystemとstudy実行へ依存しない `pcd/ml/corpus.py` に分離した。
- M0の物理境界を観測付きcaseへ拡張した。構造化builderは論理素子とngspice renderingを分け、0 V観測源と
  ESR/DCR内部nodeをgraphから除外する一方、直列損失値は論理素子属性としてinstance fingerprintへ残す。
  raw SPICE、磁気結合、外生時間profileは推測せずgraph-unavailableとする。実ngspice 46でpi-match stressと
  L-match goldenの2 studyを新規実行し、2 graph、2 AC sample、5 component行（3行は観測targetあり）を
  一つのcorpusへ統合した。二つのtopology familyは識別でき、データ不足のdesign/condition splitは利用不可と
  明示される。通常のsolver、study、optimizer、合否判定は変更していない。

- P3不合格後の整理として、`ConstraintResult` に物理単位の符号付き `margin` を追加し、metric上限/下限と
  control余裕で同じ正負規則を用いた。完全評価表では行単位、study historyでは選択scenario中の
  最悪値、ML datasetではresponse targetとして同じ値を再計算せず利用する。新しいoptimizer、確率model、依存package、
  専用実行経路は追加していない。
- P4は完了。`bench/release/run_suite.py` を追加し、solver診断、7入力のstrict検証、`sim-run -> analyze` と
  `run`、成果物・coverage・failed limit・cache確認を一つの外側向きrunnerへ統合した。131件の実ngspice評価を
  新規rootで再生し、全workflow PASS、cache hit 0だった。P3証拠を添付した最終判定は回路基盤GO、ML候補提案と
  当初全範囲NO-GOである。runnerはsolver/study/analysis実装をimportせず、公開CLIと返却pathだけを読む。
- P1は完了。`bench/release/README.md` に五つの利用者workflowと再現コマンドを固定し、7入力をstrict検証した。
  ngspice 46で、8,008点の周期過渡（load 24.9998 W）、Candidate×Scenario×Controlの2×2×2完全評価、
  8,001点掃引（共振13.88725 MHz、帯域1.039898 MHz、loaded Q 13.3544）、5 snapshot×3 control、
  258点の外生R(t)、24候補の目的波形DEを実行した。意図的な不適合caseもsolver失敗と混同せず、
  failed constraintとcoverageを保持した。study要約へ解析種別・固定周波数・基準面・Z0と選択候補証拠pathを
  追加し、非周期R(t)は架空の1 Hz未整定ではなく `not_applicable` とした。raw SPICE文字列だけだった受入例は
  構造化素子へ置き換え、接続警告を0件にした。新しいmodel・metric・schemaは追加していない。
- 段階0は完了。全test、quality gate、architecture audit、実ngspice smokeを通過した。
- 段階1は完了。`AnalysisRequest`、`SimulationResult`、`ProbePlan`、`SolverSettings`、
  必要最小限の `ResolvedSimulationCase` を導入し、ngspice I/Oをmeasurementから分離した。
- 内蔵ngspice adapterはCaseではなく `SolverRunRequest` を受ける。netlistとsolverは同じ解決済み入力を使う。
  当初一か所に限定した外部旧solver pluginの互換分岐は、P2で利用参照がないことを確認して削除した。
- 段階2は完了。旧 `analysis.py` を削除し、AC、過渡、stressの純粋計算と薄いfacadeへ
  置き換えた。保存済みrunから `summary.json` と標準図を再生成する `analyze` を追加し、包含された
  `visualize-response` と `response_plot.py` を削除した。計測結果とstudyの合否判定は分離したままである。
- 現在のquality-prは704件、branch coverage 93.06%、Ruff違反0、Pyreflyエラー0（移行対象warning 62）、
  Import Linterは72モジュール・206依存に対する7規則すべて成功した。Radonは891 blockの平均A 4.09、
  review対象のC以上は34 blockである。新しいM1/M2モジュールはC以上0、MI Aである。
  `validation.py` はMI Bへ改善済みで、source/probe解決のC関数は一つの `ProbePlan` を作る凝集した処理のため、
  点数だけを目的に分割しない。
- ngspice 46で単一点ACと8周期の過渡を実行し、保存済み成果物だけから反射係数0.48493、Smith図、
  周期整定、高調波、電力・効率、波形図を再生成した。過渡ケースの周期残差は2.25e-15で、
  source 49.9997 W、load 24.9998 W、network loss 24.9998 Wを得て電力経路も確認した。
- 素子stress caseでも保存結果から |Gamma|=0.01104、source 100.619 W、load 98.204 W、network loss
  2.41441 Wを再計算し、model化した素子損失との差は2.08e-11 Wだった。C1/L1/C2のpeak VとRMS Iも
  標準図へ出力したが、定格に対する合否は意図どおり `analyze` では判定していない。
- 既知解を持つ直列RLCの7,001点実ngspice掃引では、理論loaded Q 8.5200に対し8.519998、
  半電力帯域1.59155 MHzを得た。両側の交点がない掃引ではQを出さないtestも固定した。保存済み過渡
  波形からはH1基準の電圧・電流高調波図を再生成した。
- 段階2bは完了。直接simulationのsummary/data/debug分離を実装し、rootの一次項目を3つにした。
  重複 `params.json` とsolver一時CSV/logは残さず、当初readerへ限定した旧run互換もP2で削除した。
  ngspice 46の実過渡計算と、その保存runからの `analyze` 再生成に成功した。
- 段階2bの第二change setとして、`solver.ngspice_cli`、`sim_core.simulate_case`、`core.StudyRunner` の
  失敗境界を整理した。solverが明示する実行失敗と未整定計測は結果として残す一方、設定不備、目的指標の
  欠落、metric衝突、parser/solver/evaluator実装例外は隠さず送出する。旧 `_record_failure` と空波形生成、
  二重のtraceback診断を削除し、`--allow-failure` が入力不備を迂回しないtestを追加した。
- 段階2bの第三change setとしてstudy保存を実測し、書込み専用 `evaluations/` とevaluation cache ID、
  candidate内のselected評価二重保存、回路評価の重複observationを削除した。candidateと完全評価表をv2に更新し、
  旧candidateのreader分岐はP2で削除し、`selected_trial` に一本化した。再開処理で使われていなかった最終
  `optimizer_state` の重複も公開summaryから削除した。raw cache、公開generation、完全評価表という異なる責務は
  維持した。
- P2の第一change setとして、production/example成果物に参照がない旧root `sim_manifest.json`、旧candidate
  `selected` 複製、旧4引数solver callableを削除した。test fakeも保存済み現行manifestから入力を読み、
  ngspiceと同じ `SolverRunRequest` を受ける。現行の回路・load・metric plugin、candidate証拠、評価表は利用中と
  確認できたため削除していない。
- P2の第二change setとして、2,295行の個別報告書生成器と244 kBの独自widget JSONを削除した。既存の
  core/literature suite summaryを唯一の入力とし、source hashで同定したMarkdown report、compact JSON、
  二つのflat CSV（合計約28 kB）へ置換した。candidate/raw solver artifactを再読せず、benchmark再現、
  engineering feasibility、apparatus qualificationの意味を混同しない一つのbehavior testを追加した。
- P2の第三change setとして、探索中に全candidate JSONを保存する責務を `StudyRunner` から削除した。
  全候補比較は `study_history.json`、全Scenario/Control詳細は `evaluations.csv`、条件別solver証拠は選択後に
  一度だけ書く `best_candidate.json` とした。core benchmark、figure、Lee/Colpo/GEC readerをこの契約へ移し、
  旧candidate directory APIを削除した。ngspice 46の1候補caseと3候補完全列挙caseはいずれもbenchmark PASSで、
  3候補時もhistory 3件・評価表3行・選択JSON 1件となることを確認した。
- 段階2bの第四change setとして検査ownerを一覧化し、solver timeoutとAC/tran解析条件を
  `SolverSettings` / `AnalysisRequest` へ集約した。`validation.py` の並行parserと細分化診断を削除し、
  不正な明示timeoutの暗黙default化も廃止した。入力解決前にはrun directoryを作らない。
- 段階2bの第五change setとしてsource container/name、probe列、measurement/load mappingのshape検査を
  `simulation_input.py` の一回の変換へ集約した。netlistも同じsource resolverを使い、`validation.py` には
  周期窓、reference plane、loadとの測定接続などsection横断の工学的意味だけを残した。
- 段階2bの第六change setとしてcircuit/load builder入力を整理した。実際のbuilderがmodel必須parameter、
  物理値域、YAML component、外部netlist方針、出力型を所有し、`validation.py` の重複実装を削除した。
  public RF、advanced Case、plugin経路は同じ `NetlistInputs` を通り、無効入力は成果物確保前に停止する。
- 段階3の第一change setとして準静的snapshotを実装した。公開 `impedance_profile` は時間順の
  `time_s, resistance_ohm, reactance_ohm` を既存の一点AC scenarioへ写像し、最良固定候補について
  load/input R/X、|Gamma|、進行・反射・受電電力、選択control、合否を `snapshot_response.csv` に保存する。
  時刻の有限性・非負・狭義単調性、受動性、周波数、絶対drive、基準面を入力境界で検査し、点間の
  動的状態や自己整合plasmaは主張しない。5時刻×3controlの実ngspice 46計算は15評価すべて成功し、
  π型回路の閉形式解と入力R/X、|Gamma|、進行・反射電力が一致した。宣言した `|Gamma| <= 0.5` は
  5時刻中3時刻だけ合格であり、solver成功と工学的合否を分離できている。
- 段階3の第二change setとして、advanced `from_yaml` componentの外生 `R(t)` profileを実装した。
  一つのcomponent入力ownerがCSV列、時刻0始点、狭義単調時刻、正抵抗、linear/hold、周期終端、最短区間と
  solver刻みを検査する。ngspice 46の258点過渡では50 ohm直列抵抗との分圧閉形式解に対する最大誤差が
  3.6e-15 Vで、入力CSVのarchive/replayとfingerprint変化も確認した。変更はcomponent入力、artifact、
  netlist、過渡解析だけで、study/core/optimizerには新しい分岐を追加していない。
- C(t)/L(t)は物理定義のない値置換を避けるため条件付きbacklogへ移し、現時点の段階3を完了とする。
- 段階4の第一change setとして、bounded continuous advanced case向けにseed再現可能な
  `DE/rand/1/bin` を実装した。制約penaltyは追加せず、既存の可否優先rankで世代交代し、探索履歴には
  proposal世代・親・採否、目的、失敗制約・評価、coverage、seed、時間を明示した。公開RFの完全列挙と
  control選択規則は変更していない。全候補が既に全scenarioを実solverで評価するため、証拠が同一になる
  選択後の重複solver実行は追加しなかった。
- 段階4の第二change setとして、完全なcontrol solver証拠と全scenario可行性を持つ候補だけから、目的方向を
  考慮した `pareto_front.csv` を生成した。24候補の実ngspice 46計算は失敗0、適格21候補、観測front 17候補で、
  辞書式bestもfrontに含まれた。追跡誤差は0.527345、peak出力は3.97504 Vだった。連続探索は
  `observed_candidates` と明記し、完全列挙gridと区別した。
- 以上で段階4を完了とする。
- 段階5の第一change setとして、モデル追加より先に純粋な `pcd/ml/dataset.py` と `ml-prepare` を追加した。
  確定済み `evaluation_table.v2` の世代identity、件数、Candidate x Scenario x Control粒度、成功行の有限目的値を
  一つの境界で検査する。出力manifestはdesign/scenario/control入力、目的・補助応答、制約label、失敗行、
  除外列を明示し、`selected_control`、cache、時間、artifactを特徴から除外する。同一設計値を持つ行は
  candidate IDが違っても同じdeterministic groupへ置き、失敗行は削除せず目的回帰からだけ除外する。
  solver成功、feasible、各制約labelのtrain/testクラス数もmanifestへ出し、片側に違反例がない分割を
  校正済み分類器として扱わない。
  ファイル探索・書込みはCLIに残し、`pcd/ml` からsolver、study orchestration、保存へのimportを
  Import Linterで禁止した。scikit-learn等の新しい実行依存は追加していない。
- 段階5の第二change setとして、純粋な `pcd/ml/evaluation.py` と `ml-evaluate` を追加した。
  訓練平均/多数派を必須基準とし、宣言済みlog/linear変換後の数値特徴だけを使う固定3近傍モデルを、
  第一change setの設計group holdoutから再分割せず評価する。出力はtest行ごとのactual/baseline/surrogateと、
  目的別MAE/RMSE/R²、分類混同行列、両classが存在する場合だけのbalanced accuracy、実行時間、明示的な
  evidence gateである。キャッシュ済み評価時間はsolver費用比較から除外し、categoryの暗黙encoding、
  hyperparameter探索、候補提案は追加していない。
- 24候補の既存ngspice 46結果では、6候補holdoutに対して normalized RMSE のRMSEが0.20222から
  0.0380805（81.2%改善、R² 0.961）、peak voltageが2.07341 Vから0.303880 V（85.3%改善、
  R² 0.971）となり、応答近似は平均基準を上回った。model評価は今回7.0 msで、未cache一評価中央値76.2 ms未満だった。
  ただし違反3候補がすべてtrain側でtest側は可行6/6のため、100% accuracyを制約分類性能とは扱わない。
  ngspice評価削減も未測定なので `bayesian_optimization_ready=false` を維持する。
- 段階5の第三change setとして、固定holdoutを変更せず別の確定済みdatasetを制約検証にだけ渡せる
  `ml-evaluate --constraint-validation` を追加した。feature/target roleと変換の一致、異なるdataset identity、
  固定設計値の非重複を必須とし、標準化・3近傍fitは元datasetのtrain splitだけで行う。検証datasetの
  splitはfitに使わず全行をscoreし、行別予測を別CSVへ保存する。これによりsolver/study/persistenceを
  `pcd/ml` へ依存させず、固定holdout評価と外部制約評価の責務を分けた。
- 検証候補は結果確認前に `generic_rc_constraint_boundary.yaml` の5×5固定gridとして宣言し、実ngspice 46で
  25/25評価成功、可行15・違反10を得た。元の18 train候補だけでfitした分類器は、feasibleと4.9 V制約の
  両方でaccuracy 0.84、balanced accuracy 0.80（多数派基準はaccuracy 0.60、balanced accuracy 0.50）、
  TP 15、TN 6、FP 4、FN 0だった。元holdoutの回帰指標とseed=23は変更していない。
- 既存結果を見た後にleave-one-design-group-outを追加する案は、事前固定した独立証拠と誤認されるため採用しなかった。
  制約境界の外部simulation証拠は通過したが、この時点ではngspice評価削減は未測定だったため
  `bayesian_optimization_ready=false` を維持する。
- 段階5の残作業はP1/P2完了後のP3で、既評価候補だけを使う事前固定のretrospective candidate rankingとして
  再開した。完全評価と同じ可否優先・目的方向・Scenario x Control集約を保ち、10.2の固定基準で
  ngspice評価削減を判定し、削減を再現できない場合は逐次提案、不確かさ取得関数、BOを追加しない方針とした。
- P3では `ranking_protocol.yaml` に結果確認前の二つの81候補pool、各6初期観測、3近傍、seed、上位10%、
  random 101反復、30%閾値を固定した。`ml/ranking.py` は候補単位の純粋な回顧順序だけを所有し、solver、study、
  保存をimportしない。case実行と選択候補の別run再検証は `bench/ml/run_ranking_benchmark.py` が所有する。
- ngspice 46でpool 162評価と選択候補2件のfresh再評価はすべて成功した。RCは12対14評価で14.3%削減、
  RFは19対27評価で29.6%削減となり、いずれも30%基準に届かなかった。プロトコルhashは
  `368adca9a59520b59ecf005f303de9576555cb912751e065c2618aad1c9b87bb` である。P3を不合格とし、
  seed・pool・model・閾値は変更せず、逐次提案、BO、評価削減claimを追加しなかった。
- 2026-10-01の回路検証として、60 MHz上部RF、2 MHz下部RF、負DC pulse、matching/feed/choke/blocking、
  electrode/ESC stray、wall loss、時間変化する `Rp/Lp/Cs,u/Cs,w` を持つcase-local etch-CCP benchmarkを追加した。
  C/Lは `q=C(t)v` と `phi=L(t)i` を明示するraw ngspice elementに限定し、汎用plasma modelの責務や段階3の
  完成境界は変更していない。独立charge/flux MNAとngspice 46のwafer電圧RMSEは0.287151 V、反射電圧RMSEは
  0.364325 Vで、電流・charge・fluxも直接一致した。wafer電圧だけを使う3^4完全列挙は81/81完走し、
  `15 ohm / 160 nH / 520 pF / 720 pF` の真値を回収、未使用反射波もhold-out一致した。
- 同benchmarkの初期試行で、ngspiceがtime-step abort後に部分CSVとreturn code 0を残す挙動を確認した。
  solver adapterへ要求`stop_s`到達の一回の検査を追加し、部分波形を`incomplete_transient`として失敗させた。
  これは診断層の追加ではなく、simulation成功契約の偽陽性を除く一般修正である。

装置データの校正・モデル同定は別工程である。段階5の既存評価成果は保持するが、回路解析基盤の
利用者経路を閉じるP1/P2より先にML機能を増やさない。

Pyreflyのwarning 68件を一度に全体gate化しない。新しい型付き内部モデルと分割済みpackageから
warningを0にし、旧 `Any` 経路を移行に合わせて狭める。Radonも一律の点数gateにせず、
Dを責務分割候補、Cをreview対象とする。数値計算の一貫した式を点数のために分割しない。

## 11. 参照した一次資料

- [ngspice User's Manual 47](https://ngspice.sourceforge.io/docs/ngspice-47-manual.pdf):
  AC/tran/noise/sensitivity/pole-zero/S-parameter、behavioral R/C/L、`time` 変数。
- [ngspice documentation](https://ngspice.sourceforge.io/docs.html):
  公式manual・tutorial・release documentationの一次索引。
- [Storn and Price, Differential Evolution](https://doi.org/10.1023/A:1008202821328):
  連続空間の `DE/rand/1/bin`、差分mutation、binomial crossover、greedy selectionの原典。
- [NIST Technical Note 1354](https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1354.pdf):
  半電力帯域と `Q=f0/Delta f` の定義。今回の実装は保存された電力応答に対するloaded Qに限定した。
- [Coakley et al., Estimation of Q-Factors and Resonant Frequencies](https://www.nist.gov/publications/estimation-q-factors-and-resonant-frequencies):
  離散周波数応答からの3 dB法と、より高度なfit法の精度差。現段階では単純で監査可能な3 dB法だけを採用した。
- [Qu et al., Power matching to pulsed inductively coupled plasmas](https://doi.org/10.1063/5.0002522):
  pulsed ICP の時間変化 R/X、整合点、反射、電力経路。
- [Colpo, Ernst, and Rossi, Determination of the equivalent circuit of an inductively coupled plasma source](https://doi.org/10.1063/1.369268):
  fixture/dummy-load較正、複素インピーダンス測定、等価回路同定。
- [Hargis et al., Gaseous Electronics Conference RF reference cell](https://doi.org/10.1063/1.1144770):
  CCP装置の電気的比較条件と V/I/位相表。
- [Lee, Kwon, and Chung, impedance corrected to the plasma terminal](https://doi.org/10.1063/6.0000883):
  VI-probe、coax/fixture、plasma terminalの基準面分離。
- [RF source and matcher efficiency by voltage-current probing](https://doi.org/10.1109/TPS.2022.3154397):
  等価抵抗、供給電力、matcher・プラズマ間の効率分解。
- [Lee et al., A simple model of solenoidal inductively coupled plasma sources considering finite size](https://doi.org/10.1063/1.5133862):
  ICP 端子縮約モデル。
- [Schmidt et al., Consistent simulation of capacitive radio-frequency discharges and external matching networks](https://doi.org/10.1088/1361-6595/aae429):
  状態を持つプラズマモデルと外部回路の自己整合連成。
- [Lee et al., high-speed impedance measurement for pulsed plasma](https://doi.org/10.1063/1.4928121):
  時間分解端子インピーダンス測定。
- [Rauf et al., Effect of low frequency voltage waveform on plasma uniformity in a dual-frequency CCP](https://doi.org/10.1116/6.0001732):
  40 MHzと800 kHz矩形bias、直列blocking capacitorを持つ2周波CCPの装置・波形条件。
- [Kim et al., Effect of source frequency and pulsing on SiO2 etching in a DF-CCP](https://doi.org/10.7567/JJAP.54.01AE07):
  300 mm chamber、20 mm gap、上部13.56–60 MHz source、下部2 MHz biasの実験装置。
- [Yamaguchi et al., DC-superposed DF-CCP selective etching](https://doi.org/10.1088/0022-3727/45/2/025203):
  上部60 MHzへの負DC重畳、下部RF bias、-800～-1200 Vの電圧規模。
- [Kim, Lee, and Hong, CCP impedance monitoring through the matching unit](https://doi.org/10.3390/electronics14102022):
  sheath capacitance、bulk plasma inductance/resistanceとmatching networkからのload impedance整理。
- [Song and Kushner, Role of the blocking capacitor in pulsed DF-CCPs](https://doi.org/10.1116/1.4863948):
  上部HF・下部LF接続、pulse中のDC bias応答とblocking capacitor感度。
- [Press et al., Sub-rf period electrical characterization of a pulsed CCP](https://doi.org/10.1116/1.5132753):
  時間分解V/I・impedanceで必要なprobe較正、寄生、伝搬遅延の扱い。
- [NBS Monograph 137, Applications of waveguide and circuit theory](https://nvlpubs.nist.gov/nistpubs/Legacy/MONO/nbsmonograph137.pdf):
  実数基準impedanceに対する端子V/Iと入射・反射電圧波の変換式。
- [Schmidt et al., Multi frequency matching for voltage waveform tailoring](https://doi.org/10.1088/1361-6595/aad2cd):
  多周波整合と等価回路連携。
- [Budak et al., ML-assisted analog circuit sizing](https://doi.org/10.1109/TCAD.2021.3081405):
  代理モデル支援 sizing と予測誤差の課題。
- [AutoCkt](https://arxiv.org/abs/2001.01808): 固定回路のパラメータ sizing と SPICE 再評価。

アーキテクチャ監査ツールの仕様は公式文書で確認した。

- [Ruff settings](https://docs.astral.sh/ruff/settings/): lint/formatとMcCabe設定。
- [Pyrefly configuration](https://pyrefly.org/en/docs/configuration/): project include、severity、診断設定。
- [Import Linter contract types](https://import-linter.readthedocs.io/en/stable/contract_types/):
  forbidden/layers/independence契約の意味。
- [Radon documentation](https://radon.readthedocs.io/en/master/):
  cyclomatic complexityとmaintainability indexの解釈。

## 12. PCD v1完成条件

この基盤の価値は、保存ファイル数、スキーマ数、診断項目数では測らない。次の全項目がYESになった時だけ、
本依頼の目的まで完成したと判定する。P4はNOを隠さずrelease判断を閉じるgateであり、全目的の完成を意味しない。

| 完成条件 | 判定内容 | 現在 |
|---|---|---|
| 入力から結果まで単純 | `sim-run` / `analyze` / `run` のどれを使うかが明確で、一つの入力から主要結果と保存先を短時間で理解できる | YES |
| 回路解析として有用 | 波形、Z、Γ、電力、stress、共振・帯域・Q、高調波、制約余裕を成立条件・単位・基準面付きで読める | YES |
| plasma境界が正直 | 静的R+jX、準静的profile、外生R(t)を再現でき、自己整合plasmaや未実装C(t)/L(t)を主張しない | YES |
| 最適化が完全評価へ戻る | 素子値・検証済みtemplateの探索結果が、全Scenario/Controlの実ngspice証拠、制約、seed、履歴を持つ | YES |
| MLが実益を示す | P3の事前固定基準で評価数30%以上削減を再現し、提案候補を通常ngspice経路で確認する | **NO** |
| 責務が明確 | simulation、analysis、study/search、ML、reportが保存形式やCLI都合を逆importせず、追加時のownerが一つに定まる | YES |
| 不要物を残さない | P1で未使用と確認した旧互換、重複成果物、到達不能分岐とそれだけを固定するtestを削除する | YES |
| 再現可能 | release受入case、全test、品質gate、architecture auditが成功し、保存済み入力から最終候補を再実行できる | YES |

MLが固定基準を満たさない場合、回路解析・決定論的最適化基盤はrelease可能でも、本依頼で求める
「MLによる最適化」まで完成したとは報告しない。モデルを追加して見かけ上通すのではなく、原因と不足dataを残す。

coverage 93%は退行検知の参考値であり、分岐を増やしてtestを増やす目標にはしない。不要機能を
削除した結果としてtestも削除し、公開動作と工学的根拠のcoverageを優先する。
