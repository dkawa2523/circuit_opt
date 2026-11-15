# PCD 回路モデリング統合マニュアル

PCDは、指定した端子で回路の応答を計算し、波形・電力・反射などの数値を確認するためのツールです。素子値を選ぶ設計探索や、観測データから有効な負荷値を求める端子同定にも使えます。

まず[付属のRC回路で計算と描画を試す](#quickstart)。用途に合う設定が分かっている場合は[実行例の一覧](../examples/README.md)、計算が終わっている場合は[結果の見方](#results)へ進みます。

## 目次

1. [用途と適用範囲](#scope)
2. [実行環境と最初の計算](#quickstart)
3. [モデリングの手順](#modeling)
4. [入力の共通規則](#input)
5. [任意回路と外部SPICE](#circuit)
6. [整合回路の設計](#network)
7. [負荷モデルの選択と物理的意味](#loads)
8. [端子モデルの同定](#identification)
9. [計測面・数値計算・部品ストレス](#measurement)
10. [回路図・グラフ・数値表](#output)
11. [結果の見方と再開](#results)
12. [実行結果を確認する目安](#evidence)
13. [CLIとPython API](#commands)
14. [上級入力と拡張](#advanced)
15. [開発と再現検証（リポジトリ利用者向け）](#development)
16. [よくある問題](#troubleshooting)

<a id="scope"></a>

## 1 用途と適用範囲

| 求めたいこと | 入力と操作 | 得られる判断 |
|---|---|---|
| 既存回路の応答 | 素子列またはSPICEを`sim-run` | 指定時刻・周波数でのV/I、追加プローブ、電気的指標 |
| 多ポートの反射・伝達 | DUTと端子を宣言して[`network-run`](#multiport) | 共通のDC動作点における複素S行列と周波数ごとの算出可否 |
| RLC値を決める | `parameters.design`、目標、制約を指定して`run` | 評価した候補の順位、選択回路、制約余裕、再計算結果 |
| 負荷変動に耐える整合回路を選ぶ | `network`、負荷点、設備の調整値を指定して`run` | 条件ごとの調整値、反射、部品ストレス、全条件の合否 |
| 有効な負荷値を実測から推定する | 固定校正値、`latent`の範囲、fit/holdout観測を指定して`identify` | 推定値、未使用条件の誤差、局所感度、同定判定 |
| 保存結果を考察する | 成功した単体runを`analyze` | 計測結果、回路図、波形、伝達比、高調波と対応CSV |

PCDが扱う境界は電気的な端子・基準面である。CCP/ICPの有効回路は、その端子から見える損失やリアクタンスを表す。電子密度、化学反応、温度、イオンエネルギー分布、製造歩留まりを求めるには、それぞれを裏付ける別のモデルと観測が必要になる。

通常の反射・電力判定は観測したsource-portを基準とする。多ポート計測は、同じDC条件で励振を順に切り替えて線形化応答を測る。端子モデルの同定は、宣言した校正、回路パラメータ、複素V/I観測に基づく。

<a id="quickstart"></a>

## 2 実行環境と最初の計算

### 2.1 付属例で順計算と描画を試す

Python 3.10以上とngspiceが必要です。Windowsでは、PATH上にコンソール版`ngspice_con.exe`があれば優先されます。リポジトリのルートで、次のコマンドを上から実行します。

```powershell
uv run --frozen --no-dev python -m pcd solver-diagnose --json
uv run --frozen --no-dev python -m pcd validate-case examples/circuit_transient.yaml --plan
$run = uv run --frozen --no-dev python -m pcd sim-run examples/circuit_transient.yaml --run-root runs/first --json | ConvertFrom-Json
uv run --frozen --no-dev python -m pcd analyze $run.run_dir --json
```

`solver-diagnose`の`batch_runnable: true`を確認します。`validate-case --plan`は設定を解決して表示し、`sim-run`が波形を計算します。計算結果は`$run.run_dir`にあり、`summary.json`の`artifacts.waveform`から元波形を開けます。`analyze`が返す`output_dir`には図表と解析用の`summary.json`があります。

この回路の時定数は`RC = 1 µs`です。5 Vの理想ステップなら2 µs後は`5(1 − exp(−2)) ≈ 4.32 V`で、実行例には2 nsの立上りが含まれます。波形の立上りと最終電圧を確認してください。

実行ファイルがPATHにない場合は、`solver-diagnose --executable PATH --json`でその実行ファイルを確認します。任意回路のYAMLでは`execution.executable`、上級入力では`solver.executable`に同じパスを指定できます。

以降の`pcd`はコマンドの略記です。チェックアウトでは`uv run --frozen --no-dev python -m pcd`、インストールした環境では`python -m pcd`を使えます。

### 2.2 付属例から自分の設定を作る

[circuit_transient.yaml](../examples/circuit_transient.yaml)をコピーし、`parameters.fixed`のRとC、`circuit.components`の接続、`analysis.tran`の刻みと終了時刻を変更します。`case_id`を変えると実行結果を見分けやすくなります。

計算と同時に図を作るには、YAML末尾へ`output: {}`を追加します。プローブ名、グラフの種類、表示範囲を指定する方法は[回路図・グラフ・数値表](#output)を参照してください。立上りを見る短い過渡計算では、定常周期の電力や高調波評価に必要な信号範囲を別途確保します。

### 2.3 配布wheelから使う

wheelから利用する場合は、配布されたファイル名を指定してインストールする。

```powershell
python -m pip install PATH_TO_WHEEL
python -m pcd --help
```

wheelにはマニュアルと例一式が入ります。次のコマンドで同梱データの場所を調べ、その中の`examples`ディレクトリを丸ごと作業場所へコピーします。コピー先の親ディレクトリで例を実行すると、SPICEやCSVへの相対パスも解決されます。

```powershell
python -c "from pathlib import Path; import sysconfig; print(Path(sysconfig.get_path('data')) / 'share' / 'pcd')"
```

<a id="modeling"></a>

## 3 モデリングの手順

1. **判断したい量を決める。** 反射、出力波形、受入電力、部品ストレス、負荷パラメータのどれを判断に使うかを決める。
2. **基準面を決める。** ケーブル、治具、feedthrough、コイルを回路側と負荷側のどちらへ含めるかを記録する。同じ寄生を二重に含めない。
3. **観測に見合うモデルを選ぶ。** R+jXの一点しかなければ一点モデルから始める。多周波数の観測や校正情報がある場合に有効CCP/ICPモデルを使う。
4. **変更できる値を役割別に分ける。** 固定装置、校正値、外部条件、設備調整値、設計値、未知負荷値を分離する。
5. **解析とプローブを決める。** 定常線形応答にはAC、立上り・パルス・非線形応答には過渡を選ぶ。負荷電力には負荷端子のVと実際に流入するIを取得する。
6. **目的と合格条件を分ける。** 小さい反射や波形誤差を順位付けに使い、電流・電圧・損失上限を独立した制約にする。
7. **解決済み入力を確認して実行する。** `validate-case --plan`で役割・条件・解析・既定値を確認し、計算後は数値状態、工学的合否、出力完了を別々に読む。

同定時は観測した設備調整値を固定する。未知負荷と調整値を同時に自由化すると、調整によって観測誤差を隠し、負荷値の識別が難しくなる。

### 3.1 パラメータの役割

| グループ | 役割 | 実行中の扱い |
|---|---|---|
| `fixed` | 既設回路や選定済み値 | 固定 |
| `calibration` | 校正済みの治具・寄生 | 同定でも固定 |
| `operating` | 周波数、外部条件 | 条件行として与える |
| `control` | 設備で操作できる値 | 設計では条件別に選択。同定では観測値へ固定 |
| `design` | 設計で決める素子値 | `run`が探索 |
| `latent` | 観測から推定する有効値 | `identify`が探索 |

公差を持つ既設部品は、その実現値を`operating`として条件に列挙する。`fixed`に探索範囲を付けたり、同じ名前を二つの役割へ置いたりしない。列挙した頂点の合格率を製造歩留まりや連続公差領域全体の保証とは解釈しない。

<a id="input"></a>

## 4 入力の共通規則

通常入力は`pcd.rf.v1`のYAML一つで構成する。既存SPICE、観測表、目標波形など、独立した回路・データだけを必要に応じて別ファイルへ置く。設定ファイルの多段includeは必要ない。

| 項目 | 任意回路の入力 | 標準整合回路の入力 |
|---|---|---|
| 回路 | `circuit.components`または`circuit.file` | `network.type`と各素子の役割 |
| 値と所有者 | `parameters` | `network.fixed/search/tuning` |
| 電源と解析 | `source/sources`、`analysis` | 周波数、`drive_peak_V`、負荷から生成 |
| 条件 | `conditions`のinline行またはCSV | 負荷表、またはinline条件 |
| 順位付け | `objective` | 反射の最悪値を使用 |
| 合格条件 | `acceptance.metric_bounds` | 反射、部品、source、調整余裕の専用項目 |
| 計測と出力 | `measurement`、`output` | 同じ`measurement`、`output` |

`network`と`circuit`はどちらか一方を選ぶ。一般回路のYAMLと標準networkの項目を混在させない。

### 4.1 単位と構造

数値の基本単位はSI。公開YAMLでは`100 pF`、`13.56 MHz`、`2 us`、`50 ohm`、`4 Mrad/s`を一度だけSIへ変換する。対応接頭辞は`p/n/u/µ/m/k/M/G`。CSVとCLIのJSON上書き値にはSI数値を使う。

`unit`はSI単位の注記であり、数値を再変換する指定ではない。`default: 100, unit: pF`は拒否する。`default: "100 pF"`または`default: 1e-10, unit: F`を使う。SPICEファイルと`raw`行の構文は、そのままSPICEとして扱う。

YAML/JSONの重複キー、担当範囲内の未知キー、重複CSV列名、長さの異なるCSV行、非整数の試行数は入力エラーになる。誤記を既定値に置き換えない。プラグイン独自項目はその回路・負荷の設定か`extensions`に置く。

### 4.2 条件行と重み

任意回路の`conditions`はinline行または`{file: conditions.csv}`を受け付ける。`id/scenario_id`と`weight`は省略可能。行を列挙した順序と内容で評価し、暗黙の直積を作らない。異なるCSV列名には`columns`でパラメータ名との対応を指定する。

重みは正かつ有限で、平均・RMS・CVaRなどの集約へ使う。重みが小さくてもその条件の工学的制約を免除しない。条件の独立性は、IDが異なるだけでは成立しない。

### 4.3 一点の計算と探索の区別

`sim-netlist`は既定値で作るプレビュー。`sim-run`は一つの解決済み点を計算する。複数の条件・設計候補・調整値がある場合は、次のように対象を明示する。

```powershell
pcd sim-run examples/rf_ccp_lumped.yaml --scenario nominal_frequency --params '{"L1":1e-6,"C1":2e-10}' --json
```

必要に応じて`--controls`も指定する。全候補・全条件を比較する場合は`run`を使う。Pythonの公開`simulate_case`も同じ選択規則を適用する。

### 4.4 実行設定

| 入力 | `execution`で指定する項目 |
|---|---|
| 任意回路 | `solver/executable/timeout_s`、探索用`optimizer/trials/seed`、`control_state_limit` |
| 標準network | `solver`、`candidate_state_limit/control_state_limit` |

solverの既定は`ngspice_cli`。標準networkの状態上限は各250で、完全列挙できる範囲を指定する。任意回路の`trials`は提案候補数の上限であり、各候補の条件×control評価数とは異なる。同定のholdout・局所感度と、選択点の再計算も別に必要になる。`analysis.options`にはngspiceの解析optionを指定できる。

<a id="circuit"></a>

## 5 任意回路と外部SPICE

### 5.1 素子列と電源

R/L/Cは`ref/n1/n2/value`で記述し、`value`には定数またはパラメータ名を使う。`observe: true`で元の部品端子電圧と直列電流を保存する。`series_resistance_ohm`は有効ESR/DCR等を表す直列抵抗を追加する。

```yaml
circuit:
  output_node: load
  components:
    - {ref: L1, n1: src, n2: load, value: L, series_resistance_ohm: DCR, observe: true}
```

| 電源type | 主な入力 |
|---|---|
| `sine_voltage` | `dc_V`、`amplitude_V`、`frequency_Hz`、`phase_deg` |
| `voltage_pulse` | `v1_V`、`v2_V`、`delay_s`、`rise_s`、`fall_s`、`width_s`、`period_s` |
| `dc_voltage` | `voltage_V` |
| `current_dc` | `current_A`。電流観測用の直列電圧源を生成 |

共通項目は`name/p/n`とAC励振の大きさ・位相。`source`の正端子`p`の既定値は`src`、負端子`n`の既定値はSPICEの基準節点`0`。電圧源には`ac_magnitude_V`、電流源には`ac_magnitude_A`を使う。正弦波のAC大きさは省略時に振幅を使用する。パルスやDCをAC解析にも使うときは、小信号励振を明示する。

初期条件からの自然応答など、過渡解析では`source`を省略できる。電圧と指定したプローブを保存し、電源電流の列は空欄になる。

複数電源は`sources`へ列挙し、観測sourceを`measurement.current_source`、構造化電源のAC励振源を`analysis.ac.source`でそれぞれ選ぶ。source電圧を明示する場合は`measurement.source_voltage: v(p,n)`とし、選択したsourceの端子・極性に合わせる。

### 5.2 ACと過渡の指定

```yaml
analysis:
  ac: {sweep: dec, points: 20, start_Hz: "100 kHz", stop_Hz: "10 MHz"}
  tran: {step_s: "10 ns", stop_s: "20 us"}
```

AC一点は`ac: {frequency_Hz: "1 MHz"}`。一点指定と掃引指定は併記しない。`lin`の`points`は点数、`dec`は一桁当たり、`oct`は一オクターブ当たりの点数である。

過渡の`step_s`は出力要求に加えてngspiceの最大内部刻み`tmax`へも使う。実際に保存される時刻は適応刻みであり、等間隔配列とは限らない。`stop_s`は終了時刻、計算は0秒から開始する。細いパルスや急な入力変化を分解できる刻みを選ぶ。

### 5.3 既存SPICEの取り込み

```yaml
schema: pcd.rf.v1
case_id: imported_ac
circuit: {file: circuits/divider.cir, output_node: out}
parameters:
  fixed: {Rload: "50 ohm"}
analysis:
  ac: {frequency_Hz: "1 MHz"}
output: {}
```

この例の`circuits/divider.cir`は[実行可能な例](../examples/circuits/divider.cir)を参照する。ファイル名は入力YAMLからの相対パスとして解決する。

公開入力の既定は`format: deck`と`source_policy: preserve`。deckの先頭行はSPICEのタイトル。タイトルのない素子断片には`format: fragment`を指定する。独立電圧源が一意なら観測端子を解決し、曖昧な場合はsourceと電圧vectorを明示する。

元の回路の`.ac/.tran/.control/.op/.dc/.noise/.end`等の実行指示は、YAMLのAC/過渡要求へ置き換える。include/library依存を展開・保存するが、任意のSPICEジョブをそのまま実行する汎用ランチャーではない。

元sourceとYAML sourceの同名重複は拒否する。意図した同名置換に限って`source_policy: replace_named`を選ぶ。節点を共有することだけを根拠にsourceを削除しない。取り込んだ電流源を観測する場合は、直列の0 V電流計とsource-port電圧を明示する。

### 5.4 回路値の設計

[circuit_design.yaml](../examples/circuit_design.yaml)はRを固定し、Cを連続範囲で設計する例である。

```yaml
parameters:
  fixed: {R: "1 kohm"}
  design:
    C: {bounds: ["100 pF", "100 nF"], default: "1 nF", scale: log}
objective: {type: waveform_l2, waveform_file: advanced/target_rc.csv, metric: normalized_rmse}
acceptance:
  metric_bounds:
    peak_abs_voltage_V: {max: 4.9}
execution: {trials: 24, seed: 3}
```

上は回路・電源・解析へ追加する断片。完全な例を`pcd run examples/circuit_design.yaml --require-acceptance`で実行する。目標CSVは`time_s,voltage_V`で、計算時間が目標の範囲を覆う必要がある。

`objective.type`は`impedance_match`、`rf_load`、`waveform_l2`。`metric/direction/aggregation`で順位付けを指定できる。通常は前二者のうち`rf_load`が負荷電力の最大化、それ以外が損失の最小化を既定とする。`waveform_l2`のRMSEは目標の観測点上で比較する。

`aggregation`は`worst/mean/rms/cvar`。`worst`はminimize目的で最大値、maximize目的で最小値を使う。その他は条件の正規化重みで平均、二乗平均平方根、悪い側の尾部平均を求める。CVaRの尾部質量`cvar_alpha`や複数目的を直接設定するときは、上級入力の`study.objectives`を使う。

有限な`choices`はgrid、連続`bounds`はDEを既定とする。DEは数値の連続範囲を扱い、離散トポロジーを連続座標へ置き換えない。予算終了と収束、工学的合格は別の状態である。

<a id="network"></a>

## 6 整合回路の設計

標準networkは回路接続を選ぶ入口である。値・調整範囲・適用する負荷・合格条件はYAMLで与える。

| `network.type` | 接続 | 必須の部品名 |
|---|---|---|
| `l_match` | sourceから負荷へ直列L、負荷側へ並列C | `L1,C1` |
| `pi_match` | source側並列C、直列L、負荷側並列C | `C1,L1,C2` |
| `pi_match_harmonic` | piに、負荷側の直列LC分岐を追加 | `C1,L1,C2,Lh,Ch` |

各部品は`fixed`、`search`、`tuning`のいずれか一つへ置く。

```yaml
schema: pcd.rf.v1
case_id: load_window
frequency_Hz: "13.56 MHz"
network:
  type: pi_match
  fixed: {L1: "643 nH"}
  tuning:
    C1: ["480 pF", "670 pF", "890 pF"]
    C2: ["45 pF", "160 pF", "210 pF"]
load:
  type: impedance_table
  reference_plane: electrode_terminal
  evidence: {origin: supplied_terminal_data}
  points:
    - {resistance_ohm: 25, reactance_ohm: -80.77}
    - {resistance_ohm: 50, reactance_ohm: -80.77}
    - {resistance_ohm: 12.5, reactance_ohm: -80.77}
acceptance:
  reflected_power_fraction_max: 0.10
```

`fixed`は一つの装置値、`search: {C1: {values: [...]}}`はハードウェア候補、`tuning`は各条件で選べる設備設定である。上の例ではハードウェアは一つで、3負荷条件それぞれについて9調整値を比較する。

searchとtuningは正の一意な値を列挙する。標準networkは宣言した候補と調整値を完全列挙し、状態上限を超えると拒否する。既定のcandidate/control状態上限は各250。上限はサンプリング予算ではない。

### 6.1 反射 損失 ストレスの制約

反射制約は`reflected_power_fraction_max`または`reflection_magnitude_max`の一方を必ず指定する。前者は`|Gamma|²`の上限である。基準インピーダンス`reference_impedance_ohm`は既定50 Ωで、物理的な直列抵抗とは別の設定になる。

```yaml
drive_peak_V: 100
network:
  loss_ohm: {C1: 0.1, L1: 0.5, C2: 0.1}
acceptance:
  reflected_power_fraction_max: 0.10
  component_limits:
    L1: {current_rms_A_max: 1.0, loss_W_max: 0.5}
  source_limits: {current_rms_A_max: 1.0, apparent_power_VA_max: 50}
  control_margin_min: 0.20
  loss_balance_fraction_max: 1.0e-5
```

上はnetworkへ追加する設定断片。`component_limits`は`voltage_peak_V_max/voltage_rms_V_max/current_peak_A_max/current_rms_A_max/loss_W_max`、`source_limits`は`current_rms_A_max/apparent_power_VA_max`を受け付ける。損失制約にはその部品の`loss_ohm`、損失収支制約には観測した有効損失が必要になる。

絶対的な電圧・電流・損失制約には`drive_peak_V`を宣言する。省略した反射計算は1 V peak基準となり、図と信号辞書に`normalized_1V_peak`を記録する。その条件のW/V/Aを装置の駆動値と混同しない。

`control_margin`は調整範囲端で0、中央で1となる正規化余裕である。幅のある数値軸・条件の最悪値を使う。固定値とカテゴリ値は除外する。幅のある数値軸がない条件では未定義となり、`control_margin_min`を満たさない。範囲のある軸を観測値に固定する場合は、宣言した範囲に対する位置で評価する。

<a id="loads"></a>

## 7 負荷モデルの選択と物理的意味

`reference_plane`は任意の表示ラベルで、省略時は`load_ports`と実際の負荷端子を出力する。実測値を使う場合は測定の基準面と値を得た方法を記録し、校正やde-embeddingはデータを入れる前に行う。`evidence`に記録した由来や適用範囲は入力とともに保存する。モデルの適用性は基準面、校正、観測との比較から判断する。

### 7.1 一点または測定周波数のR+jX

`impedance_point`は一つの周波数での`R+jX`を表す。`R>=0`を要求し、Xが正なら直列L、負なら直列Cで実現する。

```text
L = X / (2πf)               X > 0
C = -1 / (2πfX)            X < 0
```

ACの指定は一点とし、その周波数を`model_frequency_Hz`に一致させる。過渡でこの等価回路を使う場合も基本周波数を一致させる。RとXを再現する根拠はこの一点にある。複数の測定周波数は`impedance_table`の行として与える。

```csv
scenario_id,frequency_Hz,resistance_ohm,reactance_ohm,weight
point_a,10000000,25,-80,1
point_b,13560000,28,-60,1
```

```yaml
load:
  type: impedance_table
  file: chamber_impedance.csv
  reference_plane: electrode_terminal
  evidence: {origin: measured, deembedding: fixture_v3}
```

CSVに周波数を与える場合はトップレベルの`frequency_Hz`を省略する。行の周波数がsource、負荷実現の基準、AC一点のすべてへ使われる。行間の周波数補間は行わない。追加の来歴列は回路パラメータにしない。

### 7.2 準静的な時間変化

`impedance_profile`のCSVは`time_s,resistance_ohm,reactance_ohm`を持つ。時刻は非負で厳密な昇順、Rは非負、全値は有限とする。`frequency_Hz`、`drive_peak_V`、`weight`も各行へ与えられる。

各行は独立したAC一点であり、状態を次の行へ伝搬しない。RF周期より十分遅い包絡変化を局所定常とみなせる場合に使う。絶対電力を示すため、周波数と駆動振幅を各行またはcaseへ与える。各調整値の選択は可能だが、チューナーの移動時間や遅延は計算しない。

選択結果の`snapshot_response.csv`に、負荷R/X、source側入力R/X、反射、電力、調整値、制約を時刻順に保存する。実例は[rf_quasi_static_profile.yaml](../examples/rf_quasi_static_profile.yaml)。

### 7.3 有効CCPモデル

```text
Z_CCP(ω) = R_eff + jωL_eff + 1/(jωC_sheath_eq)
```

```yaml
load:
  type: ccp_lumped
  reference_plane: electrode_terminal
  evidence: {origin: qualified_fit, frequency_Hz: [10000000, 20000000]}
  parameters: {R_eff_ohm: 18, L_eff_H: "250 nH", C_sheath_eq_F: "120 pF"}
```

`R_eff`は基準面より下流で受け入れる総実電力を表す有効抵抗、LとCも端子から見える有効量である。電子・イオン・sheathの電力配分を個別に求める量ではない。

一点の複素V/Iは実数2観測なので、R/L/Cの3値を同時に自由化して一意性を主張しない。一つを独立校正で固定するか、多周波数・多条件で識別可能性を確認する。

### 7.4 有効ICPモデル

```text
Z_series(ω) = R_coil + jωL_coil + ω² L_reflected / (γ_secondary + jω)
Z_ICP(ω) = 1 / (1/Z_series(ω) + jωC_parallel)
```

```yaml
load:
  type: icp_transformer
  reference_plane: coil_terminal
  evidence: {origin: plasma_on_and_fixture_fit}
  parameters:
    R_coil_ohm: 0.2
    L_coil_H: "2 uH"
    reflected_inductance_H: "180 nH"
    secondary_damping_rate_rad_s: "4 Mrad/s"
    C_parallel_F: "20 pF"
```

`0<=L_reflected<=L_coil`、`γ_secondary>0`。減衰率は端子fitの有効値であり、独立した物理的根拠がある場合にのみ衝突周波数として解釈する。任意の`C_parallel`は理想的な端子サセプタンスで、容量結合による実電力加熱を表す枝ではない。

コイル・治具の校正、plasma-off/onの複素観測、多周波数の残差とholdoutがモデル選択の根拠になる。coil-portとelectrode-portの電力を単純加算してハイブリッド装置の総電力を作らない。

<a id="identification"></a>

## 8 端子モデルの同定

既知回路と校正値を固定し、有限な範囲を与えた`latent`だけを推定する。最初の完全な例は[terminal_identification.yaml](../examples/terminal_identification.yaml)と[観測CSV](../examples/terminal_observations.csv)である。

```powershell
pcd validate-case examples/terminal_identification.yaml --plan
pcd identify examples/terminal_identification.yaml --output runs/terminal --json
```

公開入力は`parameters.calibration/latent`と、次の短い同定設定を使う。

```yaml
identification:
  observations: {file: terminal_observations.csv}
  error_scale_ohm: 100
  fit_loss_max: 0.02
  holdout_loss_max: 0.03
  condition_number_max: 100
```

標準の観測列は次のとおり。電流は**sourceから回路へ供給する向き**のpeak phasorである。

```csv
frequency_Hz,voltage_re_V,voltage_im_V,current_re_A,current_im_A,partition
10000000,1,0,0.00158218328044895,0.00875247736869483,fit
20000000,1,0,0.0222536258692152,0.0248486906925447,holdout
```

`scenario_id`は任意。標準列ならmappingは不要で、異なる列名だけ`observations.columns`へ指定する。`partition_column`でfit/holdout列名を変更でき、inlineの観測行も使える。

取得単位が分かる場合、CSVに`acquisition_group`列を追加する。既存の列名を使うには`observations: {file: terminal_observations.csv, acquisition_group_column: campaign_id}`と書く。inlineでは各行へ`acquisition_group`を記す。ラベルは回路条件と独立した来歴で、宣言した場合は全行に必要となる。同じラベルをfitとholdoutに分ける入力は、計算前に拒否する。

### 8.1 誤差と識別可能性

```text
Z_observed = V / I_delivered
e_k = |Z_predicted,k - Z_observed,k| / error_scale_ohm
fit_loss = sqrt(Σ w_k e_k² / Σ w_k)
```

`error_scale_ohm`は工学的な誤差の尺度であり、測定ノイズの標準偏差を自動推定した値ではない。既定は100 Ω。最小二乗法は、この重み付きRMSと同じ意味の実部・虚部残差を最適化する。

通常は有界最小二乗を使い、明示的にDEへ変更できる。最小二乗は端子V/I同定の単一RMS目的に対応し、一般波形の設計最適化へ読み替えない。差分ヤコビアンの提案も試行予算に数える。標準の端子V/I同定では観測残差を目的に使い、局所感度は観測したインピーダンス成分について求める。

fit後は推定値を凍結してholdoutを計算し、正規化した局所感度行列のrankと条件数を確認する。局所full rankは、ノイズ下の信頼区間や大域的一意性を保証しない。既知の正解への回収判定は、通常の実測入力ではなくbenchmark側に置く。

### 8.2 holdoutの意味

標準の同定は条件に対応付けた複素V/Iを観測とする。観測に`frequency_Hz`を含む場合は各行の周波数で実際に計算した応答を使う。周波数が行ごとに異なる一点ACでは`analysis.ac.frequency_Hz: frequency_Hz`で参照する。観測と固定のAC一点が食い違う入力は拒否する。単一の`target.waveform_file`を使う`waveform_l2`は波形目標に対する設計指標である。カスタム同定指標は条件ごとの観測から目的値と感度評価用の予測値を返す。比較表の観測・残差列は、指標が対応する値を返す場合に埋める。

既定の`holdout_mode: new_conditions`では、解決済みの運転条件または固定調整値が推定用条件と異なる必要がある。必要なら`condition_parameters`で比較項目を指定する。周波数の区別にはAC標本の照合と同じ許容差を使う。

同じ設定での反復測定には`holdout_mode: replicates`を宣言する。この場合は再現性の評価であり、新しい運転条件への汎化とは呼ばない。表だけから測定campaignや取得装置の独立性は確認できないため、実測では取得groupも管理する。

結果の`validation.acquisition_group_separation`は、宣言されたラベルの分離を示す。取得の独立性は測定記録で裏付ける。`holdout.by_acquisition_group`と`holdout.by_condition`は、部分集合の誤差、有効観測数、全holdoutに対する重みの割合を示す。合否には入力で指定した行重みを使う。各取得の総重みを揃える場合は、取得内の行重みを調整する。

条件値の型はパラメータ宣言に従う。文字列と宣言した`001`はそのまま保持し、数値条件の異なる表記は同じ値として比較する。観測CSVを読み込んだ後に分割や取得グループが変わった場合は、ケースを読み込み直す。

fit・holdout・感度の必要な証拠が利用不能なら`not_identified`となる。fit不能時の推定値は空で、後続段階は`not_run`となる。開放端子の順解析が成功しても、有限なZを必要とする同定には使えない。

### 8.3 校正値・観測値の不確かさ

校正や観測の不確かさを評価するには、根拠のある値の組を`identification.uncertainty`に名前付きで追加する。次は標準例に対して治具抵抗と受信電流を別々に変える例である。各値は置換後の値で指定する。共通の校正ずれを複数の観測に反映する場合は、同じ名前の中で該当する行をまとめて指定する。

```yaml
identification:
  # 既存の observations、誤差上限などと併記する。
  uncertainty:
    fixture_resistance_high:
      calibration: {fixture_R_ohm: "3 ohm"}
    holdout_receiver_shift:
      observations:
        holdout_40_68MHz: {target_current_re_A: "20 mA"}
```

各組で校正値を固定し、fit・holdoutの分割、運転条件、制御値、行重み、推定範囲を共通にして同定をやり直す。変更したholdout観測も評価だけに使う。端子V/Iでは`target_voltage_re_V`、`target_voltage_im_V`、`target_current_re_A`、`target_current_im_A`を指定できる。試行予算は一組ごとに適用する。

`estimated_latent`と`fit`、`holdout`、`identifiability`は基準値での結果を示す。基準値を含めた全組が合格したときに`status: identified`となる。`uncertainty`には基準値での合否、各組の変更値と判定、証拠が得られなかった組の数、推定値とholdout予測値の最小・最大および集計対象の組数を保存する。範囲はfitと局所識別の判定に合格した組から求め、holdout予測値は利用できる値を集計する。各組の結果から保存した入力と計算結果にたどれる。評価範囲は有限個の宣言値に限る。区間内部の網羅や確率的な信頼区間の推定には別途根拠が要る。

<a id="measurement"></a>

## 9 計測面・数値計算・部品ストレス

### 9.1 表示電圧 source電圧 負荷電圧

```yaml
measurement:
  voltage_node: mid
  probes:
    input_V: v(src)
    filter_V: {vector: "v(out)", unit: V, label: フィルタ出力}
    resistor_drop_V: v(src,mid)
    branch_A: i(Vmeter)
```

| 信号 | 過渡の列 | ACの列 | 定義 |
|---|---|---|---|
| 表示電圧 | `voltage_V` | `voltage_V_re/im` | `measurement.voltage_node`。指定がなければ回路出力 |
| source-port電圧 | `source_voltage_V` | `voltage_re/im` | 選択sourceの差動電圧 |
| source電流 | `current_A` | `current_re/im` | source正端子へ流入。供給方向は負号を付ける |
| 負荷電圧 | `load_voltage_V` | `load_voltage_V_re/im` | 実際の負荷p,n間電圧 |
| 宣言probe | `<alias>` | `<alias>_re/im` | 指定vectorと極性の観測 |

列名から意味を推測せず、`data/signals.json`のvector・単位・端子・正方向・保存領域を読む。一般のSPICE式には単位を宣言する。任意のdevice内部量はbackend・素子・領域で利用可否が異なり、取得不能vectorは計算失敗と理由を保存する。

上の`Vmeter`は回路に置いた0 V計測源の名前を表す。任意の別名を宣言すれば電流計が自動生成される指定ではない。構造化部品の電流は`observe: true`が生成する`component_<ref>_current_A`でも取得できる。

負荷モデルが別ブロックの場合は`measurement.load_current: auto`で負荷入口へ0 V電流計を挿入する。ここでprobe名を指定した場合も、負荷電力の計算には負荷直列の電流を使い、指定したprobeは別の観測として保存する。取り込み回路内の負荷は、直列の電流計を置いて`load_current`へそのprobe名を指定する。負荷電流が保存されていなければ、計測を加えて再計算する。

### 9.2 ACの規約

ACは動作点まわりの線形応答で、PCDのphasorはpeak規約を使う。

```text
I_delivered = -I_source
Z_input = V_source / I_delivered
Gamma = (Z_input - Z0) / (Z_input + Z0)
P_real = 0.5 Re(V conj(I))
V_rms = |V|/sqrt(2)       I_rms = |I|/sqrt(2)
V_forward = (V + Z0 I_delivered)/2
V_reflected = (V - Z0 I_delivered)/2
P_forward/reflected = |V_forward/reflected|²/(2 Z0)
```

`Z0`は正の実数。入力Zはsource-port、負荷受入電力はload-portで求める。波電力はこの基準に対する計算量で、実発振器の内部保護や方向性結合器の校正値を直接モデル化したものではない。

整合判定と端子同定の動作点は、指定周波数で実際に解いた複素V/Iを使う。掃引の解析表示では動作点の複素V/Iを補間でき、`frequency_basis`と`frequency_bracket_Hz`で根拠を示す。return lossは`-20 log10(|Gamma|)`で、標準計算の振幅床`1e-30`により完全整合の表示上限は600 dBとなる。負荷電力を観測する掃引に欠測がある場合は、負荷電力の帯域幅を未定義とする。負荷電力の観測を指定していない場合は、駆動端の受入電力比を帯域幅の根拠に使う。

AC一点では取得した周波数を使う。掃引では観測sourceの基本周波数が取得範囲内にある場合に動作点として使い、基本周波数がない場合や範囲外の場合も全周波数の表・図を保存する。`analyze --frequency-Hz`で保存範囲内の動作点を選び直せる。ngspiceでは組み立て済みデッキから独立電源と観測した電圧端子・電流枝・極性の対応を確認する。駆動点が確定できない場合は、その点を必要とするインピーダンス、反射、電力、損失を利用不能とする。外部の0 V電流計で駆動点を測る場合は、AC源と分岐のない直列接続に置き、その給電端子を同じ極性の電圧probeで測る。接地端子を省略した`v(p)`は`v(p,0)`を意味するため、この照合に使う外部SPICEの接地端子は`0`と記述する。分析結果の`measurement.source_scope`に根拠を保存する。回路で取得できた観測信号や伝達比はそれぞれ利用できる。

<a id="multiport"></a>

#### 多ポートの複素S行列

`network-run`はSPICE被測定回路（DUT）の各ポートに直列抵抗付きの電源を接続し、AC励振を一つずつ切り替える。例えば`R1 left right 40`を含む`series.cir`を次の設定で測る。

```yaml
schema: pcd.network.v1
dut: {file: series.cir, format: fragment}
ports:
  - {name: P1, p: left,  n: 0, reference_impedance_ohm: 50, termination_ohm: 80}
  - {name: P2, p: right, n: 0, reference_impedance_ohm: 75, termination_ohm: 90}
ac: {sweep: lin, points: 11, start_Hz: 1e6, stop_Hz: 2e6}
```

`pcd network-run network.yaml --out runs/network`を実行する。DUTがタイトル行付きなら`format: deck`を指定する。`p,n`はDUT直下で接続が確認できる端子を指定し、接地は`0`で表す。`reference_plane`は結果に残す任意の計測面ラベル。`termination_ohm`の既定値は各ポートの正の実参照抵抗。`dc_bias_V`は治具の開放DC電圧で、DUT接続時の端子電圧は回路の動作点で決まる。各計算でDUT・DC源・終端を固定し、非ゼロのAC励振だけを一つずつ切り替える。DUTには独立DCバイアス源を含められる。独立AC励振、既知の乱数関数、全ノードに分岐を作る`.options rshunt/cshunt`を含むDUTは入力エラーになる。

| 成果物 | 判断に使う内容 |
|---|---|
| `waves.csv` | 各励振と観測端子の複素V、DUT流入を正とする複素I、入射・反射電力波（√W） |
| `scattering.csv` | 周波数・出力側・入射側ごとの複素S。未定義点は空欄 |
| `basis.csv` | 周波数ごとの入射行列のrank、条件数、相対残差、算出可否 |
| `network_result.json` | 完了状態、各励振run、端子面、参照抵抗、再実行用DUTと各表の参照とhash |

同じ周波数で全端子の観測を使って`B = S A`を解くため、治具の終端抵抗と電力波の参照抵抗を別に指定できる。欠測、rank不足、条件数上限超過の周波数はSを返さず、コマンドは非ゼロで終了する。励振数とポート数が等しい場合、残差は数値解の確認用である。Pythonでは`pcd.api.run_network`で同じ処理を実行できる。別手段で取得した複素V/Iには`PortReference`・`PortExcitation`・`solve_scattering`を使える。同じ線形化点、周波数、計測面であることはデータ取得側で確かめる。

### 9.3 過渡と周期測定

保存した適応時刻を保持し、積分とphasor fitへ時間重みを使う。周期窓の端点だけを補間し、細いピークを固定の粗い時刻列へ置換しない。

| `measurement`項目 | 既定値 | 意味 |
|---|---:|---|
| `periodic_cycles` | 3 | 最後の測定周期数 |
| `settling_comparisons` | 2 | 隣接周期を比較する最小回数 |
| `settling_tolerance` | 0.001 | 正規化した周期差の上限 |
| `harmonic_count` | 3 | 基本波を含む高調波数 |

必要な履歴は`max(periodic_cycles, settling_comparisons+1)`周期で、その区間の隣接周期をすべて比較する。周期判定には使用する正のサンプル間隔が基本波の半周期未満であることを求める。高調波fitでは対象の最高次成分の半周期未満を必要条件とする。判定不能・未収束の定常測定は制約を合格させない。

電力、RMS、高調波は選択した同じ全周期窓を使用し、実際の開始・終了時刻を保存する。高調波図も同じ設定を継承する。ACの動作点周波数と、過渡の基本周波数は独立であり、`analyze --frequency-Hz`は過渡の基本周波数を変更しない。

非周期の過渡にも、保存した最初の時刻からの符号付き端子エネルギーを出力する。`source_net_energy_J`はsourceが供給した正味エネルギー、`load_net_energy_J`は負荷端子へ入った正味エネルギーである。電力が欠けた正の時間幅を越える積分は欠測になる。エネルギーは宣言した端子面での積分なので、反応や電子・イオンへの分配には独立した計測や物理モデルが必要になる。実測パルスとの比較はチェックアウト内の`bench/literature/p2_minesi2022_pulse/README.md`にある。

平均電力、RMS、積算エネルギーは保存した時刻間隔と区間の端点を使う。等時刻に前後の標本がある場合は両側を保持し、整定比較と高調波にも使う。評価に使う信号が区間内で欠測した場合、補間でその区間を埋めず、定常値は判定不能とする。波形ピーク制約は元の保存波形が揃う場合の観測点上の最大値を使う。波形目標の範囲が固定の過渡解析時間から外れる場合は入力エラー、条件に依存する解析で不足した場合はその条件の計測失敗として保存する。

この二つの名前はエネルギーの計算結果に使うため、プローブ名には指定できない。

### 9.4 損失と制約

部品の有効損失は`I_rms² R_series`。過渡のRMSには高調波も寄与する。端子のpeak電圧・RMS/peak電流を返すが、内部電界や巻線温度を直接求めるモデルではない。

```text
component_L1_voltage_peak_V       component_L1_voltage_rms_V
component_L1_current_peak_A       component_L1_current_rms_A
component_L1_loss_W               modeled_component_loss_W
network_loss_W = source_real_power_W - load_real_power_W
component_loss_balance_residual_W = network_loss_W - modeled_component_loss_W
transfer_efficiency = load_real_power_W / source_real_power_W
component_loss_balance_fraction_of_source = abs(component_loss_balance_residual_W) / abs(source_real_power_W)
```

電源電力は`observed_source`で示した電源端子の値である。負荷電力は指定の負荷端子面で求める。AC・過渡とも、回路損失・効率・損失残差は、その観測が回路に入る給電を覆うと確認できる場合に計算する。未計測の独立電源・振る舞い電源・従属電源などがある場合は各端子の計測値を保持し、回路全体の差分と比は理由付きで未定義とする。

電力比が未定義ならJSONは`null`、CSVは空欄とし、`<指標名>_unavailable_reason`を保存する。有限な微小電力の比に任意の分母床を代入しない。未定義指標を使う制約は数値余裕を空欄とし、判定に必要な証拠の不足を理由付きで返す。

各制約の`.margin`は元の単位で、上限なら`limit-value`、下限なら`value-limit`。正は余裕、0は境界、負は違反を表す。順位付け用の正規化`.violation`と区別する。

<a id="output"></a>

## 10 回路図・グラフ・数値表

### 10.1 出力を一つのYAMLで指定する

完全な例は[circuit_outputs.yaml](../examples/circuit_outputs.yaml)。R1=1 kΩ、R2=470 Ω、C1=1 nFで、表示点midとフィルタ出力outを別々に観測する。

![RC回路の部品値と観測位置](assets/rc-circuit.svg)

図の青いマーカーと矢印は観測位置・電流の正方向を示す。計測用0 V sourceは物理回路表示で畳み込み、元の部品と端子名を残す。

```yaml
output:
  formats: [png, svg, pdf]
  schematic: {enabled: true, view: physical}
  data: {waveforms: selected}
  plots:
    - type: waveform
      name: final_cycles
      signals: [input_V, filter_V]
      time_s: [18 us, 20 us]
    - type: ac
      signals: [filter_V]
      frequency_Hz: [100 kHz, 10 MHz]
    - type: bode
      name: filter_transfer
      numerator: filter_V
      denominator: input_V
    - type: harmonics
      signals: [filter_V]
      count: 5
```

`output: {}`を使うと、計算後に適用できる図を自動作成する。例のように`plots`を列挙すると個別計算のグラフを指定できる。`plots: []`は個別計算のグラフを省略し、回路図は`schematic`に従う。`circuit`という名前は回路図に使う。

waveformは単位ごとに段を分ける。表示窓は軸の切出しであり、元データと対応CSVを間引かない。対数軸は非正値を黙って除外せず、利用不能を返す。

### 10.2 伝達比と高調波の解釈

![フィルタ出力と入力の複素伝達比](assets/rc-transfer.svg)

Bodeは、明示した同じ単位の2信号から複素比を計算してgainとphaseを求める。`Vout/Vin`とsource側の入力インピーダンスを混同しない。分母ゼロは欠測、振幅ゼロの位相は未定義、gainは−∞として保存し、欠測区間を跨いで位相をunwrapしない。

高調波の`time_s`は解析対象範囲で、その中の最後の全周期を共通規則で選ぶ。H1=0なら相対dBcは利用不能。表示下限−120 dBcは図だけに適用し、振幅とdBcの元値は変更しない。H1より強い高調波も上限で切らない。

### 10.3 回路図と対応CSV

Schemdrawと共通の回路描画部品を使い、実行したnetlistの並列・別枝・複数sourceを保持する。多端子素子は全pinとnet名を持つblockで示す。同じnet名は接続を意味する。

`physical`は生成した計測源を畳み込み、`instrumented`は電流計と内部節点も表示する。条件分岐、未対応素子、未解決subcircuitを推測で補完せず、図と`circuit.json`へ`partial`を記録する。回路図は説明用途であり、製造用CADや電気ルール検査の代わりにはしない。

| ファイル | 内容 |
|---|---|
| `signals.json` | 描画に使った信号辞書。派生信号には式を記録 |
| `frequency_metrics.csv` | 全周波数のV/I、R/X、反射、電力、取得できたストレス |
| `operating_metrics.csv` | 評価できた動作点のスカラー指標。`analysis`でACと周期過渡を分け、駆動基準と負荷基準面を併記 |
| `instantaneous_power.csv` | 元の時刻に対応する供給電流、瞬時電力、観測できた端子の累積正味エネルギー |
| `harmonic_metrics.csv` | 負荷V/Iの次数、振幅、dBc、実際の窓と周期数 |
| `<name>_data_<n>.csv` | 選択した図と同じ波形・AC・伝達比・高調波表 |
| `summary.json` | 各図・表の参照、設定、完了状態、利用不能理由 |

全曲線に有限な描画値がない図は`not_available`、一部が欠ける場合は`partial`。`coverage`に表示範囲内の観測数と曲線ごとの有限値数を記録する。未定義を含む表と理由付きの図を保存し、出力全体は未完了となる。

### 10.4 再描画と比較

```powershell
pcd analyze RUN_DIR --out run_view --json
pcd analyze STUDY_ROOT --out study_view --json
pcd analyze IDENTIFICATION_ROOT --out identification_view --json
```

`RUN_DIR`などには、実行時に出力された結果ディレクトリを指定する。`analyze`は保存済みの回路・計測設定・観測データから図表を作る。`--out`を省略すると結果ディレクトリ内に`analysis`を作り、既にあれば`analysis_001`などへ保存する。図の設定だけを変えるには`--case OUTPUT_YAML`を追加し、そのYAMLの`output`を編集する。新たな信号が必要なら、プローブを追加して元の計算を行う。

設計・同定の再描画では保存済みの選択点、目的履歴、制約余裕、条件比較、fit/holdoutの残差と局所感度を使う。これらの概要図は`plots: []`でも作る。studyのルートは現在の公開世代を選ぶ。特定の世代を選ぶには、世代ディレクトリまたはその`generation_result.json`を指定する。同定の再描画は元のfit/holdout世代を参照する。再描画の完了状態は保存済みの工学的合否や同定の判定と別に記録する。

条件比較では解析の種類、観測端子の組、測定面、励振基準が揃う系列を同じパネルに置く。交流は観測電圧のピーク振幅、過渡は瞬時電圧で表示する。`voltage_comparison`には各パネルの条件と有効標本数、表示できなかった理由を保存する。制約余裕の図では、必須値が利用できない条件に`unavailable`を表示する。

再描画時は保存データと参照先の整合性を確認する。照合できた範囲は解析用`summary.json`の`source_integrity`や`source`から確認できる。参照先が見つからない場合は、結果ディレクトリ全体を同じ構成で保持しているかを確認する。

<a id="results"></a>

## 11 結果の見方と再開

### 11.1 成功を判断する場所

| 状態 | 読む場所 | 意味 |
|---|---|---|
| 入力の妥当性 | `validate-case`の`ok` | 構造と設定の検査。回路計算の保証ではない |
| 順計算 | `summary.json`の`status` | `ok`で数値計算が完了。波形・ACの元データは`artifacts`から開く |
| 多ポート計測 | `network_result.json`の`status`、`basis.csv` | S行列を計算できた周波数と、未定義の理由を確認する |
| 工学的合否 | `study_result.json`の`best.status`と`best.conditions` | 選択候補の全体判定、条件ごとの合否、制約余裕を確認する |
| 探索の終了 | `execution.optimizer_outcome.termination`と`verification` | 列挙・収束・予算終了の違いと、選択点の再計算を確認する |
| 同定 | `identification_result.json`の`status`、`fit`、`holdout`、`sensitivity` | 推定に使った条件と未使用条件の誤差、局所感度を確認する |
| 依頼出力 | `output.status`と各図の状態 | 要求した図・表を完了できたか |

`run --require-acceptance`は工学的不合格と証拠不足を非ゼロ終了にする。必須の計測値が利用できない制約は`evidence_available: false`を記録する。`identify`の`not_identified`、計算失敗、要求出力の未完了も非ゼロになる。入力エラーは終了コード2。機械処理では終了コードとJSONの両方を使う。

描画処理で例外が発生しても、完了した計算・検証・同定の証跡を保存する。この場合は`output.status: partial`と`output.error`を返し、描画のsummaryが生成されなかった場合は`output.summary`が`null`になる。

選択候補の`meets_declared_acceptance`は宣言した条件と制約の合格である。DEの予算終了や評価済み点の証跡完全性を、連続設計空間の大域最適性へ読み替えない。

候補は、受入条件を満たす、必要な証拠が揃って制約を外れる、証拠不足の順で比較する。ソルバー成功率は判定に必要な量の取得状況と別に扱う。数値のように見える文字列はカテゴリ値なので、調整余裕の数値軸には含めない。

### 11.2 単体runとstudyの保存

```text
単体run/
  summary.json
  data/transient.csv  data/ac.csv  data/signals.json
  debug/manifest.json  debug/case.yaml  debug/netlist.cir  debug/solver.log
  analysis/summary.json と図・派生CSV

study_root/
  study_result.json
  generations/g_<id>/case.yaml  input_case.yaml  resolved_plan.yaml
  generations/g_<id>/input_manifest.json  evaluations.csv  study_history.json
  generations/g_<id>/generation_result.json  best_candidate.json  output/
  artifacts/ 配下の個別計算
  archives/ 個別計算が参照する共有入力アーカイブ
  raw/ 再利用可能なraw結果のキャッシュ
```

実際の参照はsummary/manifestの`artifacts`と`output.summary`から辿る。AC-only runの過渡表は空の表である。

studyは最後にroot結果の参照を公開し、前の世代を上書きしない。全探索評価は`evaluations.csv`、候補ごとの履歴は`study_history.json`、選択点の詳細は`best_candidate.json`に保存する。選択controlはキャッシュを使わず再計算し、探索の評価行とは区別する。

study IDは大文字と小文字を区別する。衝突を避けるため、出力ディレクトリ名にはハッシュが付くことがある。保存場所は結果の`run_root`で確認する。

外部データ、SPICEのinclude、プラグインは元のケースとともに保存する。回路、入力データ、ソルバーが同じかを確認できる条件で、保存済みの計算を再利用する。追跡できないソルバー依存ファイルがある場合は新規計算を行う。

設計探索・同定では、同じ入力を再計算できる回路を使う。外部SPICEに`AGAUSS`、`TRRANDOM`などの乱数式や乱数源がある場合は`sim-run`で個別に計算する。設計探索・同定でばらつきを比較する場合は、具体的な値をケースの条件として宣言する。

同じ入力アーカイブは`archives/`で共有する。`result-prune`は公開済みの旧世代を整理し、現行世代、処理中の世代、および公開済みマーカーのない旧版の世代を保持する。旧世代を参照する分析と、削除を伴う世代整理は別の時間に実行する。`archives/`と個別計算の証跡は残るため、保持した評価行から参照した計算を再解析できる。

### 11.3 解析用の統合CSV

`output.data.waveforms: selected`は選択点、`all`は探索点も統合CSVへexportする。内部raw証跡やキャッシュを削除する設定ではない。

| 名前空間 | 例 | 内容 |
|---|---|---|
| `context.*` | `context.candidate_id`、`context.scenario_id`、`context.control`、`context.phase`、`context.status` | 評価の文脈 |
| `parameter.*` | `parameter.C` | 計算に用いたパラメータ値 |
| `signal.*` | `signal.time_s`、`signal.frequency_Hz`、`signal.voltage_V`、`signal.phase` | 観測軸と信号 |

`observations.csv`は一評価一行の索引、`transient_observations.csv`と`ac_observations.csv`は一観測一行。`context.manifest`と`context.signals`から元の定義へ辿れる。`phase`というprobe名でも`context.phase`を置き換えない。

評価表の`control`列と統合CSVの`context.control`列には、調整値を元の型のままJSONで記録する。`evaluations.csv`には列ごとに比較するための`control.*`もある。

```python
import pandas as pd

data = pd.read_csv("transient_observations.csv")
selected = data.loc[data["context.phase"] == "verification"]
print(selected.groupby("context.scenario_id")["signal.voltage_V"].max())
```

同定では`terminal_observations.csv`へfit/holdout、観測値、予測、残差を併記する。CSVの行数を独立した実験数とみなさず、候補・条件・control・取得groupの粒度を確認する。

### 11.4 中断した設計探索の再開

組み込みの`grid`、`random`、`differential_evolution`を使う設計探索では、候補ごとの条件×制御評価が完了するたびに`generations/g_…/checkpoint/`へ結果を保存する。中断前と同じケース、出力先、solver、optimizer、試行数、seedで実行し、`--resume`を追加する。

```powershell
pcd run examples/circuit_design.yaml --output runs --resume
```

一致する未完了の世代が複数ある場合は、表示されたIDを`--resume g_…`で指定する。Pythonでは`run_case_study(..., resume="auto")`または世代IDを使う。設定、外部入力、実装・数値環境・solverの識別情報が一致し、保存済みの候補と計算ファイルを検証できる場合に再開する。再開処理中の世代を別プロセスから同時に再開するとエラーになる。

保存した候補の提案と評価をseedから順に照合してoptimizerへ戻し、次の候補から探索を続ける。中断した候補は再評価し、利用可能なrawキャッシュを使う。`execution.restart`には復元した候補・評価、新規評価で使ったrawキャッシュ、以前の未完了回数が入る。選択候補の最終検証は新規計算し、全成果が揃ってから世代を公開する。すでに公開済みの世代を指定した場合は、保存した評価と選択候補の証跡を照合してから結果を返す。

`--resume`を付けない`run`は新しい世代を開始する。プラグインのoptimizer、`least_squares`、同定のfit/holdoutでは探索チェックポイントを提供せず、該当するrawキャッシュを再実行時に利用する。

<a id="evidence"></a>

## 12 実行結果を確認する目安

次の確認は合成回路や宣言した入力に対する電気的・数値的な根拠である。自分の装置へ適用するときは、基準面、校正、モデル残差、未使用条件を追加して判断する。

| 例 | 確認する数値 |
|---|---|
| [外部SPICEのAC計算](../examples/circuit_external.yaml) | 直列50 Ω＋50 Ωで入力抵抗100 Ω。駆動1 V peakで供給実電力0.005 W |
| [目標波形に対する回路値の設計](../examples/circuit_design.yaml) | 選択点の波形誤差とピーク電圧上限4.9 Vに対する余裕 |
| [負荷変動に対する整合設計](../examples/rf_impedance_point_study.yaml) | 3条件とも計算成功し、反射制約を満たすのは公称条件のみ |
| [有効端子モデルの同定](../examples/terminal_identification.yaml) | 推定に3点、検証に2点を使用。18 Ωと120 pFを回収し、局所感度のランクは2 |

![fitとholdoutを分けた観測対予測と残差](assets/terminal-fit.svg)

同定図を再現するときは、同定例のYAMLに`output: {formats: [png, svg], schematic: false, plots: []}`を追加する。この出力指定でもfit/holdoutの比較・残差・局所感度の概要図を作る。

同定例の18 Ωは未知負荷抵抗であり、既知の治具抵抗2 Ωは固定して計算する。合成観測の端子抵抗20 Ωをそのまま負荷抵抗として返す例ではない。図の微小残差は合成データへの回収精度であり、実測ノイズ下の精度を示さない。

図の入力とSHA-256は[図の生成情報](assets/evidence.json)に記録する。実行可能な元入力は`examples/`にある。

<a id="commands"></a>

## 13 CLIとPython API

| コマンド | 使い方と主な設定 |
|---|---|
| `list` | 登録された回路・負荷・solver・metric・optimizerを確認 |
| `solver-diagnose` | `--executable`、`--timeout-s`、`--json` |
| `validate-case CASE` | `--plan`で解決済み設定、`--strict`で警告も失敗、`--json` |
| `sim-netlist CASE` | `--out FILE`。既定値でのプレビュー |
| `sim-run CASE` | `--scenario`、`--params`、`--controls`、`--run-root`、`--solver`、`--json` |
| `network-run SPEC` | 固定したDUTの複素S行列を指定端子間で測る。`--out`必須、`--json` |
| `run CASE` | `--output`、`--require-acceptance`、任意回路の探索には`--optimizer/--trials/--seed`、対応する中断探索には`--resume [GENERATION]` |
| `identify CASE` | `--output`、`--optimizer/--trials/--seed`、`--solver`、`--json` |
| `analyze RUN` | 単体・study・同定の保存結果。`--out`、`--case`の出力設定、単体のAC marker用`--frequency-Hz`、`--json` |
| `visualize-netlist FILE` | `--out`、`--view physical/instrumented`、`--format auto/deck/fragment`、`--summary-json` |
| `result-summary STUDY` | 候補一覧。studyまたは公開世代のディレクトリと、対応する結果JSONを指定できる。`--out`で保存 |
| `result-prune STUDY` | 旧世代の整理をpreview。`--keep N`の既定値は3。`--apply`で適用 |
| `result-prune STUDY --raw-cache` | 再利用用の索引を整理。`--keep N`または`--max-bytes N`で保持上限を指定。`--apply`で適用 |

標準networkの完全列挙集合を`--optimizer/--trials/--seed`で縮小できない。`sim-run --allow-failure`は失敗データを明示的に収集する場合だけ使う。`result-prune --raw-cache`は索引ファイルを更新日時が新しい順に保持する。件数とバイト数の両方を指定した場合は両方の上限を適用する。世代、波形、solver出力、入力アーカイブは保持され、索引を整理した入力点は次回の探索で再計算される。

```python
from pcd.api import load_case, describe_case, simulate_case

case = load_case("examples/circuit_transient.yaml")
print(describe_case(case))
record = simulate_case(case, run_root="runs/python")
print(record.summary())
```

設計は`run_case_study(case, run_root)`、同定は`run_identification(case, run_root)`を使う。低位の`execute_case`は解決済み一点を受ける実行部で、複数点の対象選択を代行する公開入口として使わない。

保存結果の再解析は`pcd.reporting.analyze_saved(path, out_dir=...)`を使う。候補一覧と評価表は`pcd.results.candidate_summary(path)`、`pcd.results.read_evaluation_table(path)`で読み込める。

<a id="advanced"></a>

## 14 上級入力と拡張

### 14.1 明示的な実行case

`case_yaml.v1`は回路ビルダー、プラグイン、複数目的などを明示する形式である。通常の`pcd.rf.v1`は実行時に同じ設定構造へ変換される。用途に応じてどちらか一つのYAMLを使う。

| 公開入力 | 上級入力での所有箇所 |
|---|---|
| `circuit.components` | `circuit.builder: from_yaml` |
| `circuit.file` | `circuit.builder: from_netlist`、`netlist_file/netlist_mode/source_policy` |
| `analysis`と実行環境 | `solver` |
| 役割別parameters | `variables`またはcircuit/load/source内のvariablesと`role` |
| 条件とcontrol | `study.scenarios/scenario_table/controls` |
| 目的と制約 | `target`、`study.objectives` |
| 試行数と手法 | `run.trials`、`optimizer` |

parameter仕様は`default`、`bounds`、`choices`または`values`、`type`、`scale`、control用`grid`、`role`、`unit`、`description/metadata/extensions`を扱う。`choices`と`values`は併記しない。重複宣言や役割の重複は拒否する。上級入力でroleなしのboundsはdesignとして警告し、固定値へ読み替えない。

上級複数目的はfeasibility-firstの辞書順で選択し、適格な評価済み候補の`pareto_front.csv`も作る。連続探索のParetoは観測した候補の非劣集合であり、未探索空間のfrontではない。

### 14.2 外から与える抵抗軌跡

```yaml
circuit:
  builder: from_yaml
  components:
    - ref: Rchamber
      n1: out
      n2: 0
      observe: true
      value:
        profile: chamber_resistance.csv
        time_column: time_s
        value_column: resistance_ohm
        interpolation: linear
        repeat_period_s: 75e-6
solver:
  tran: {step_s: 1e-8, stop_s: 150e-6}
```

これは上級入力の断片。CSVは0秒から始まる厳密昇順の時刻と正の有限抵抗を2行以上持つ。`linear`は区分線形、`hold`は直前値保持。反復時は最後の時刻と`repeat_period_s`、最初と最後の抵抗を一致させる。`step_s`は最短入力間隔以下とする。

このR(t)は外から与える一方向の軌跡で、プラズマ状態を自己整合に更新しない。時変C/Lは電荷・磁束則と外部エネルギー交換の定義が必要なため、同じvalue置換機能には含めない。実例は[time_varying_resistor.yaml](../examples/advanced/time_varying_resistor.yaml)。

### 14.3 plugin

拡張は`pcd.api`の型と`register_simulation/register_solver/register_metric/register_optimizer`を使う。二端子負荷は`register_simulation("load", 名前)`で登録し、`.subckt load_model p n`を返す。solverは`SolverRunRequest`で受け取った生成済みSPICEデッキと解決済みの解析・probe・測定面に従う。optimizerは共通評価関数を通じて候補を計算する。探索側が試行予算を検査し、選択候補の新規再計算は探索の試行数と別に記録する。

例は[plugin_case.yaml](../examples/advanced/plugin_case.yaml)と[custom_methods.py](../examples/advanced/plugins/custom_methods.py)。pluginは利用者が信頼する実行コードとして扱い、入力データと別にhashで識別する。

<a id="development"></a>

## 15 開発と再現検証（リポジトリ利用者向け）

```text
YAML SPICE CSV
      ↓
入力解決と役割分離
      ↓
共通のnetlist生成とsolver実行
      ↓
保存したV/Iと共通の電気計測
      ↓
制約判定 探索 同定
      ↓
世代を持つ結果保存と図表
```

| 変更したい内容 | 所有モジュール |
|---|---|
| 公開入力と既定値 | `case.py`、`authoring.py`、`plan.py`、`inputs.py` |
| パラメータの役割・値域と条件 | `problem.py`、`study_config.py`、`core/` |
| 標準接続と有効負荷 | `topology_catalog.py`、`rf_loads.py`、`sim_methods.py` |
| solver境界 | `simulation_input.py`、`netlist.py`、`solver.py`、`sim_core.py` |
| 電気的な式と周期窓 | `signals/`、`analysis/` |
| 目的と宣言された制約 | `metrics.py`、`core/aggregation.py` |
| 候補の提案と同定 | `search.py`、各optimizer、`identification.py` |
| 保存と描画 | `results/`、`records.py`、`reporting/`、`figures/` |

| 担当 | 確認すること |
|---|---|
| 入力と事前診断 | 入力側が単位、型、範囲、役割を解決する。`validate-case`は実行と共通の回路組立と手法名の解決を使う。周期長の警告は定常RF計測を要求する評価指標に対して、既定条件で判明する不足を伝える。 |
| ソルバーと観測 | `solver-diagnose`は実行環境を確認する。実計算では組立側が回路の接続と生成名を、読取側がベクトル配置を、応答型が要求した時刻・周波数の範囲を確認する。外部実行の失敗は状態・理由・ログで確認する。 |
| 計測と合否 | 計測は候補と条件ごとに、必要な信号、観測面、周波数や周期窓が揃っているかを判断する。工学的な合否は取得できた計測値と宣言した限界を比較する。取得できなかった評価は状態と理由で示し、重み付きの成立割合は結果の内訳に使う。 |
| 保存と表示 | 保存側が世代、成果物の参照とハッシュの照合を扱う。表示側は確定した計測表と判断を使い、描画できた範囲を報告する。 |

```powershell
uv sync --frozen --group dev
uv run --frozen python -m nox -s quality-fast
uv run --frozen python -m nox -s quality-pr
uv build --wheel
```

品質検査はRuff、型、依存方向、振る舞いを確認する。依存方向は、CLIの入口、実行と数値処理の層、ソルバーから計測への逆依存の3規則で確認する。単体テストは数値式や入力の境界条件を、利用経路のテストはCLI・APIの状態と保存結果を、ベンチマークは解析解や公開実測との比較を担当する。coverageは検査範囲を読むために出力する。

実ngspiceを使うテストは、共通の実行環境診断でngspiceが利用できなければ失敗する。ソルバーを使わない開発中の確認には`quality-fast`を使う。

### 15.1 ベンチマークと配布検査

コアの電気回路検証、実行経路の配布受入、文献・公開実測との比較は、ソースの`bench/README.md`から選ぶ。ベンチマークの合格は、宣言した結果や判定を再現したことを表す。制約を外れる設計も、想定した判定を再現すればベンチマークとして合格する。

配布検査ではPython 3.10の最小依存構成とPython 3.12でwheelをインストールし、ソースツリー外から同梱入力、CLI、保存結果、マニュアルを確認する。品質検査と定期ベンチマークの実行条件は、ソースの`.github/workflows/`にある。

<a id="troubleshooting"></a>

## 16 よくある問題

| 状況 | 確認と対処 |
|---|---|
| solverが見つからない | `solver-diagnose`と実行ファイルの指定を確認。別solverへ黙って置換しない |
| 複数条件から一点を選べない | `--scenario/--params/--controls`で選ぶか、全体を`run` |
| SPICEの先頭素子が消える | タイトルの有無と`deck/fragment`を確認 |
| sourceが重複・曖昧 | 元deckとYAMLの同名源、観測source、差動電圧、AC励振の選択を確認 |
| `no such vector` | `error`とsolverログで実際のvector名・領域を確認。部品電流はobserveまたは直列電流計で取得 |
| 負荷電力が想定と違う | source電流と負荷電流、負荷p,n、電流の正方向、寄生の二重計上を確認 |
| 高調波が利用不能 | 基本周波数、必要周期数、収束差、刻み、H1=0を確認 |
| 図がpartial | `coverage`と理由を読み、欠測、表示範囲、未対応接続を確認 |
| 設計が不合格 | 条件別の失敗制約とmarginを読む。計算失敗・調整範囲不足・部品上限を区別 |
| 同定が不合格 | fit/holdout/感度のどの段階が不成立かを読む。校正・範囲・観測数・固定controlを確認 |
| 推定値が複数組ある | 観測の独立性と局所感度を確認。独立校正か追加条件を用意 |
| 古い図と新しい結果が食い違う | root結果のactive generationとanalysisのprovenanceを辿る |
