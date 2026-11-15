# 利用経路の受入検証

公開CLIを実ngspiceで実行し、入力の事前検証、回路計算、判定結果の保存、保存結果からの図表生成を確認します。判定の範囲は、回路解析、設計探索、電気的基準面での有効端子モデル同定です。

リポジトリのルートで、まだ存在しない出力先を指定して実行します。

```powershell
uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance
```

結論は `REPORT.md`、確認項目ごとの判定と成果物の場所は `release_result.json`、コマンドのログは `logs/` にあります。候補の評価と採用した調整値の再計算を確認し、キャッシュの再利用件数と実際のソルバー計算回数を記録します。同定では収束すると試行の上限に達する前に停止できます。

| 利用経路 | 入力例 | 確認できる結果 |
|---|---|---|
| 過渡解析と保存結果の再解析 | [RF負荷端子の過渡解析](../../examples/advanced/rf_port_transient.yaml) | 波形、周期電力、図表、再解析に使う入力 |
| 複数条件での設計探索 | [設計候補・外部条件・調整値](../cases/role_factorial_search.yaml) | 組み合わせの全列挙と各条件の調整値 |
| 制約を満たさない設計 | [整合設計](../../examples/rf_impedance_point_study.yaml) | 不成立の制約と条件別の内訳 |
| 周波数掃引 | [既知のダミー回路](../literature/p1_colpo1999_icp/dummy_resonance.yaml) | インピーダンス、スミスチャート、半電力帯域幅、負荷込みのQ |
| 負荷値の準静的な変化 | [スナップショット比較](../../examples/rf_quasi_static_profile.yaml) | 各負荷条件・時刻に対応した独立の解析結果 |
| 指定された抵抗の時間変化 | [時変抵抗の過渡解析](../../examples/advanced/time_varying_resistor.yaml) | 入力したR(t)、過渡応答、周期計測の適用可否 |
| 波形を目標にした回路設計 | [RCフィルター](../../examples/advanced/generic_rc_filter.yaml) | 制約優先の候補選定、目的値の履歴、評価した候補のパレートフロント |
| 有効端子モデルの同定 | [既知の端子モデル](../../examples/terminal_identification.yaml) | 推定値、同定に使っていない周波数での誤差、再計算、局所感度のランク |

整合設計の不合格例では、計算が成功したうえで、期待どおりの条件と制約で不合格になったことを確認すると受入検証は合格になります。
