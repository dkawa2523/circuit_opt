# 用途別の実行例

回路や条件が近いYAMLを選び、参照しているSPICEやCSVと一緒にコピーしてください。以下のコマンドはリポジトリのルートで実行します。配布wheelで使う場合は、コピーした `examples` の親フォルダーで `python -m pcd` を使います。入力を変更したときは `validate-case --plan` で条件、変数の役割、参照ファイルを確認できます。 [初回の実行手順](../docs/manual-ja.md#quickstart) · [入力の共通規則](../docs/manual-ja.md#input)

## 計算する

```powershell
# 観測点を指定した回路を一つ計算し、図表も生成する
uv run --frozen --no-dev python -m pcd sim-run examples/circuit_outputs.yaml --run-root runs/outputs

# 3つの負荷条件に共通する整合回路を選ぶ
uv run --frozen --no-dev python -m pcd run examples/rf_impedance_point_study.yaml --output runs/matching

# 校正した治具を固定し、観測から端子モデルの値を推定する
uv run --frozen --no-dev python -m pcd identify examples/terminal_identification.yaml --output runs/fit
```

| 実行 | 最初に開く結果 | 確認する値 |
|---|---|---|
| `sim-run` | 表示された実行フォルダーの `summary.json` | `status`、`artifacts` の波形・AC表、`output.summary` の図表 |
| `run` | 表示された設計フォルダーの `study_result.json` | `best.status` と `best.conditions` の制約余裕。全評価は `artifacts.evaluation_table` が指す `evaluations.csv` |
| `identify` | 表示された同定フォルダーの `identification_result.json` | 推定値、同定に使ったデータと検証専用データの誤差、局所感度 |

整合回路の例では公称条件が制約を満たし、両端の負荷条件は反射の上限を超えます。制約を満たすことを終了コードでも判定する場合は `run --require-acceptance` を使います。条件別の図表は保存した設計フォルダーを `analyze` に渡して生成できます。 [結果の状態と保存仕様](../docs/manual-ja.md#results)

## 入力例から選ぶ

| 入力例 | コマンド | 設定と確認の要点 |
|---|---|---|
| [circuit_transient.yaml](circuit_transient.yaml) | `sim-run` | 固定RC回路の立ち上がりと時定数 |
| [circuit_external.yaml](circuit_external.yaml) | `sim-run` | SPICEの電源を保持した一点AC解析 |
| [circuit_outputs.yaml](circuit_outputs.yaml) | `sim-run` | 観測点、回路図、波形、伝達比、高調波と対応CSV |
| [circuit_design.yaml](circuit_design.yaml) | `run` | 目標波形とピーク電圧制約に基づく容量の連続設計 |
| [rf_impedance_point_study.yaml](rf_impedance_point_study.yaml) | `run` | 複数のR+jX負荷点に共通の整合回路と反射制約 |
| [rf_impedance_frequency_table.yaml](rf_impedance_frequency_table.yaml) | `run` | 測定周波数ごとの独立した負荷点 |
| [rf_quasi_static_profile.yaml](rf_quasi_static_profile.yaml) | `run` | 緩やかに変動する負荷を独立したAC計算で比較 |
| [rf_component_stress.yaml](rf_component_stress.yaml) | `run` | 部品端子の電圧・電流と有効損失 |
| [rf_ccp_lumped.yaml](rf_ccp_lumped.yaml) / [rf_icp_transformer.yaml](rf_icp_transformer.yaml) | `run` | CCP・ICPの有効端子モデルと周波数ごとの応答 |
| [terminal_identification.yaml](terminal_identification.yaml) | `identify` | 治具を固定した有効抵抗・容量の同定と未使用周波数での検証 |
| [多ポート二端子対の設定](../docs/manual-ja.md#multiport) | `network-run` | 指定端子間の複素S行列 |

外部から与えた時変抵抗やプラグインの記述例は [advanced/](advanced) にあります。各例を実装やモデルの動作確認に使い、装置への適用は測定面、校正、観測値との比較で判断してください。 [負荷モデルの適用範囲](../docs/manual-ja.md#loads)
