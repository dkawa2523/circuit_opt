# 用途別の実行例

設定方法と入出力の意味は[統合マニュアル](../docs/manual-ja.md)を参照してください。通常は一つの `pcd.rf.v1` YAMLを編集し、既存回路や大きな観測表だけを別ファイルに置きます。

| 入力例 | コマンド | 確認する用途 |
|---|---|---|
| [circuit_transient.yaml](circuit_transient.yaml) | `sim-run` | 固定RC回路の立上り応答 |
| [circuit_external.yaml](circuit_external.yaml) | `sim-run` | 既存SPICEの電源を保持したAC解析 |
| [circuit_outputs.yaml](circuit_outputs.yaml) | `sim-run` | プローブ、回路図、波形・Bode・AC・高調波と対応CSV |
| [circuit_design.yaml](circuit_design.yaml) | `run` | 波形目標とpeak制約を使ったCの連続設計 |
| [terminal_identification.yaml](terminal_identification.yaml) | `identify` | 固定治具、latent R/C、fit/holdoutの複素V/I |
| [rf_impedance_point_study.yaml](rf_impedance_point_study.yaml) | `run` | 少数のR+jX点に対する整合と設備調整 |
| [rf_impedance_frequency_table.yaml](rf_impedance_frequency_table.yaml) | `run` | 測定周波数の独立した負荷点 |
| [rf_quasi_static_profile.yaml](rf_quasi_static_profile.yaml) | `run` | 独立AC snapshotによる緩やかな負荷変動 |
| [rf_component_stress.yaml](rf_component_stress.yaml) | `run` | 部品端子の電気ストレスと有効損失 |
| [rf_ccp_lumped.yaml](rf_ccp_lumped.yaml) / [rf_icp_transformer.yaml](rf_icp_transformer.yaml) | `run` | 有効CCP/ICPの端子モデル |

チェックアウトではコマンドを `uv run --frozen python -m pcd` に続け、実行前に `solver-diagnose --json` と `validate-case INPUT --plan` を実行します。`run --require-acceptance` は工学的不合格も非ゼロ終了にします。

[advanced/](advanced)は従来の `case_yaml.v1` と拡張の例です。通常利用の設定へ追加する必要はありません。例の数値は合成・宣言した電気モデルの検証であり、装置の妥当性は測定面・校正・独立観測で確認してください。
