# PCD — 回路モデリングと設計・端子同定

PCDは、回路の順計算、負荷変動に対する整合回路の設計、校正済み複素V/Iからの有効端子モデル同定を行います。固定した装置値と、条件ごとに変更できる設備調整値を分け、宣言した電気的制約を判定します。

**[統合マニュアル・入出力仕様](docs/manual-ja.md)** を現行仕様の原本とします。目的に合うモデルの選び方から、YAML、外部SPICE、プローブ、回路図、解析グラフ、結果の読み方までをまとめています。[開発計画と残件](docs/roadmap-ja.md)では、実装済み範囲と次に進める実測検証を確認できます。

## 最初の計算

Python 3.10以上とngspiceが必要です。Windowsではコンソール版のngspiceを推奨します。リポジトリのルートで実行してください。

```powershell
uv sync --frozen --group dev
uv run --frozen python -m pcd solver-diagnose --json
uv run --frozen python -m pcd validate-case examples/circuit_transient.yaml --plan
uv run --frozen python -m pcd sim-run examples/circuit_transient.yaml --run-root runs/first --json
```

通常の設定は一つのYAMLです。既存SPICEや大きな観測表だけを必要に応じて別ファイルに置きます。図も必要なら同じYAMLへ `output: {}` を追加します。

| 目的 | マニュアル | 完全な入力例 |
|---|---|---|
| 固定回路・外部SPICEの順計算 | [任意回路と外部SPICE](docs/manual-ja.md#circuit) | [過渡RC](examples/circuit_transient.yaml) / [外部SPICE](examples/circuit_external.yaml) |
| 負荷変動に対する整合設計 | [整合回路の設計](docs/manual-ja.md#network) | [R+jX点と調整範囲](examples/rf_impedance_point_study.yaml) |
| 回路値の連続設計 | [回路値の設計](docs/manual-ja.md#circuit) | [波形目標と電圧制約](examples/circuit_design.yaml) |
| 未知負荷の有効値の同定 | [端子モデルの同定](docs/manual-ja.md#identification) | [校正済み治具とV/I観測](examples/terminal_identification.yaml) |
| プローブ・回路図・考察用グラフ | [出力仕様](docs/manual-ja.md#output) | [観測点と伝達比](examples/circuit_outputs.yaml) |

`sim-run`は一点の計算、`run`は設計探索、`identify`は端子同定です。設計の合格を終了コードで確認する場合は `run --require-acceptance` を使います。計算成功、工学的合格、同定成立、要求した出力の完了は[別々の状態](docs/manual-ja.md#results)として保存します。

## 結果と適用範囲

数値、解決済み入力、solver情報、図と対応CSVは `runs/` に保存します。JSONが返すmanifestとartifact参照を辿ってください。[実行例と数値による確認](docs/manual-ja.md#evidence)に、回路式との比較と同定例を示しています。

PCDが扱うのは宣言した電気的基準面です。有効CCP/ICP回路は端子特性を表し、化学・密度・温度・製造歩留まりの予測にはそれぞれのモデルと観測が必要です。

- [用途別の実行例](examples/README.md)
- [開発・拡張・品質検査](docs/manual-ja.md#development)
- [独立した数値検証と文献データ](bench/README.md)
- [固定版の図表と報告](output/README.md)
- [変更履歴](CHANGELOG.md)

配布wheelには統合マニュアル・図・完全なexamplesを同じ原本から収録します。インストール先とPython APIは[マニュアル](docs/manual-ja.md#quickstart)を参照してください。
