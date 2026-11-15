# PCD — 回路モデリングと設計・端子同定

PCDは、回路の電圧・電流の計算、複数条件に対する回路設計、校正した複素電圧・電流からの有効端子モデルの同定に使えます。設計では、装置に固定する部品値と、運転条件ごとに変更できる調整値を指定し、各条件の電気的制約を評価できます。

## 最初の計算

Python 3.10以上、uv、ngspiceを用意します。Windowsではコンソール版の `ngspice_con.exe` を推奨します。次のPowerShellコマンドをリポジトリのルートで実行すると、固定RC回路の立ち上がりを計算できます。 [インストールとソルバー設定](docs/manual-ja.md#quickstart)

```powershell
uv run --frozen --no-dev python -m pcd solver-diagnose --json
uv run --frozen --no-dev python -m pcd validate-case examples/circuit_transient.yaml --plan
$rc = uv run --frozen --no-dev python -m pcd sim-run examples/circuit_transient.yaml --run-root runs/first --json | ConvertFrom-Json
$rc.run_dir
```

診断で `batch_runnable: true` を確認し、計算後は表示されたフォルダーの `summary.json` と `data/transient.csv` を開きます。コンデンサー端子の `voltage_V` は2 µsで約4.32 Vです。保存した波形からグラフを作る場合は、次を実行して `runs/first/analysis/transient_response.png` を開きます。

```powershell
uv run --frozen --no-dev python -m pcd analyze $rc.run_dir --out runs/first/analysis
```

自分の回路を計算するときは用途に近いYAMLをコピーし、回路、条件、必要な観測を編集します。参照するSPICEやCSVも一緒にコピーしてください。計算時に標準の図表を生成する場合は `output: {}` を追加します。 [用途別の入力例と実行コマンド](examples/README.md)

## 設定と結果を調べる

- YAML、外部SPICE、回路モデル、観測点の指定。 [統合マニュアル](docs/manual-ja.md)
- 計算の成否、条件別の制約、同定の成立、図表の生成状況。 [出力の読み方](docs/manual-ja.md#results)
- 指定した回路端子間の小信号S行列。 [多ポート解析](docs/manual-ja.md#multiport)
- 配布wheelに同梱するサンプルとマニュアルの場所。 [配布wheelの使い方](docs/manual-ja.md#quickstart)
- [ベンチマークと文献データ](bench/README.md) · [開発と品質検査](docs/manual-ja.md#development) · [変更履歴](CHANGELOG.md)

有効CCP・ICP回路が表すのは、指定した電気的基準面で見たプラズマ負荷の端子特性です。化学反応、密度、温度、製造歩留まりを求める場合は、対象量を検証できる物理モデルと観測が必要です。 [負荷モデルの選び方](docs/manual-ja.md#loads)
