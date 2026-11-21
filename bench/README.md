# ベンチマークの選び方

確認したい内容に合う検証を選んでください。通常の回路設定や結果の読み方は[統合マニュアル](../docs/manual-ja.md)にあります。

| 確認したい内容 | リポジトリのルートで実行するコマンド | 実行後に読むファイル |
|---|---|---|
| 解析式とSPICE結果の一致、候補の全列挙、制約判定、基準面 | `uv run --frozen python bench/run_suite.py --run-root runs/benchmark_suite` | `REPORT.md`、`benchmark_result.json` |
| 公開CLIで入力から計算、保存、図表まで進められること | `uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance` | `REPORT.md`、`release_result.json`。詳細は[利用経路の受入検証](release/README.md) |
| 文献の値・式の再現と、それを使った整合設計の判定 | `uv run --frozen python bench/literature/run_suite.py --run-root runs/literature/suite` | `REPORT.md`、`evaluation.json`。詳細は[文献ベンチマーク](literature/README.md) |
| 公開測定データを使ったモデル比較と適用範囲 | [測定データの一覧](literature/README.md#measured-cases)からケース別に実行 | 各ケースのREADMEを参照 |

結果は指定した `--run-root` の下に保存されます。ベンチマークは毎回新しい出力先で計算し、`passed`に全体の判定を保存します。合格は、期待している電気特性や設計判定が再現されたことを表します。制約を満たさない設計例も、期待した不合格理由が再現されればベンチマークは合格です。

## 回路の基礎検証

16ケースで、設計候補・外部条件・調整値の組み合わせを計411件評価します。ケースの合格条件、比較する複素インピーダンス、採用されるべき調整値は[期待値ファイル](expectations.yaml)に記載しています。

| ケース | 確認すること |
|---|---|
| A1〜A5 | L形・π形回路、高調波分岐を持つ回路の基本波応答、有効CCP・ICP負荷の周波数特性 |
| B1〜B4 | 固定ハードウェア、調整範囲と余裕、独立に与えた周波数点 |
| B5〜B8 | 部品と電源の制約、設計候補と調整値の全列挙、実現部品値のコーナー条件 |
| D1〜D3 | 二つの基準面で等価な治具表現の一致と、治具の二重計上による応答差 |

入力は合成データです。B8の適合割合 `3/8` は、等しい重みを持つ8つの指定コーナーのうち3つが制約を満たしたことを示します。D1〜D3を比較するときは、治具がどちらの基準面に含まれるかをケースの設定で確認してください。
