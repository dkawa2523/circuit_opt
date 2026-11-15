# 文献・公開測定データのベンチマーク

## 文献ベンチマーク一括実行

リポジトリのルートで実行します。

```powershell
uv run --frozen python bench/literature/run_suite.py --run-root runs/literature/suite
```

まず `REPORT.md` で判定を確認します。詳しい比較結果とケースごとの出力先は `evaluation.json`、実行の詳細は `logs/` にあります。実行時に参照するデータの由来・変換式・前提は以下の各ケースに記録されています。

<a id="executable-evidence"></a>

| 確認の目的 | 文献とケース | 検証する内容 |
|---|---|---|
| 文献値の再現 | [Lee 2021](p0_lee2021_bias/README.md) | 表Iの値から補正済みプラズマ端子インピーダンスを再構成 |
| 文献値の再現 | [Hargis 1994](p1_gec_ccp/README.md) | 電圧・電流・位相の32点を端子インピーダンスに変換し、電力と回路結果を照合 |
| モデル式の確認 | [Lee 2020](p1_lee2020_icp/README.md) | ICP変圧器モデルの式18・19を異なる減衰条件で比較 |
| 文献値の再現 | [Colpo 1999](p1_colpo1999_icp/README.md) | 治具の共振と黒鉛ダミーのインピーダンスを再現 |
| 整合設計の比較 | [Hargis・Sobolewski](p1_gec_ccp/README.md#independent-evidence-families) | 運転条件、装置間のばらつき、位相の感度を別々に適用して調整範囲を評価 |
| 整合設計の比較 | [Colpoの図から得た点](p1_colpo1999_icp/digitized/README.md) | 15の圧力・電力条件ごとに4通りの読み取り範囲を設け、固定回路と調整可能な回路を評価 |

Hargisの24 MHz・34 MHzは空セル共振で区別した装置群を指します。駆動周波数は13.56 MHzです。Colpoの60コーナーは15の親条件に対する図の読み取り幅です。反射電力比10%の上限と整合器の調整範囲は、それぞれの設計ケースで設定した評価条件です。

<a id="measured-cases"></a>

## 公開測定データを使った個別検証

各ケースのREADMEに実行コマンド、元データの由来、出力の読み方があります。

| 目的 | ケースと確認内容 |
|---|---|
| 等価回路の比較 | [Danzerら 2024](p2_danzer2024_rlc/README.md) — 同じ数の自由パラメータで二つの回路を比較し、同定から外した共振帯を予測 |
| 放電パルスの過渡応答 | [Minesiら 2022](p2_minesi2022_pulse/README.md) — 端子V/Iからエネルギーを積算し、総エネルギーが一致する等価回路の波形誤差と光学観測を確認 |
| 放電状態の異なる条件への適用 | [Höftら 2023](p2_hoft2023_dbd/README.md) — 拡散放電で得たR∥Cをハイブリッド放電に適用し、平均電流の変化と光学画像を照合 |
| 測定面と反射係数 | [Kopf・Roc'h 2023](p2_kopf2023_common_mode/README.md) — 二つの基準面でコモンモードの一端子対を再現し、基準インピーダンスを変更した反射係数を比較 |
| 二端子対の予測 | [Ausdenら 2025](p2_ausden2025_two_port/README.md) — 二つの校正済み取得データを使い、同定周波数帯の外で複素伝達と反射の再現範囲を評価 |
