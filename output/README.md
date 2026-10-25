# 固定版の図表と報告

このディレクトリは、生成時の版、source hash、入力、run参照を持つ数値検証のsnapshotです。現行の機能・利用仕様は[統合マニュアル](../docs/manual-ja.md)、現在位置は[開発計画](../docs/roadmap-ja.md)を参照してください。

- `reports/`: core・文献の報告と入力summaryのhash。
- `pdf/`: 計算根拠、core、個別CCP検証の図集と、その図を読むためのguide。
- `pptx/PCD_RF_Circuit_Design_Engineering_Review_JA_v12_9slides.pptx`: 保持するreview deck。その他のローカルexportは旧生成物です。

個別の探索手法比較は、記録された入力・モデル・予算についての結果です。装置の妥当性やML一般の優位性を表すものではありません。

図表の再生成は、必要なsource suiteを再現した上で[reportsの手順](../bench/reports/README.md)と[figuresの手順](../bench/figures/README.md)を使います。通常の計算と一時出力は `runs/` に保存してください。
