# 撤去した機械学習実験の結果

更新日: 2026-10-04
状態: 履歴資料（製品仕様・利用手順ではない）

## 結論

旧ML実装は、回路計算基盤に残すだけの有用性を示せなかったためR0で削除した。ここに残す数値は、同じ問題を
名前だけ変えて再実装しないための判断記録である。CLI、データschema、モデル形式の互換性は維持しない。

## 実施した評価

1. 完了済みAC studyから108 sample、3 topology familyのcorpusを作った。
2. 訓練平均、ridge、二層MLP、component/net relational GNNを同じsplitで比較した。
3. 別に、81候補のRC過渡問題と81候補のRF整合問題で、固定KNN順位と101本のrandom順位を比較した。
4. 選択候補は通常のngspice studyで再計算した。

## 不合格結果

正規化RMSEは次の通りで、値が小さいほど良い。

| 評価 | 定数baseline | ridge | MLP | GNN |
|---|---:|---:|---:|---:|
| 未知design | 0.534 | 0.592 | 0.763 | 0.801 |
| 未知condition | 0.289 | 0.460 | 1.479 | 1.535 |
| leave-one-topology-out | 1.273 | 1.317 | 1.638 | 1.463 |

全aggregate protocolで定数baselineが最良だった。固定候補ランキングのngspice評価削減率はRCで14.3%、
RFで29.6%であり、事前に固定した30%基準を両方で満たさなかった。したがって、モデル保存、online candidate
proposal、Bayesian optimization readyという主張は認めなかった。

## 原因

- 三つの限定的な回路だけでは、配線と運転条件をまたぐ学習範囲を支えられない。
- 現行graphは二端子p/nと三つのportに限定され、外部netlist、磁気結合、時変素子を表せなかった。
- 固定候補を後から並べ替える評価であり、未評価点を逐次提案してsolver費用を減らす製品機能ではなかった。
- 回路の固定量、設計量、運転量、同定対象が一つの問題定義で明確になっていなかった。
- プラズマ端子同定と回路素子最適化が成立する前に、代理モデルだけを先行させていた。

## 再導入条件

MLは[再設計計画](platform-reset-plan-ja.md)のR6まで再導入しない。再導入時は、複数装置または独立campaignを
含むgroup holdout、単純baselineとの同一条件比較、同じsolver予算でのregretまたはfeasible到達率、事前に
固定したngspice呼出し削減率を全て評価する。最終候補は必ず通常の全scenario ngspice経路で再計算する。

旧実装の詳細が必要な場合はGit履歴を参照する。この履歴文書を根拠に旧CLIや旧schemaを復活させない。
