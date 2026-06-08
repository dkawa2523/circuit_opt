# CCP Benchmark Pack 実行・可視化・評価レポート

## 結論

今回実行した内容は、`ccp_benchmark_pack` の3ケースに対する次の検証です。

- YAMLケース定義の検証
- dummyソルバによる最適化ワークフローの実行
- 波形・メトリクス・最適化履歴・サロゲート学習の生成
- ngspice向けネットリスト生成
- Schemdrawベースの回路図可視化
- `ngspice_cli` による実行試行

重要な点として、この環境では `ngspice` 実行ファイルが見つからなかったため、実回路シミュレーションとしてのngspice過渡解析は完了していません。ngspice試行はすべて `missing_executable=True` として失敗runに記録され、`loss=1e30`、`metrics_status=failed`、`metrics_reason=simulation_failed` が保存されています。

したがって、今回の評価結果は「製品基盤としてワークフロー、記録、可視化、失敗検出が動作するか」の評価であり、「物理的なプラズマ回路設計として妥当な最適解が得られたか」の評価ではありません。

## 対象ベンチマーク

`ccp_benchmark_pack` には GEC 風のアルゴンCCP回路ベンチマークが3段階で含まれています。

| Level | YAML | 目的 |
|---|---|---|
| Level 1 | `ccp_gec_level1_fixed_match.yaml` | 固定マッチング回路と状態由来RLCプラズマ負荷の基本確認 |
| Level 2 | `ccp_gec_level2_timevarying_plasma.yaml` | 合成プラズマテーブルを用いた時変負荷ケース |
| Level 3 | `ccp_gec_level3_topology_and_load_choice.yaml` | トポロジ、負荷モデル、連続パラメータを含む混合探索 |

すべてのケースは非strict validationでは通過しました。警告として `dummy` ソルバ使用が検出されています。これは研究・スクリーニング用途では許容されますが、製品安全モードや物理検証では `ngspice_cli` などの実ソルバに置き換える必要があります。

## dummyベンチマーク実行

出力ルート:

```text
runs/ccp_benchmark_eval
```

実行条件:

| 項目 | 内容 |
|---|---|
| optimizer | `random` |
| solver | `dummy` |
| trials | 各ケース30 |
| failed trials | 全ケース0 |

生成物:

- `runs/ccp_benchmark_eval/benchmark_summary.csv`
- `runs/ccp_benchmark_eval/surrogate_summary.csv`
- `runs/ccp_benchmark_eval/<case>/summary.csv`
- `runs/ccp_benchmark_eval/<case>/trial_*/`
- `runs/ccp_benchmark_eval/figures/plasma_table_rlc.png`
- `runs/ccp_benchmark_eval/figures/loss_curves.png`
- `runs/ccp_benchmark_eval/figures/best_waveform_overlays.png`
- `runs/ccp_benchmark_eval/figures/best_metric_summary.png`
- `runs/ccp_benchmark_eval/figures/level3_category_performance.png`

## dummyベンチマーク評価結果

| Case | Trials | Failed | Best loss | Norm. RMSE | Harmonic error | Power error | Best peak V |
|---|---:|---:|---:|---:|---:|---:|---:|
| Level 1 fixed/state RLC | 30 | 0 | 0.093730 | 0.038031 | 0.037997 | 0.999996 | 311.42 V |
| Level 2 time-varying plasma | 30 | 0 | 0.491199 | 0.271650 | 0.678226 | 0.999847 | 315.83 V |
| Level 3 topology/load choice | 30 | 0 | 0.501831 | 0.284901 | 0.667723 | 0.999998 | 280.68 V |

解釈:

- Level 1は波形RMSEと高調波誤差が小さく、基盤のスモークテストとして最も良好です。
- Level 2とLevel 3は、dummyソルバでは時変プラズマ負荷やトポロジ差を物理的に解けないため、目標波形への一致度は限定的です。
- Power errorが全ケースでほぼ1.0です。これはdummyソルバの電流・電力 proxy が物理評価には不十分であることを示しています。

## サロゲート評価

| Case | Train rows | Features | Train RMSE | Train R2 | CV RMSE |
|---|---:|---:|---:|---:|---:|
| Level 1 fixed/state RLC | 30 | 7 | 0.185419 | 0.112457 | 0.274372 |
| Level 2 time-varying plasma | 30 | 7 | 0.122149 | 0.319327 | 0.225863 |
| Level 3 topology/load choice | 30 | 18 | 0.093336 | 0.540628 | 0.214386 |

Level 3はカテゴリ特徴量を含むため、特徴量数が18に増えています。学習R2は最も高いものの、CV RMSEはまだ大きく、30試行だけでは設計順位を確定するには不十分です。現状のサロゲートは候補絞り込み支援として扱うべきです。

## Level 3カテゴリ傾向

| Category | Count | Min loss | Mean loss |
|---|---:|---:|---:|
| topology `l_match` | 11 | 0.501831 | 0.638352 |
| topology `pi_match` | 12 | 0.502475 | 0.640974 |
| topology `pi_match_harmonic` | 7 | 0.522930 | 0.697173 |
| load `plasma_fixed_rlc` | 16 | 0.501831 | 0.688387 |
| load `plasma_state_rlc` | 5 | 0.511038 | 0.611989 |
| load `electrode_stray` | 9 | 0.512192 | 0.613293 |

今回の最小lossは `l_match` と `plasma_fixed_rlc` の組み合わせで得られました。ただし、カテゴリ間の平均差は大きくなく、ランダム30試行だけでトポロジ優劣を断定するべきではありません。

## ngspiceネットリスト生成と回路図可視化

出力ルート:

```text
runs/ccp_ngspice_visual_eval
```

実行したこと:

- 各CCPケースからngspice向けネットリストを生成
- `load_model` を展開し、プラズマ負荷のRLC要素を回路図へ反映
- Schemdrawで通常の回路記号に近い図を生成
- `ngspice_cli` 実行を試行し、失敗状態をmanifestとmetricsに記録

生成物:

- `runs/ccp_ngspice_visual_eval/netlists/*.cir`
- `runs/ccp_ngspice_visual_eval/netlists/*_netlist_summary.json`
- `runs/ccp_ngspice_visual_eval/figures/level1_fixed_match_schematic.png`
- `runs/ccp_ngspice_visual_eval/figures/level2_timevarying_plasma_schematic.png`
- `runs/ccp_ngspice_visual_eval/figures/level3_topology_load_choice_schematic.png`
- `runs/ccp_ngspice_visual_eval/figures/component_counts.png`
- `runs/ccp_ngspice_visual_eval/figures/ngspice_attempt_status.png`
- `runs/ccp_ngspice_visual_eval/ngspice_attempt_summary.csv`

## ngspice試行結果

| Case | Circuit | Load | Nodes | Expanded components | ngspice status | 原因 |
|---|---|---|---:|---:|---|---|
| Level 1 fixed/state RLC | `pi_match` | `plasma_state_rlc` | 5 | 8 | failed | `ngspice` 実行ファイルなし |
| Level 2 time-varying plasma | `pi_match_harmonic` | `plasma_table_rlcq` | 6 | 10 | failed | `ngspice` 実行ファイルなし |
| Level 3 topology/load choice | `pi_match` | `plasma_state_rlc` | 5 | 8 | failed | `ngspice` 実行ファイルなし |

展開後の部品数:

| Case | V source | Capacitors | Inductors | Resistors |
|---|---:|---:|---:|---:|
| Level 1 fixed/state RLC | 1 | 3 | 2 | 2 |
| Level 2 time-varying plasma | 1 | 4 | 3 | 2 |
| Level 3 topology/load choice | 1 | 3 | 2 | 2 |

## 製品基盤としての評価

良い点:

- ケース検証、波形検証、manifest provenance、failed run記録、サロゲートschema記録が入っており、研究用途から運用寄りへ近づいています。
- ngspice実行ファイル欠落時も黙って成功扱いにせず、failed metricsとして明示できます。
- `runs/` 配下に、case、params、netlist、waveform、metrics、manifest、solver logが保存され、監査可能性があります。
- 回路図可視化はmatplotlibで無理に描く方式から、Schemdrawベースの回路記号描画へ改善されています。

制約:

- 今回の数値最適化はdummyソルバ結果であり、物理設計の結論には使えません。
- ngspice本体が未導入のため、過渡解析波形、電流、電力、プラズマ負荷の物理整合性は未検証です。
- Level 2の時変プラズマ表は、現在のngspiceネットリストでは完全な時間依存素子として解かれたわけではありません。実運用ではPWL/制御電源/外部結合など、ngspiceで解釈可能な表現へ落とす必要があります。
- サロゲートの訓練データは各ケース30点で少なく、CV RMSEも大きいため、製品判断には追加試行と外部検証が必要です。

## 推奨される次ステップ

1. `ngspice` をインストールし、PATHへ追加する。
2. `pcd sim-run ccp_benchmark_pack/ccp_gec_level1_fixed_match.yaml --solver ngspice_cli --strict-exit` で実ソルバ経路を確認する。
3. Level 2の時変プラズマ負荷を、ngspiceで解けるPWLまたは制御素子表現へ明示的に変換する。
4. dummy結果とngspice結果を分離して集計し、物理評価レポートではdummy結果を除外する。
5. 試行数を増やし、外部測定、流体モデル、global model、PIC/MCCなどの参照データで物理妥当性を確認する。

## 最終判定

このコードベースは、製品基盤として必要な「検証、記録、失敗runの明示、ネットリスト入出力、回路図可視化、サロゲート学習」の土台を備えています。

一方で、今回の環境ではngspiceが実行できていないため、CCPプラズマ回路の物理設計プラットフォームとして完成と判断するにはまだ早いです。現時点の到達点は、製品運用へ進めるための基盤検証と、ngspice導入後に物理検証へ進む準備ができた段階です。
