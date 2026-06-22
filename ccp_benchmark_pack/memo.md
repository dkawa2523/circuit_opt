以下の例題設定を、v6 の構成に合わせて具体化しました。
v6 にそのまま追加して試せるベンチマーク用ファイル一式も作成しています。

[CCP GEC-like Argon Benchmark Pack をダウンロード](sandbox:/mnt/data/ccp_gec_benchmark_pack.zip)

このパックは、v6 のプロジェクトルートに `examples_ccp_gec/` としてコピーして使う想定です。`dummy` solver でのワークフロー確認、`sim-netlist` による netlist 生成、`workflow-optimize → ml-score → ml-fit-surrogate` の動作確認まで行っています。実 ngspice 実行は、この環境に ngspice がないため未確認です。

---

# 提案する例題：GEC-like Argon CCP RF電極回路の波形制御・整合・プラズマ負荷ロバスト設計

例題の中心は、**半導体製造装置の低圧 CCP プラズマチャンバーを想定し、RF 電源・整合回路・電極・プラズマ等価負荷をまとめて設計する問題**です。物理設定は GEC reference cell に近い構成にしています。GEC CCP reactor は容量結合プラズマ研究の標準的なベンチマークとして使われ、COMSOL の Argon GEC CCP 例では、Ar、100 mTorr、13.56 MHz、約 10 cm 径の powered electrode、2.45 cm の電極間ギャップという条件が示されています。

この問題を選ぶ理由は、単純な RLC 回路ではなく、**RF 回路、整合、電極電圧波形、自己バイアス、高調波、プラズマ負荷変動、外部プラズマシミュレーション連携**が同時に出てくるためです。GEC reference cell と外部整合回路を含む PIC/MCC 研究でも、外部回路との結合、電力伝送、反射、高次高調波が半導体プラズマプロセスの impedance-matching 設計に重要であることが示されています。

---

# 1. 物理問題設定

## 1.1 チャンバー条件

想定する装置は、平行平板に近い CCP 型 RF プラズマチャンバーです。

| 項目                         |                                                     設定 |
| -------------------------- | -----------------------------------------------------: |
| ガス                         |                                                     Ar |
| 圧力                         |                                    100 mTorr ≈ 13.3 Pa |
| RF 基本周波数                   |                                              13.56 MHz |
| powered electrode diameter |                                                 0.10 m |
| electrode gap              |                                               0.0245 m |
| electrode area             | (A=\pi(0.05)^2 \approx 7.85\times10^{-3},\mathrm{m^2}) |
| 解析対象                       |                             電源、整合回路、電極、プラズマ等価負荷、電極電圧波形 |

COMSOL の GEC CCP Argon 例では、プラズマの電力吸収は非線形かつ複数周波数成分を含むため、電位を単純な周波数領域だけで扱えず、荷電粒子の周期時間発展を扱う必要があると説明されています。これは、今回の v6 例題で **時間波形・高調波・時変負荷**を扱う理由になります。

---

## 1.2 回路側の設計対象

v6 では、以下のように表現します。

```text
RF source
  ↓
matching network
  ↓
electrode node
  ↓
optional load
     ├─ electrode_stray
     ├─ plasma_fixed_rlc
     ├─ plasma_state_rlc
     └─ plasma_table_rlcq
```

回路トポロジ候補は、まず v6 に既にある以下で十分です。

```text
l_match
pi_match
pi_match_harmonic
```

設計変数は以下です。

```text
連続変数:
  Vsrc_amp
  Vsrc_dc
  C1, L1, C2
  Lh, Ch
  Rp, Lp, Csh
  ne, nu_m
  Ce

カテゴリ変数:
  topology_choice = l_match / pi_match / pi_match_harmonic
  load_model = electrode_stray / plasma_fixed_rlc / plasma_state_rlc
```

この設定により、v6 の有用性を 3 段階で評価できます。まず「普通の回路最適化」、次に「プラズマ等価負荷つき回路最適化」、最後に「トポロジ・負荷モデル選択を含むデータサイエンス型最適化」です。

---

# 2. プラズマ等価負荷モデル

## 2.1 固定・状態由来の RLC 負荷

低圧 CCP の最初の等価回路として、プラズマ bulk を (R_p, L_p)、シースを (C_{\mathrm{sh}}) として近似します。

[
L_p = \frac{\ell m_e}{A n_e e^2}
]

[
R_p = \nu_m L_p
]

[
C_{\mathrm{sh}} = \frac{\epsilon_0 A}{s_{\mathrm{sh}}}
]

ここで、(\ell) は bulk 長さ、(A) は電極面積、(n_e) は電子密度、(\nu_m) は電子運動量移行衝突周波数、(s_{\mathrm{sh}}) はシース厚さです。このモデルは厳密な CCP モデルではありませんが、v6 の `plasma_state_rlc` に対応し、回路側の設計空間探索には十分使いやすい縮約モデルです。

今回の benchmark pack では、代表値として次の範囲を使っています。

```yaml
ne:
  bounds: [2.0e14, 1.5e15]
  scale: log
nu_m:
  bounds: [5.0e7, 2.0e8]
  scale: log
Csh:
  bounds: [1.5e-11, 1.5e-10]
  scale: log
```

この範囲は「最初の回路設計ベンチマーク用」であり、実プロセスでは Langmuir probe、VI probe、OES、PIC/MCC、fluid model、global model などで較正する前提です。低圧・非局所・kinetic regime のプラズマでは PIC/MCC が重要であり、eduPIC の文献でも、低圧 RF プラズマの自己無撞着記述に PIC/MCC が重要な手法として説明されています。

---

## 2.2 時間変化 RLCQ 負荷

より本コードの特徴を評価するには、外部プラズマ計算から得られた時系列テーブルを使う `plasma_table_rlcq` が重要です。

benchmark pack には、以下の列を持つ合成テーブルを入れています。

```csv
time_s,Rp_ohm,Lp_H,Csh_F,electron_density_m3,momentum_collision_Hz,sheath_thickness_m
```

v6 の `plasma_table_rlcq` は、ngspice netlist に以下の形を生成します。

```spice
Rbulk p nb R = 'pwl(time, ...)'
Lbulk nb ns L = 'pwl(time, ...)'
Csh ns n Q = '(pwl(time, ...))*V(ns,n)'
```

ここで (C_{\mathrm{sh}}(t)) は、単なる `C = C(t)` ではなく、

[
Q_{\mathrm{sh}}(t) = C_{\mathrm{sh}}(t) V_{\mathrm{sh}}(t)
]

として扱います。これは、時間変化容量では

[
i(t) = \frac{dQ}{dt}
]

を意識する必要があるためです。プラズマシリーズ共振の文献でも、シースの charge-voltage 関係の非線形性や bulk inductance の時間変調が重要で、等価回路モデルに含めるべき要素として議論されています。

---

# 3. 目的関数

この例題では、単に「電圧波形を合わせる」だけではなく、回路設計としての実用性を評価できるように、以下の目的関数を推奨します。

[
J =
J_{\mathrm{wave}}

* w_h J_{\mathrm{harmonic}}
* w_p J_{\mathrm{power}}
* J_{\mathrm{constraint}}
* w_{\mathrm{proc}} J_{\mathrm{plasma}}
  ]

## 3.1 波形誤差

[
J_{\mathrm{wave}}
=================

\frac{
\sqrt{\frac{1}{N}\sum_i\left(V_e(t_i)-V_{\mathrm{ref}}(t_i)\right)^2}
}{
\sqrt{\frac{1}{N}\sum_i V_{\mathrm{ref}}(t_i)^2}+\epsilon
}
]

v6 の既存 `waveform_l2` と同じ考え方です。

---

## 3.2 高調波誤差

[
J_{\mathrm{harmonic}}
=====================

\frac{1}{|K|}
\sum_{k\in K}
\frac{| \hat{V}*e(kf_0)-\hat{V}*{\mathrm{ref}}(kf_0) |}
{| \hat{V}_{\mathrm{ref}}(kf_0) |+\epsilon}
]

電圧波形 tailoring は、RF 電圧の高調波の振幅・位相を調整し、電極表面での ion flux-energy distribution を制御する方法として研究されています。Schüngel らの文献では、印加 RF 電圧波形の高調波振幅・位相を調整することで、電極表面の ion flux-energy distribution のピーク位置、すなわちイオンエネルギーとピーク内フラックスを制御できると説明されています。

---

## 3.3 電力・電流・安全制約

回路側の proxy として、保存波形から

[
P_{\mathrm{proxy}} = \langle V_e(t) I_{\mathrm{src}}(t) \rangle
]

[
V_{\mathrm{peak}} = \max_t |V_e(t)|
]

[
I_{\mathrm{rms}} = \sqrt{\langle I(t)^2\rangle}
]

を計算します。

benchmark pack では、v6 の ML plugin として `ccp_waveform_power_proxy` を入れています。これは `waveform.csv` と `target waveform CSV` だけを読み、`loss`, `normalized_rmse`, `harmonic_error`, `avg_power_proxy_W`, `dc_bias_proxy_V`, `v_peak_abs_V`, `i_rms_A` を返します。v6 の思想通り、**この objective は ngspice 計算部を import しない**ため、実測波形や外部連成波形にもそのまま適用できます。

---

## 3.4 プラズマ側のプロセス指標

v6 の現状では、直接の PIC/fluid 計算は行いません。外部プラズマシミュレータから以下を受け取る設計にします。

```yaml
plasma_outputs:
  equivalent_load:
    Rp_time
    Lp_time
    Csh_time
  process_metrics:
    ion_flux_m2_s
    ion_energy_peak_eV
    ion_energy_fwhm_eV
    dc_self_bias_V
    plasma_potential_V
    uniformity_index
```

COMSOL Plasma Module の説明では、低温プラズマの輸送方程式を Poisson 方程式と自己無撞着に解くこと、電子平均エネルギー方程式、電子衝突反応、表面反応、電極での charged particle flux、外部回路との連携などが説明されています。したがって、v6 側は全プラズマ物理を抱え込まず、**波形・回路パラメータ・等価負荷・プロセスメトリクスを交換する境界**に徹するのが妥当です。

---

# 4. benchmark pack の内容

作成した pack には、以下を入れています。

```text
ccp_benchmark_pack/
  README.md

  ccp_gec_level1_fixed_match.yaml
  ccp_gec_level2_timevarying_plasma.yaml
  ccp_gec_level3_topology_and_load_choice.yaml

  target_gec_fundamental.csv
  target_gec_tailored.csv
  plasma_table_gec_argon_synthetic.csv
  plasma_scenarios.csv

  plugins/
    ccp_metrics.py
```

---

## Level 1: 固定・状態由来プラズマ負荷での RF 整合

目的は、v6 の **simulation layer と ML layer の基本動作**を確認することです。

```yaml
case_id: ccp_gec_level1_fixed_match

source:
  type: sine_voltage
  amplitude_V: Vsrc_amp
  frequency_Hz: 13560000

circuit:
  builder: pi_match

load:
  name: plasma_state_rlc
  electron_density_m3: ne
  momentum_collision_Hz: nu_m
  bulk_length_m: 0.0245
  area_m2: 0.00785398163397
  Csh_F: Csh

target:
  waveform_file: target_gec_fundamental.csv
  objective: ccp_waveform_power_proxy
```

評価したい観点は以下です。

```text
- netlist が読みやすく生成されるか
- ngspice/dummy の artifact が保存されるか
- sim_manifest.json + waveform.csv を境界に ML scoring できるか
- C/L と plasma state variables の探索が可能か
```

---

## Level 2: 時間変化プラズマ負荷での waveform tailoring

目的は、外部プラズマシミュレーション結果を模擬した `plasma_table_rlcq` を使い、**時変負荷・高調波・波形追従**を評価することです。

```yaml
case_id: ccp_gec_level2_timevarying_plasma

circuit:
  builder: pi_match_harmonic

load:
  name: plasma_table_rlcq
  table_file: plasma_table_gec_argon_synthetic.csv

target:
  waveform_file: target_gec_tailored.csv
  objective: ccp_waveform_power_proxy
  harmonics: [1, 2, 3]
```

この問題は Level 1 より難しく、以下を見られます。

```text
- time-varying Rp/Lp/Csh table を netlist に反映できるか
- Q = C(t)V 型のシース容量が生成されるか
- 高調波を含む目標波形に対する誤差評価ができるか
- 外部プラズマ計算との連携境界が明確か
```

GEC CCP の COMSOL 例でも、電極電流は正弦波でなく高次高調波での電力吸収が見られることが示されています。このため、高調波成分を評価指標に入れることは、単なる数学的な都合ではなく物理的にも意味があります。

---

## Level 3: トポロジ・負荷モデル選択を含むデータサイエンス評価

目的は、v6 の **ML/最適化基盤としての有用性**を評価することです。

```yaml
variables:
  topology_choice:
    choices: [l_match, pi_match, pi_match_harmonic]

  load_model:
    choices: [electrode_stray, plasma_fixed_rlc, plasma_state_rlc]

circuit:
  builder: $topology_choice

load:
  name: $load_model
```

この問題では、連続値だけではなく、カテゴリ変数も含みます。

```text
連続:
  Vsrc_amp, C1, L1, C2, Lh, Ch, Ce, Rp, Lp, Csh, ne, nu_m

カテゴリ:
  topology_choice
  load_model
```

v6 の `random` / `optuna` optimizer を使うことで、候補生成、simulation batch、scoring、surrogate 学習を分離して評価できます。Optuna は define-by-run API、探索・pruning、分散・軽量運用を意識して設計された最適化フレームワークとして提案されています。

---

# 5. 推奨ワークフロー

## 5.1 Simulation only

ngspice 計算部だけを確認します。

```bash
pcd sim-netlist examples_ccp_gec/ccp_gec_level2_timevarying_plasma.yaml \
  --out /tmp/ccp_level2.cir

pcd sim-run examples_ccp_gec/ccp_gec_level2_timevarying_plasma.yaml \
  --solver ngspice_cli \
  --run-root runs/ccp_level2_sim
```

この段階で確認する artifact は以下です。

```text
netlist.cir
waveform.csv
sim_manifest.json
solver.log
circuit_to_plasma.json
```

ngspice は open-source SPICE simulator で、netlist を入力し、電流・電圧などをグラフまたはデータファイルとして出力できると説明されています。今回の v6 では、この ngspice 実行部と ML/最適化部を分けるため、`sim_manifest.json + waveform.csv` を境界 artifact にしています。

---

## 5.2 ML only

既存の波形や simulation record だけを採点します。

```bash
pcd ml-score examples_ccp_gec/ccp_gec_level2_timevarying_plasma.yaml \
  runs/ccp_level2_sim
```

この段階では ngspice を呼びません。実験波形や外部プラズマシミュレータの連成波形を `sim_manifest.json + waveform.csv` 形式に変換すれば、ML layer だけで評価できます。

---

## 5.3 Batch study

候補生成、シミュレーション、採点を分けて実行します。

```bash
pcd ml-propose examples_ccp_gec/ccp_gec_level3_topology_and_load_choice.yaml \
  --n 100 \
  --out runs/ccp_candidates.csv
```

```bash
pcd sim-batch examples_ccp_gec/ccp_gec_level3_topology_and_load_choice.yaml \
  runs/ccp_candidates.csv \
  --solver ngspice_cli \
  --run-root runs/ccp_batch_001
```

```bash
pcd ml-score examples_ccp_gec/ccp_gec_level3_topology_and_load_choice.yaml \
  runs/ccp_batch_001
```

```bash
pcd ml-fit-surrogate runs/ccp_batch_001 \
  --out runs/ccp_batch_001/surrogate.json
```

これにより、回路担当者は ngspice batch を担当し、データサイエンス担当者は `scores.csv` や `surrogate.json` を使って分析する、という分業ができます。

---

## 5.4 Closed-loop optimization

閉ループ最適化を一括で確認します。

```bash
pcd workflow-optimize examples_ccp_gec/ccp_gec_level3_topology_and_load_choice.yaml \
  --optimizer random \
  --solver ngspice_cli \
  --n-trials 100 \
  --run-root runs/ccp_closed_loop
```

内部的には以下です。

```text
optimizer.ask
  ↓
simulate_case
  ↓
score_record
  ↓
optimizer.tell
  ↓
repeat
```

この workflow は便利ですが、v6 の設計評価としては、まず `ml-propose → sim-batch → ml-score` を個別に動かす方が、simulation layer と ML layer の分離が確認しやすいです。

---

# 6. 外部プラズマシミュレーションとの接続

今回の例題では、外部プラズマ計算は次の 3 段階で導入するのがよいです。

## Stage A: 合成 plasma table

benchmark pack に含めた `plasma_table_gec_argon_synthetic.csv` を使います。

これは以下の簡易式で生成しています。

[
L_p(t) = \frac{\ell m_e}{A n_e(t)e^2}
]

[
R_p(t) = \nu_m(t)L_p(t)
]

[
C_{\mathrm{sh}}(t) = \frac{\epsilon_0 A}{s_{\mathrm{sh}}(t)}
]

この段階は、v6 のデータ I/O、netlist 生成、時変負荷、ML scoring を評価するためのものです。

---

## Stage B: 0D/global/Boltzmann 系モデル

BOLSIG+ や LXCat を使い、電子輸送係数や衝突係数を得て、`plasma_state_rlc` または plasma table に変換します。BOLSIG+ は、弱電離ガス中の電子 Boltzmann 方程式を解き、基礎的な断面積データから電子輸送係数と衝突 rate coefficient を得るためのプログラムとして説明されています。

LXCat は、低温プラズマ modeling に必要な電子・イオン衝突断面積、swarm parameters、反応率、energy distribution などを収集・表示・ダウンロードする open-access website として説明されています。

ThunderBoltz のような 0D DSMC 系コードも候補です。ThunderBoltz は arbitrary cross sections を扱える軽量な 0D DSMC code で、LXCat database との互換や Python interface、電子輸送・反応率の後処理などが説明されています。

---

## Stage C: 1D/2D PIC-MCC または fluid モデル

高忠実度化する場合は、PIC/MCC や fluid model で以下を出します。

```text
回路 → プラズマ:
  electrode_voltage_waveform
  electrode_current_waveform
  delivered_power
  geometry
  gas / pressure
  wall material
  secondary electron yield

プラズマ → 回路:
  Rp(t)
  Lp(t)
  Csh(t)
  plasma potential
  dc self-bias
  ion flux
  ion energy distribution
  uniformity index
```

eduPIC は 1D electrostatic PIC/MCC code を CCP 向けに提示しており、学習・拡張の starting tool として位置づけられています。研究用・教育用の外部プラズマモデルと v6 の境界確認に適しています。

2025 年の Boltzsim 文献では、1D 空間の電子 Boltzmann transport equation solver が RF glow discharge plasma に適用され、低圧、特に 1 Torr 未満では近似手法との差が大きいことが報告されています。今回の GEC-like 条件は 100 mTorr なので、低圧 kinetic/nonlocal 効果を無視しすぎない設計評価が必要です。

---

# 7. この例題で評価できること

この benchmark で、v6 の有用性を以下の観点から評価できます。

| 評価観点                      | 確認方法                                                        |
| ------------------------- | ----------------------------------------------------------- |
| 汎用回路として使えるか               | `load: none` や `electrode_stray` と比較                        |
| プラズマ等価負荷を扱えるか             | `plasma_fixed_rlc`, `plasma_state_rlc`, `plasma_table_rlcq` |
| 時変素子を扱う境界があるか             | `plasma_table_gec_argon_synthetic.csv`                      |
| ngspice 計算部と ML 部が分離しているか | `sim-run` 後に `ml-score` を別実行                                |
| 外部データを採点できるか              | `waveform.csv + sim_manifest.json` を ML layer で読む           |
| トポロジ探索できるか                | `topology_choice` を categorical 変数化                         |
| surrogate 学習できるか          | `ml-fit-surrogate`                                          |
| 外部プラズマ連携できるか              | `circuit_to_plasma.json` と plasma table                     |
| 実務上の限界が見えるか               | 固定 RLC と時変 RLCQ のスコア比較                                      |

---

# 8. 期待される分析結果

最初に見るべき結果は、以下です。

```text
scores.csv:
  loss
  normalized_rmse
  harmonic_error
  avg_power_proxy_W
  dc_bias_proxy_V
  v_peak_abs_V
  i_rms_A
  topology_choice
  load_model
  C1, L1, C2, Lh, Ch
```

例えば、以下を比較します。

```text
1. electrode_stray vs plasma_fixed_rlc vs plasma_state_rlc
2. l_match vs pi_match vs pi_match_harmonic
3. fixed plasma load vs time-varying plasma load
4. fundamental target vs tailored waveform target
5. random search vs optuna
```

有用な可視化は以下です。

```text
- target waveform と electrode waveform の重ね描き
- FFT 振幅スペクトル
- loss vs trial
- topology_choice ごとの loss boxplot
- load_model ごとの harmonic_error boxplot
- C1/L1/C2 と loss の散布図
- surrogate predicted_loss と actual_loss の比較
```

この分析により、v6 が単なる ngspice wrapper ではなく、**回路計算 artifact を独立に保存し、後段のデータサイエンス・最適化で再利用できる基盤**になっているかを評価できます。

---

# 9. 注意点

この例題は、最初の benchmark としては有用ですが、物理的には以下の限界があります。

```text
- plasma_fixed_rlc は線形等価負荷なので、自己バイアスや非線形シースを十分には表現できない
- plasma_state_rlc は一様 bulk 近似であり、空間分布や非局所電子加熱は表現しない
- plasma_table_rlcq は外部プラズマ結果を入れる器であり、v6 自体が plasma kinetics を解くわけではない
- ion energy distribution や ion flux は、現状では proxy または外部モデル出力として扱うべき
- 実プロセスでは secondary electron emission、壁反応、ガス化学、表面反応、実形状、RF ケーブル寄生を追加する必要がある
```

COMSOL Plasma Module の説明でも、プラズマ系には流体、化学反応、表面化学、物理 kinetics、熱・物質移動、電磁気が関わるとされており、CCP、ICP、microwave plasma、global model、EEDF、ion energy distribution、etching/deposition などを用途ごとに扱う必要があります。したがって、v6 の役割は **高忠実度プラズマソルバの代替**ではなく、**回路設計・波形制御・等価負荷連携・最適化の実験基盤**として位置づけるのが適切です。

---

# 10. 結論

最初に取り組む例題としては、次を推奨します。

```text
GEC-like Argon CCP における
RF matching network + electrode + plasma equivalent load の
電極電圧波形追従・高調波制御・プラズマ負荷ロバスト最適化
```

この例題なら、v6 の重要な特徴をすべて評価できます。

```text
- ngspice計算部だけ使える
- ML/最適化部だけ使える
- 外部プラズマ計算データを table として入れられる
- プラズマなし・固定プラズマ・時変プラズマを比較できる
- トポロジ選択と素子値最適化を同じ枠組みで扱える
- 後から PIC/MCC、fluid、BOLSIG+、LXCat、COMSOL などへ拡張できる
```

作成した benchmark pack では、この問題を Level 1〜3 に分けています。まず `dummy` solver で v6 のワークフロー確認を行い、その後 `ngspice_cli` に切り替えて物理波形を確認し、最後に外部プラズマシミュレータの出力を `plasma_table_rlcq` に接続する流れがよいです。
