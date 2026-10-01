# 2周波＋DC重畳エッチングCCPの動的インピーダンス検証

## 1. この問題で確認すること

このベンチマークは、300 mm級の平行平板エッチングチャンバーを想定し、次の二つを分離して確認する。

1. 既知の60 MHz上部RF、2 MHz下部RF、負DC pulse、外部回路、時間変化する上下sheath容量・bulk抵抗・
   electron inertia相当inductanceから、ngspiceがwafer電圧、上部50 Ω参照面の反射電圧、bulk電流を正しく計算するか。
2. wafer電圧だけを既知としたとき、あらかじめ宣言した狭い候補範囲から、pulse-on時の
   `Rp`, `Lp`, `Cs,u`, `Cs,w` を回収できるか。反射電圧はlossに使わずhold-out検証に残す。

これは回路レベルの順問題と有界同定問題である。plasma密度、電子温度、化学種、ion energy distribution、
etch rateを自己整合に解くplasma solverではない。

## 2. 文献から採用した構造と、採用していない主張

### 2.1 装置クラス

[Kim et al., *Effect of source frequency and pulsing on the SiO2 etching characteristics of dual-frequency capacitively coupled plasma*](https://doi.org/10.7567/JJAP.54.01AE07)
は、300 mm DF-CCP、20 mm electrode gap、上部13.56–60 MHz source、下部2 MHz biasを用いる。今回の
「上から上部電極―upper sheath―bulk―wafer sheath―wafer/ESC」の順序、60 MHz上部source、2 MHz下部biasは
この装置クラスを基礎にした。

[Yamaguchi et al., *Direct current superposed dual-frequency capacitively coupled plasmas in selective etching of SiOCH over SiC*](https://doi.org/10.1088/0022-3727/45/2/025203)
は、上部60 MHz VHF電極への負DC重畳と下部RF-biased waferを扱い、-800～-1200 VのDC条件を報告する。
今回の上部 `0 → -800 V` は、この接続と電圧規模を回路問題へ取り込んだものである。

ただし、前者は60/2 MHzでDC重畳を扱わず、後者の下部biasは13.56 MHzである。したがって今回の
60 MHz + 2 MHz + pulsed DCは、二つの報告から作った意図的なverification problemであり、特定装置recipeの
再現ではない。400 kHz pulse周波数、RF電圧振幅、外部素子値、時間変化profileも計算検証用の工学的仮定である。

### 2.2 plasma等価回路

[Kim, Lee, and Hong, *Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum Variable Capacitor Positions of Impedance Matching Unit*](https://doi.org/10.3390/electronics14102022)
はCCP負荷をsheath capacitance `Cs`、bulk plasma inductance `Lp`、bulk resistance `Rp`で整理し、
`Rp ∝ νm/ne`、`Lp ∝ 1/ne`、`Cs = ε0 A/ds` の対応を示す。今回もこの最小電気的構成を使うが、上下sheathを
別容量にし、各sheathの有限lossをparallel resistanceで表す。

[Schmidt, Mussenbrock, and Trieschmann, *Consistent simulation of capacitive radio-frequency discharges and external matching networks*](https://doi.org/10.1088/1361-6595/aae429)
は、generator、matching network、stray lossを含む外部回路とplasma dynamicsを同時に扱う必要性を示す。
これに従い、plasma branchだけでなくgenerator 50 Ω、matching L/C、feed resistance/inductance、DC choke、
blocking capacitor、electrode/ESC stray capacitance、wall leakageを明示した。

今回の `Rp(t)`, `Lp(t)`, `Cs,u(t)`, `Cs,w(t)` は、外部plasma calculationまたは測定同定で与えられる
有効端子parameterである。後述の平行平板・一様bulk近似による逆算は桁の妥当性確認に限り、密度やsheath厚さの
計測値、あるいはprocess physicsの検証結果とは扱わない。

## 3. 問題設定

### 3.1 入力と観測量

| 区分 | 設定 | この問題での役割 |
|---|---:|---|
| upper RF | 60 MHz, 220 Vpeak | plasma sustain側のVHF excitation |
| lower RF | 2 MHz, 450 Vpeak | wafer ion-energy側を想定したLF bias |
| upper DC | 0 / -800 V, 400 kHz | upper electrodeへの負DC重畳 |
| pulse edges | delay 0.25 µs, rise/fall 0.10 µs | 急変を含む動的素子検証 |
| period | 2.5 µs | 最終1周期を比較・同定に使用 |
| objective | wafer voltage `VW(t)` | inverse lossに使う唯一の波形 |
| held-out | upper reflected voltage `V-(t)` | inverse lossに使わない検証波形 |

上部50 Ω参照面の電流をチャンバーへ流入する向きに取ると、進行・反射波は

```text
V+(t) = (VP(t) + Z0 IP(t)) / 2
V-(t) = (VP(t) - Z0 IP(t)) / 2,   Z0 = 50 Ω
```

である。ngspiceの電圧源電流は正端子へ流入する向きなので、保存された `i(Vhf)` からは
`V- = (VP + 50 i(Vhf))/2` として計算する。

### 3.2 外部回路

| 経路 | 素子 |
|---|---|
| upper RF | 50 Ω source、120 pF series C、75 nH match L、0.6 Ω feed R、25 nH feedthrough L |
| upper electrode stray | 90 pF to chamber ground |
| DC bias tee | 250 Ω supply resistance、200 µH RF choke |
| lower RF | 50 Ω source、4.7 nF blocking C、1.35 µH match L、0.4 Ω feed R、60 nH feedthrough L |
| wafer/ESC stray | 140 pF to chamber ground |
| sidewall loss | bulk-bottomからgroundへ5 kΩ |

これらは単純なplasma直列RLCより現実的な回路経路を持たせるためのbenchmark値であり、特定matcherの測定値では
ない。別のwall capacitanceをbulk nodeへ追加すると、上下sheath・wafer strayとの理想capacitor loopにより
charge stateが冗長になるため採用しなかった。数値安定化だけの微小抵抗を隠して追加する代わりに、実在しうる
electrode/ESC strayとwall leakageだけを明示している。

### 3.3 時間変化するplasma branch

pulse phaseを `τ = t mod 2.5 µs` とし、共通envelope `s(t)` を次で与える。

```text
s = 0                         0 <= τ <= 0.25 µs
s = linear 0 -> 1            0.25 < τ < 0.55 µs
s = 1                         0.55 <= τ <= 2.00 µs
s = linear 1 -> 0            2.00 < τ < 2.40 µs
s = 0                         2.40 <= τ <= 2.50 µs
```

`x(t) = xoff + (xon - xoff)s(t)` とし、値を次のように定める。

| 素子 | off | on | 電気的意味 |
|---|---:|---:|---|
| `Rp(t)` | 35 Ω | 15 Ω | density上昇に伴う有効bulk resistance低下 |
| `Lp(t)` | 35 nH | 16 nH | electron inertia相当inductance低下 |
| `Cs,u(t)` | 260 pF | 520 pF | upper sheath薄化相当 |
| `Cs,w(t)` | 360 pF | 720 pF | wafer sheath薄化相当 |
| upper sheath loss | 1.5 kΩ | fixed | 有限conductive loss |
| wafer sheath loss | 2.0 kΩ | fixed | 有限conductive loss |

instantaneous valueの置換ではなく、capacitor chargeとinductor fluxをstateとして定義する。

```text
qs,u = Cs,u(t) vs,u          is,u = dqs,u / dt
qs,w = Cs,w(t) vs,w          is,w = dqs,w / dt
φp   = Lp(t) ip              vp,L = dφp / dt
vp   = vp,L + Rp(t) ip
```

したがって `dC/dt` および `dL/dt` に対応する項を落とさない。[ngspice User Manual](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf)
にあるcharge-formulated behavioral capacitor `Q='expression'` とtime-dependent behavioral inductor
`L='expression'` をcase-local raw elementとして使う。汎用PCDのplasma modelへ、物理則を伴わない動的C/L optionを
追加してはいない。

### 3.4 数値スケールの妥当性と限界

300 mm円板の全面積と18 mmの一様bulkを仮定した桁確認では、設定値は次の範囲に対応する。

| 確認量 | 単純換算 | 解釈上の注意 |
|---|---:|---|
| sheath厚さ `ds = ε0 A / Cs` | 0.87–2.41 mm | 平行平板近似。edge、非一様性、誘電体を無視 |
| electron density `ne = me l / (e² A Lp)` | 2.58e14–5.65e14 m^-3 | 一様bulk・全面積導通の近似 |
| collision scale `νm ≈ Rp/Lp` | 9.38e8–1.00e9 s^-1 | 有効R/Lの比であり、衝突周波数の測定値ではない |

したがって、300 mm級CCPの回路端子モデルとして桁外れではないが、装置校正済みのparameterではない。
60/2 MHz構成、上部負DC、CCPのsheath-C/bulk-RL表現は文献に基づき、400 kHz pulse、理想電圧源振幅、
matcher/feed/stray値、時間profileは既知真値benchmark用の明示的な仮定である。この区別はPDF 6ページにも記載した。

## 4. 順問題の独立検証

reference targetはngspice出力から作らず、同じ回路のmodified nodal equationを独立に実装した。

- state: capacitor charge 6個、inductor flux 4個
- integrator: fixed-step RK4
- internal step: 0.125 ns
- target sampling: 1 ns
- ngspice maximum step: 0.25 ns
- total simulation: 7.5 µs、比較window: 5.0–7.5 µs

比較対象を端子電圧だけに限定せず、state lawが正しいかを確認するため、bulk current、upper-sheath charge、
plasma-inductor fluxも直接比較した。

| 波形 | RMSE | target RMS正規化RMSE | peak-to-peak正規化RMSE |
|---|---:|---:|---:|
| wafer voltage | 0.0657924 V | 1.87231e-4 | 6.25756e-5 |
| upper reflected voltage | 0.154504 V | 2.45190e-3 | 6.55509e-4 |
| bulk current | 0.00251420 A | 1.60585e-3 | 4.06541e-4 |
| upper-sheath charge | 8.85867e-12 C | 5.38622e-5 | 2.09899e-5 |
| plasma flux | 5.52968e-11 Wb | 1.66730e-3 | 3.63615e-4 |

target RMS正規化は `RMSE / RMS(target)` で、PCDのwaveform objectiveと同じ定義である。RF carrierを含むため、
transition近傍の最大瞬時誤差だけでなく全周期RMSEを主判定に使う。

## 5. 動的インピーダンスの解釈

時間ごとのparameterを凍結したmain vertical branchの小信号表示を

```text
Zs,u(ω,t) = 1 / (1/Rs,u + jωCs,u(t))
Zs,w(ω,t) = 1 / (1/Rs,w + jωCs,w(t))
Zmain(ω,t) = Zs,u + Rp(t) + jωLp(t) + Zs,w
```

として2 MHzと60 MHzで図示した。さらに `Z0=50 Ω` に対する
`Γ=(Zmain-Z0)/(Zmain+Z0)` をSmith chartへ写像し、距離 `|Γ|` と電力反射比 `|Γ|²` を読めるようにした。

| 周波数 | pulse-off `Zmain` | pulse-on `Zmain` | off/on `|Γ|²` | 読み方 |
|---|---:|---:|---:|---|
| 2 MHz | 119.1-j511.8 Ω | 36.5-j261.4 Ω | 0.918 / 0.904 | 強い容量性で外周に近く、matcherなしでは大反射 |
| 60 MHz | 35.1-j4.4 Ω | 15.0-j2.8 Ω | 0.033 / 0.291 | 虚部は小さいが、on時は実部が50 Ωから離れるため反射増加 |

Smith chart中心は `50+j0 Ω`、外周は全反射、下半面は容量性を表す。灰丸はpulse-off、橙四角はpulse-onである。
ここで示すのはmatcherを除外したplasma main branchのfrozen-state loadであり、source planeで測定したS11ではない。
pulse transitionを含む実波形計算自体は前節のcharge/flux state equationで行う。

## 6. 逆問題

未知量はpulse-on時の4値だけとし、off値と共通envelopeは既知とする。

| 未知量 | 候補 |
|---|---|
| `Rp,on` | 12, 15, 18 Ω |
| `Lp,on` | 13, 16, 19 nH |
| `Cs,u,on` | 440, 520, 600 pF |
| `Cs,w,on` | 620, 720, 820 pF |

全 `3^4 = 81` 候補をngspiceで完全列挙した。lossはwafer電圧のnormalized RMSEだけであり、上部反射波、bulk電流、
真値parameterはrankingに使わない。

結果は次の通りである。

- 81/81 transientが要求終了時刻まで完走、solver failure 0。
- selected candidateは `Rp=15 Ω`, `Lp=16 nH`, `Cs,u=520 pF`, `Cs,w=720 pF` で真値と一致。
- objective normalized RMSEは `1.87231e-4`。
- second-best lossは `1.36547e-3` で、bestの7.29倍に分離。
- 未使用のupper reflected voltageもRMSE `0.154504 V`、normalized RMSE `2.45190e-3`。
- 4本のparameter profile RMSEはすべて0。

これは「wafer波形から任意の4関数を一意に推定できる」という主張ではない。既知の共通envelopeと狭い3水準候補を
持つsystem-identification benchmarkである。実機適用では複数recipe、同期V/I、参照面de-embedding、noise、parameter
相関を含むidentifiability確認が必要になる。

### 6.1 OptunaHub AutoSamplerと標準TPEを200 trialで評価

完全格子で真値を回収できることと、連続空間を効率よく探索できることは別の課題である。そのため、共通envelopeと
off値は固定したまま、`Rp,on`、`Lp,on`、`Cs,u,on`、`Cs,w,on`を次の連続範囲で探索した。既知真値は初期候補へ注入していない。

| 変数 | 連続探索範囲 | 既知真値 |
|---|---:|---:|
| `Rp,on` | 12–18 Ω | 15 Ω |
| `Lp,on` | 13–19 nH | 16 nH |
| `Cs,u,on` | 440–600 pF | 520 pF |
| `Cs,w,on` | 620–820 pF | 720 pF |

この問題は4次元、全変数が連続、単一目的、1評価がngspice transientを伴うblack-boxである。この条件と200 trialの
予算に対して、OptunaHubの `samplers/auto_sampler` を採用した。固定したregistry ref
`61da9ce3093a6c92da998a127b55720f9d3b8fd7` のAutoSamplerは、数値だけの単一目的空間かつ完了trial数250未満では
`GPSampler` を選ぶ。比較対象は通常の `TPESampler` とし、他手法の横並び比較は行わない。

両samplerを同じ探索範囲、同じseed 0～2、各200 trialで実行した。lossは完全格子と同じwafer電圧のnormalized RMSEだけで、
上部反射波は選択に使わないhold-outとした。1,200 trialはすべて評価recordを生成し、solver failureは0である。ただし
AutoSampler/GPは既評価点を再提案し、50 trialはPCDのraw-result cacheを利用した。したがって新規ngspice solveは
AutoSampler 550回、TPE 600回、合計1,150回である。

| sampler | 最終loss最小値 | 3 seed中央値 | 最大値 |
|---|---:|---:|---:|
| OptunaHub AutoSampler（実効GP） | 1.07206e-4 | 1.10212e-4 | 1.74637e-4 |
| 標準TPESampler | 1.21653e-4 | 1.74070e-4 | 2.06896e-4 |

AutoSamplerのwafer目的中央値はTPEの0.633倍、すなわち36.7%低い。一方、未使用の上部反射波nRMSE中央値は
AutoSampler `3.26337e-3`、TPE `2.56432e-3`で、TPEが21.4%低い。wafer lossだけでsamplerを選ぶ場合と、
別出力への整合や物理parameter回収を重視する場合で評価が分かれる。

#### 評価

| 評価観点 | 判定 | 根拠 |
|---|---|---|
| 計算完全性 | 成立 | 1,200/1,200 trial成功、solver failure 0、新規ngspice solve 1,150回 |
| AutoSampler選定 | 妥当 | 4次元数値・単一目的・200 trialなので、固定版AutoSamplerは初回以外をGPで提案 |
| wafer目的の探索 | AutoSamplerが良好 | 最終nRMSE中央値はAuto/GP `1.102e-4`、TPE `1.741e-4` |
| 提案の重複 | 要注意 | Auto/GPは50/600 trialがcache hit。うち41件がseed 0に集中、TPEは0件 |
| 未使用波形への整合 | TPEが良好 | 上部反射波nRMSE中央値はAuto/GP `3.263e-3`、TPE `2.564e-3` |
| 素子値の一意回収 | 未成立 | 低wafer lossでもsampler間でparameter回収とhold-outの順位が一致しない |

既知真値に対する絶対相対誤差の3 seed中央値は、Auto/GPが `Rp 0.878%`、`Lp 1.235%`、`Cs,u 0.182%`、
`Cs,w 0.114%`、TPEが `Rp 0.668%`、`Lp 0.159%`、`Cs,u 0.125%`、`Cs,w 0.276%` である。
Auto/GPはwafer目的をより小さくしたが、4parameterのうち3つと未使用反射波ではTPEの方が真値へ近い。これは
AutoSamplerが不適切という意味ではなく、現在の単一波形lossに沿ってより強く局所化した結果が、物理parameterの最良回収と
一致しないことを示す。

#### 考察

AutoSampler/GPはこの低次元・少数trialの数値探索に適しており、wafer波形の最小化だけが目的なら採用候補である。
ただしstock AutoSamplerは本問題が決定論的であることを明示的に受け取らず、同一点再提案が発生した。cacheにより不要な
ngspice再実行は防げたが、trialを消費するため運用上は監視すべきである。200 trialを超えるとAutoSamplerの内部選択規則も
変わり得るため、本結果を別予算へ外挿しない。

3 seedの記述比較なので、AutoSamplerまたはTPEの一般的優位性や大域最適性は主張しない。本ケースの結論は、
AutoSampler/GPがwafer目的には有効だが、素子同定全体ではTPEを無条件に置き換える根拠にならない、である。次に必要なのは
samplerを増やすことではなく、同期V/I、上部反射波、複数pulse recipeなど独立な観測を目的関数またはhold-outへ追加して
parameter相関を減らすことである。

## 7. 図と再現可能な証拠

1. [装置と上下対応した詳細等価回路](../bench/figures/etch_ccp_dcs/01-apparatus-and-equivalent-circuit.svg)
2. [入力波形と4つの時間変化素子](../bench/figures/etch_ccp_dcs/02-inputs-and-dynamic-elements.svg)
3. [独立MNAとngspiceの順問題比較](../bench/figures/etch_ccp_dcs/03-forward-conformance.svg)
4. [2 MHz / 60 MHz動的branch impedanceとSmith chart](../bench/figures/etch_ccp_dcs/04-dynamic-branch-impedance.svg)
5. [wafer電圧同定、loss履歴、hold-out反射波](../bench/figures/etch_ccp_dcs/05-inverse-identification.svg)
6. [文献根拠、benchmark固有仮定、物理スケール確認](../bench/figures/etch_ccp_dcs/06-literature-basis-and-scope.svg)
7. [OptunaHub AutoSampler（GP）と標準TPEの200 trial比較](../bench/figures/etch_ccp_dcs/07-optunahub-auto-vs-tpe.svg)

結合PDFは `output/pdf/etch-ccp-dcs-dynamic-impedance-study.pdf`、全source path、SHA-256、solver version、誤差、
全81格子候補、Optuna 1,200 trialのloss履歴と集約評価は `bench/figures/etch_ccp_dcs/figure_data.json` に保存している。
図生成器は保存済みartifactを読むだけで、simulationやsearchを再実行しない。

## 8. 再現方法

```powershell
uv run --frozen python bench/figures/generate_etch_ccp_dcs_target.py
uv run --frozen python -m pcd solver-diagnose --json

uv run --frozen python -m pcd validate-case `
  bench/figures/cases/etch_ccp_dcs_forward.yaml --strict --json

uv run --frozen python -m pcd sim-run `
  bench/figures/cases/etch_ccp_dcs_forward.yaml `
  --run-root runs/etch_ccp_dcs_forward_v7_20261001 --json

uv run --frozen python -m pcd run `
  bench/figures/cases/etch_ccp_dcs_inverse.yaml `
  --output runs/etch_ccp_dcs_inverse_v3_20261001 --json

uv run --frozen python -m pcd validate-case `
  bench/figures/cases/etch_ccp_dcs_continuous_inverse.yaml --strict --json

$continuousCase = "bench/figures/cases/etch_ccp_dcs_continuous_inverse.yaml"
foreach ($seed in 0..2) {
  uv run --frozen --group optuna-benchmark python -m pcd run $continuousCase `
    --output "runs/etch_ccp_dcs_optunahub_auto_seed${seed}_20261001" `
    --optimizer optuna_auto --trials 200 --seed $seed --json
  uv run --frozen --group optuna-benchmark python -m pcd run $continuousCase `
    --output "runs/etch_ccp_dcs_tpe_seed${seed}_20261001" `
    --optimizer optuna_tpe --trials 200 --seed $seed --json
}

uv run --frozen --group optuna-benchmark python -m bench.figures.generate_etch_ccp_dcs_evidence
```

solver adapterは、return codeとCSV存在だけでなく、transient最終時刻が要求`stop_s`へ到達したことを確認する。
ngspiceがtime-step failure後に部分CSVを残して終了コード0を返す場合も、`incomplete_transient`として失敗させる。

## 9. 次に実機へ進める場合の最小手順

1. `P_HF`、upper electrode、wafer nodeの実機reference planeを固定する。
2. open/short/dummy loadまたは既知fixtureでcable、feedthrough、matcherのR/L/Cを同定する。
3. 同期したwafer voltageと上部V/Iまたはdirectional-coupler波形を取得する。
4. 最初は今回と同じ4つのon-state parameterだけを同定し、別pulse条件と反射波をhold-outにする。
5. 残差が電圧依存・harmonic依存で構造的に残る場合だけ、非線形sheath lawまたは外部plasma solver連成を
   別責務として追加する。

この順序なら、測定で識別できないparameterを安易に増やさず、回路基盤の責務を保ったままmodel fidelityを上げられる。
