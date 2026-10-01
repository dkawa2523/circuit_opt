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
有効端子parameterである。値から密度・sheath厚さを逆算してprocess physicsを主張しない。

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
| `Lp(t)` | 350 nH | 160 nH | electron inertia相当inductance低下 |
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
| wafer voltage | 0.287151 V | 8.13188e-4 | 2.64821e-4 |
| upper reflected voltage | 0.364325 V | 8.78371e-3 | 1.65293e-3 |
| bulk current | 0.0140259 A | 5.90386e-3 | 1.46195e-3 |
| upper-sheath charge | 4.19572e-11 C | 2.54953e-4 | 9.82362e-5 |
| plasma flux | 2.75524e-9 Wb | 6.38556e-3 | 1.74149e-3 |

target RMS正規化は `RMSE / RMS(target)` で、PCDのwaveform objectiveと同じ定義である。RF carrierを含むため、
transition近傍の最大瞬時誤差だけでなく全周期RMSEを主判定に使う。

## 5. 動的インピーダンスの解釈

時間ごとのparameterを凍結したmain vertical branchの小信号表示を

```text
Zs,u(ω,t) = 1 / (1/Rs,u + jωCs,u(t))
Zs,w(ω,t) = 1 / (1/Rs,w + jωCs,w(t))
Zmain(ω,t) = Zs,u + Rp(t) + jωLp(t) + Zs,w
```

として2 MHzと60 MHzで図示した。2 MHzではsheathのcapacitive reactanceが支配的で、60 MHzでは
`ωLp` の寄与が大きくなり、このbenchmark値ではmain branchの虚部が正になる。この図は周波数ごとのRF loadingを
理解するためのfrozen-time表示であり、pulse transitionを含む実波形計算そのものは前節のcharge/flux state equationで
行う。

## 6. 逆問題

未知量はpulse-on時の4値だけとし、off値と共通envelopeは既知とする。

| 未知量 | 候補 |
|---|---|
| `Rp,on` | 12, 15, 18 Ω |
| `Lp,on` | 130, 160, 190 nH |
| `Cs,u,on` | 440, 520, 600 pF |
| `Cs,w,on` | 620, 720, 820 pF |

全 `3^4 = 81` 候補をngspiceで完全列挙した。lossはwafer電圧のnormalized RMSEだけであり、上部反射波、bulk電流、
真値parameterはrankingに使わない。

結果は次の通りである。

- 81/81 transientが要求終了時刻まで完走、solver failure 0。
- selected candidateは `Rp=15 Ω`, `Lp=160 nH`, `Cs,u=520 pF`, `Cs,w=720 pF` で真値と一致。
- objective normalized RMSEは `8.13188e-4`。
- second-best lossは `2.14343e-3` で、selectedとの分離を確認。
- 未使用のupper reflected voltageもRMSE `0.364325 V`、normalized RMSE `8.78371e-3`。
- 4本のparameter profile RMSEはすべて0。

これは「wafer波形から任意の4関数を一意に推定できる」という主張ではない。既知の共通envelopeと狭い3水準候補を
持つsystem-identification benchmarkである。実機適用では複数recipe、同期V/I、参照面de-embedding、noise、parameter
相関を含むidentifiability確認が必要になる。

## 7. 図と再現可能な証拠

1. [装置と上下対応した詳細等価回路](../bench/figures/etch_ccp_dcs/01-apparatus-and-equivalent-circuit.svg)
2. [入力波形と4つの時間変化素子](../bench/figures/etch_ccp_dcs/02-inputs-and-dynamic-elements.svg)
3. [独立MNAとngspiceの順問題比較](../bench/figures/etch_ccp_dcs/03-forward-conformance.svg)
4. [2 MHz / 60 MHz動的branch impedance](../bench/figures/etch_ccp_dcs/04-dynamic-branch-impedance.svg)
5. [wafer電圧同定、loss履歴、hold-out反射波](../bench/figures/etch_ccp_dcs/05-inverse-identification.svg)

結合PDFは `output/pdf/etch-ccp-dcs-dynamic-impedance-study.pdf`、全source path、SHA-256、solver version、誤差、
全81候補は `bench/figures/etch_ccp_dcs/figure_data.json` に保存している。図生成器は保存済みartifactを読むだけで、
simulationやsearchを再実行しない。

## 8. 再現方法

```powershell
uv run --frozen python bench/figures/generate_etch_ccp_dcs_target.py
uv run --frozen python -m pcd solver-diagnose --json

uv run --frozen python -m pcd validate-case `
  bench/figures/cases/etch_ccp_dcs_forward.yaml --strict --json

uv run --frozen python -m pcd sim-run `
  bench/figures/cases/etch_ccp_dcs_forward.yaml `
  --run-root runs/etch_ccp_dcs_forward_v6_20261001 --json

uv run --frozen python -m pcd run `
  bench/figures/cases/etch_ccp_dcs_inverse.yaml `
  --output runs/etch_ccp_dcs_inverse_v2_20261001 --json

uv run --frozen python bench/figures/generate_etch_ccp_dcs_evidence.py
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
