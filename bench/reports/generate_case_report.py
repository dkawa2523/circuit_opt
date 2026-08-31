"""Build the third-party-readable benchmark case report artifact.

The report intentionally treats scenario rows and candidate rows as evidence
inside one logical benchmark.  This prevents derived corners or hardware
alternatives from being misrepresented as independent physical observations.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE = ROOT / "runs" / "benchmark_suite" / "benchmark_result.json"
DEFAULT_LITERATURE = ROOT / "runs" / "literature" / "final_evaluation" / "evaluation.json"
DEFAULT_OUTPUT = ROOT / "output" / "reports" / "benchmark-case-report-artifact.json"
TITLE = "回路解析基盤 ベンチマークケース技術レポート"


CORE_REFERENCE_PLANES = {
    "A1_topology_l_match": "electrode_terminal",
    "A2_topology_pi_match": "electrode_terminal",
    "A3_topology_pi_match_harmonic": "electrode_terminal",
    "A4_ccp_lumped_frequency_conformance": "electrode_terminal",
    "A5_icp_transformer_frequency_conformance": "coil_feedthrough",
    "B1_fixed_nominal": "electrode_terminal",
    "B2_limited_tuner": "electrode_terminal",
    "B3_full_tuner": "electrode_terminal",
    "B4_independent_frequency_points": "electrode_terminal",
    "B5_high_drive_stress": "electrode_terminal",
    "B6_discrete_hardware_search": "electrode_terminal",
    "B7_role_factorial_search": "electrode_terminal",
    "B8_component_value_corner_stress": "electrode_terminal",
    "D1_reference_plane_explicit": "plasma_terminal（fixtureを回路に明示）",
    "D2_reference_plane_embedded": "electrode_terminal（fixture埋込み済み）",
    "D3_reference_plane_double_counted": "electrode_terminal（fixture埋込み済み入力へ再追加）",
}

CORE_EVIDENCE_GROUPS = {
    "A1_topology_l_match": "解析式に対する配線適合",
    "A2_topology_pi_match": "解析式に対する配線適合",
    "A3_topology_pi_match_harmonic": "解析式に対する配線適合",
    "A4_ccp_lumped_frequency_conformance": "合成端子式の実装適合",
    "A5_icp_transformer_frequency_conformance": "合成端子式の実装適合",
    "B1_fixed_nominal": "合成・決定論的設計負例",
    "B2_limited_tuner": "合成・決定論的設計負例",
    "B3_full_tuner": "合成・決定論的設計正例",
    "B4_independent_frequency_points": "合成・決定論的設計負例",
    "B5_high_drive_stress": "合成・決定論的電気スクリーニング",
    "B6_discrete_hardware_search": "合成・決定論的探索正例",
    "B7_role_factorial_search": "合成・決定論的役割分離",
    "B8_component_value_corner_stress": "合成・決定論的全因子角点",
    "D1_reference_plane_explicit": "解析式に対する参照面適合",
    "D2_reference_plane_embedded": "解析式に対する参照面等価性",
    "D3_reference_plane_double_counted": "合成・決定論的誤用負例",
}

CORE_OUTCOME_SCOPES = {
    **{case_id: "実装・式・境界のconformance fixture" for case_id in CORE_EVIDENCE_GROUPS if case_id[0] in {"A", "D"}},
    **{case_id: "設計workflow fixture" for case_id in CORE_EVIDENCE_GROUPS if case_id.startswith("B")},
}

CONSTRAINT_LABELS = {
    "max_reflection_magnitude": "反射係数|Γ|上限",
    "min_control_margin": "正規化Control余裕下限",
    "max_component_L1_current_rms_A": "L1実効電流上限",
    "max_component_L1_loss_W": "L1固定Rseries平均実電力上限",
    "max_source_current_rms_A": "理想源端子実効電流上限",
    "max_source_apparent_power_VA": "理想源端子見掛け電力上限",
}


CORE_META: dict[str, dict[str, str]] = {
    "A1_topology_l_match": {
        "title": "L-match接続と複素インピーダンス",
        "purpose": "engine conformance",
        "input": "13.56 MHz、負荷 30-j20 Ω。直列L1=0.5 µH、出力側シャントC1=100 pF。",
        "candidate": "固定L-match 1組。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "独立した閉形式回路計算の入力Zとngspice結果を0.02 Ω以内で比較し、50 Ω基準の反射判定も確認する。",
        "interpretation": "期待どおり不整合になることまで含め、L-match配線、負荷接続、AC抽出、反射計算が一貫している。",
    },
    "A2_topology_pi_match": {
        "title": "π-match接続と複素インピーダンス",
        "purpose": "engine conformance",
        "input": "13.56 MHz、負荷 30-j20 Ω。C1=100 pF、L1=0.5 µH、C2=50 pF。",
        "candidate": "固定π-match 1組。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "独立閉形式計算の入力Zとngspice結果を0.02 Ω以内で比較し、反射電力10%以下の分類を確認する。",
        "interpretation": "π-matchの接続と数値処理が解析解に一致する。A1/A3とのfeasible差は任意部品値の結果であり、トポロジ順位ではない。",
    },
    "A3_topology_pi_match_harmonic": {
        "title": "直列LC分岐付きπ-matchの基本波接続",
        "purpose": "engine conformance",
        "input": "13.56 MHz、負荷30-j20 Ω。π-matchのC1=100 pF、L1=0.5 µH、C2=50 pFに、出力から接地へLh=1 µH、Ch=50 pFの直列分岐を追加。",
        "candidate": "固定高調波分岐付きπ-match 1組。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "分岐を含む独立複素回路式と基本波入力Zを0.02 Ω以内で比較する。",
        "interpretation": "分岐が一度だけ正しく接続されたことを示す。非線形負荷に対する高調波抑制性能は評価しない。",
    },
    "A4_ccp_lumped_frequency_conformance": {
        "title": "CCP有効R-L-C負荷の周波数継続",
        "purpose": "model conformance",
        "input": "R_eff=25 Ω、L_eff=0.2 µH、C_sheath_eq=120 pFの直列有効一ポート。10、13.56、20 MHz。",
        "candidate": "L1=0.2 µH、C1=20 pFの固定L-match。",
        "scenario": "周波数3点。各点で同じR-L-C式を再評価。",
        "control": "なし。",
        "method": "Z=R+jωL+1/(jωC)と独立L-match式から得る入力Zを各周波数で0.02 Ω以内に再現するか確認する。",
        "interpretation": "公開入力からSPICE・指標抽出までモデル式の意味が保存される。CCP物理やパラメータ同定の妥当性は示さない。",
    },
    "A5_icp_transformer_frequency_conformance": {
        "title": "ICP有効変圧器負荷の周波数継続",
        "purpose": "model conformance",
        "input": "Rcoil=0.4 Ω、Lcoil=2 µH、反射インダクタンス0.245 µH、二次減衰率6.6667 Mrad/s、Cparallel=20 pF。10、13.56、20 MHz。",
        "candidate": "13.56 MHzで解析的に整合した固定L-match（L1=8.71259 µH、C1=74.1545 pF）。",
        "scenario": "周波数3点。",
        "control": "なし。",
        "method": "反射負荷項、コイル項、並列容量を含む独立端子式と、各周波数の入力Z・分類を比較する。",
        "interpretation": "13.56 MHzだけ整合する結果を含め、端子モデルの実装を確認する。密度・衝突・電力分配や実ICP装置は検証しない。",
    },
    "B1_fixed_nominal": {
        "title": "公称点だけで整合した固定回路の負荷窓",
        "purpose": "negative control",
        "input": "13.56 MHz、electrode-terminalの合成5負荷: 50-j161.538、25-j80.769、12.5-j40.385、12.5-j80.769、50-j80.769 Ω。",
        "candidate": "公称点用の固定π-match 1組（C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pF）。",
        "scenario": "抵抗と容量性リアクタンスを変えた5負荷点。",
        "control": "なし。",
        "method": "各負荷で|Γ|を計算し、全Scenarioが反射電力10%以下かをworst-caseで判定する。",
        "interpretation": "公称点の良好な整合は負荷窓全体のロバスト性を意味しない、という期待負例である。",
    },
    "B2_limited_tuner": {
        "title": "制限チューナの到達性と20%制御余裕",
        "purpose": "negative control",
        "input": "13.56 MHz、electrode-terminalの5負荷: 50-j161.538、25-j80.769、12.5-j40.385、12.5-j80.769、50-j80.769 Ω。固定L1=0.643146 µH。",
        "candidate": "固定インダクタ1組。",
        "scenario": "上記の合成5負荷点。",
        "control": "C1={480,550,670,860,890} pF、C2={45,135,160,200,210} pFの25状態をScenarioごとに選択。",
        "method": "全125組を列挙し、|Γ|基準に加えて選択状態の正規化制御余裕0.20以上を要求する。",
        "interpretation": "全点で電気的整合へ到達しても、端に近い3設定を制御余裕不足として棄却できることを示す。",
    },
    "B3_full_tuner": {
        "title": "拡張チューナによる負荷窓カバー",
        "purpose": "positive control",
        "input": "13.56 MHz、electrode-terminalの5負荷: 50-j161.538、25-j80.769、12.5-j40.385、12.5-j80.769、50-j80.769 Ω。固定L1=0.643146 µH。",
        "candidate": "固定インダクタ1組。",
        "scenario": "5負荷点。",
        "control": "C1={400,480,550,670,860,890,970} pF、C2={20,45,135,160,200,210,235} pFの49状態をScenarioごとに選択。",
        "method": "全245組を列挙し、反射電力10%以下かつ正規化制御余裕0.20以上を全負荷で要求する。",
        "interpretation": "固定ハードウェアと可変設定を分離したまま、宣言した制御権限で全窓を覆える正例である。",
    },
    "B4_independent_frequency_points": {
        "title": "独立周波数負荷点に対する固定回路",
        "purpose": "negative control",
        "input": "10 MHz:20-j110 Ω、13.56 MHz:25-j80.769 Ω、18 MHz:30-j55 Ω。",
        "candidate": "C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pFの固定π-match 1組。",
        "scenario": "周波数と複素負荷を一組にした独立3点。",
        "control": "なし。",
        "method": "点間を補間せず、各周波数点を独立Scenarioとして反射電力10%以下か判定する。",
        "interpretation": "公称周波数合格を連続帯域の証拠に拡張できないことを示す。",
    },
    "B5_high_drive_stress": {
        "title": "2振幅における基本波電気ストレススクリーニング",
        "purpose": "negative control",
        "input": "13.56 MHz、electrode-terminal負荷25-j80.769 Ω。理想電圧源Vsrc端子の基本波を25/100 Vpeakとする。π回路C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pFに、合成した固定等価直列抵抗を各々0.1、0.5、0.1 Ω置く。",
        "candidate": "損失を含む固定π-match 1組。",
        "scenario": "理想源端子の25 Vpeakと100 Vpeakという2つの決定論的条件。",
        "control": "なし。",
        "method": "(1) 整合、(2) YAMLで宣言した合成電気スクリーニング閾値、(3) 明示抵抗損失と源端―負荷面電力差の数値閉包を分けて判定する。閾値はresult artifactから読み、本文へハードコードしない。",
        "interpretation": "整合は両振幅で維持される一方、100 Vpeakでは宣言したL1電流・L1固定Rseries損失・理想源端子電流・理想源端子VAの合成閾値を超える。これは線形定常ACの電気スクリーニングであり、実部品定格、温度、寿命の資格結果ではない。",
    },
    "B6_discrete_hardware_search": {
        "title": "離散部品候補の完全列挙",
        "purpose": "positive control",
        "input": "13.56 MHz、負荷25-j80.769 Ω。固定L1=1.19731 µH、C2=7.45488 pF。",
        "candidate": "C1={100、258.222、800} pFの3候補。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "3候補を重複・欠落なく全列挙し、feasibility-firstで反射基準を満たす候補を選ぶ。",
        "interpretation": "有限BOM候補探索と唯一の既知解の回収を確認する。連続最適化や負荷窓ロバスト性は示さない。",
    },
    "B7_role_factorial_search": {
        "title": "Candidate×Scenario×Controlの直交性",
        "purpose": "positive control",
        "input": "13.56 MHz、electrode-terminalの解析的2負荷: 67.753-j92.917 Ωと171.269-j63.994 Ω。L-match。",
        "candidate": "L1={1、2} µH。",
        "scenario": "2負荷点。",
        "control": "C1={20、80} pF。",
        "method": "2×2×2=8組を完全列挙し、各ScenarioでControlを選び、両Scenarioを覆うCandidateを選ぶ。",
        "interpretation": "固定候補、外部条件、条件別操作量がデータ処理中に混線しないことを直接確認する。",
    },
    "B8_component_value_corner_stress": {
        "title": "選定BOMの実現部品値8隅",
        "purpose": "negative control",
        "input": "13.56 MHz、25-j80.769 Ω。公称C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pFを各々0.85または1.15倍。",
        "candidate": "選定済み公称BOM 1組。",
        "scenario": "3部品の±15%全因子2³=8頂点。",
        "control": "なし。",
        "method": "実現値係数をScenarioとして8頂点すべてで反射基準を評価する。",
        "interpretation": "3/8という結果は等重み頂点被覆であり、歩留まり、確率分布、連続箱内部の保証ではない。",
    },
    "D1_reference_plane_explicit": {
        "title": "fixtureを明示した参照面表現",
        "purpose": "reference-plane conformance",
        "input": "13.56 MHz。C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pFのπ-match後段にRfixture=2 Ω、Lfixture=225.715 nHを明示し、plasma-terminal負荷25-j100 Ωへ接続。",
        "candidate": "fixtureを明示した固定回路1組。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "fixtureと負荷を一度ずつ含む回路の源側入力Zを独立複素回路式と0.02 Ω以内で比較し、反射基準の分類を確認する。",
        "interpretation": "一つの明確な物理表現を作る基準ケース。実fixtureの同定精度は評価しない。",
    },
    "D2_reference_plane_embedded": {
        "title": "fixtureを一回だけ埋め込んだ表現",
        "purpose": "reference-plane conformance",
        "input": "13.56 MHz。C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pFのπ-matchへ、R=2 Ω・L=225.715 nHのfixtureを一度だけ畳み込んだelectrode-terminal負荷27-j80.769 Ωを接続。",
        "candidate": "fixture埋込み済み一ポートを接続した固定回路1組。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "fixture明示表現の独立oracle Zin=49.2913-j3.73865 Ωと0.02 Ω以内で一致し、設計分類も一致するか確認する。",
        "interpretation": "同じfixtureを一度だけ扱えば、明示表現と埋込み表現が等価になることを示す。",
    },
    "D3_reference_plane_double_counted": {
        "title": "fixture二重計上の負例",
        "purpose": "negative control / reference plane",
        "input": "13.56 MHz。C1=258.222 pF、L1=1.19731 µH、C2=7.45488 pFのπ-match後段で、fixture込み27-j80.769 Ω負荷にR=2 Ω・L=225.715 nHを再度追加。",
        "candidate": "fixtureを二重に含む固定回路1組。",
        "scenario": "nominal 1点。",
        "control": "なし。",
        "method": "fixture一回表現のbaseline Zin=49.2913-j3.73865 Ωから10 Ω以上移動し、反射基準の分類が反転することを確認する。",
        "interpretation": "参照面要素の二重計上が電気的に観測可能であることを示す。未申告fixtureの自動検出はしない。",
    },
}


CORE_QUESTIONS = {
    "A1_topology_l_match": "公開L-matchネットリストは、独立閉形式で求めた入力インピーダンスを再現するか。",
    "A2_topology_pi_match": "公開π-matchネットリストは、独立閉形式で求めた入力インピーダンスを再現するか。",
    "A3_topology_pi_match_harmonic": "直列LCシャント分岐を持つ公開π-matchは、独立回路式の基本波入力インピーダンスを再現するか。",
    "A4_ccp_lumped_frequency_conformance": "CCP有効R-L-C一ポートの周波数依存式が、入力定義からScenario展開、SPICE、指標抽出まで保たれるか。",
    "A5_icp_transformer_frequency_conformance": "ICP有効変圧器一ポートの周波数依存式が、入力定義からScenario展開、SPICE、指標抽出まで保たれるか。",
    "B1_fixed_nominal": "公称負荷だけで整合した固定回路は、再調整なしで宣言した負荷窓全体を覆えるか。",
    "B2_limited_tuner": "電気的に到達できる設定でも、正規化20%の調整余裕を必須にすると負荷窓全体を覆えるか。",
    "B3_full_tuner": "一つの固定回路が、Scenarioごとに許可された可変容量を選ぶことで負荷窓全体を覆えるか。",
    "B4_independent_frequency_points": "一周波数で合格した固定回路は、独立に与えた別周波数・別負荷点でも合格するか。",
    "B5_high_drive_stress": "固定線形回路は、理想源端子の振幅を4倍にしたときも、宣言した合成電気スクリーニング閾値を満たすか。",
    "B6_discrete_hardware_search": "有限BOM候補の完全列挙は、重複や欠落なく既知の可行コンデンサを回収できるか。",
    "B7_role_factorial_search": "Candidate、Scenario、Controlを直交して列挙し、全Scenarioを覆う唯一のCandidateを選べるか。",
    "B8_component_value_corner_stress": "選定済み公称回路は、宣言した実現部品値の全8頂点でも反射基準を満たすか。",
    "D1_reference_plane_explicit": "既知の損失性R-L fixtureをプラズマ端子負荷の前へ明示したとき、源側にどの入力Zが現れるか。",
    "D2_reference_plane_embedded": "R=2 Ω・L=225.715 nHのfixtureを上流一ポートへ一度だけ畳み込むと、matcher入力Zと設計分類は保存されるか。",
    "D3_reference_plane_double_counted": "同じfixtureを回路と供給負荷Zの両方に含めると、入力Zと設計分類が期待どおり変化するか。",
}


CORE_LIMITATIONS = {
    "A1_topology_l_match": "任意の合成負荷・部品値であり、L-matchが常に優位であることや実チャンバー適合は示さない。",
    "A2_topology_pi_match": "この合成点での合格は、π-matchの普遍的優位性、負荷窓、部品定格を示さない。",
    "A3_topology_pi_match_harmonic": "一周波数の線形確認であり、非線形プラズマに対する高調波抑制を示さない。",
    "A4_ccp_lumped_frequency_conformance": "合成R-L-C値の物理妥当性、CCPパラメータ同定、実装置の周波数応答を示さない。",
    "A5_icp_transformer_frequency_conformance": "合成端子パラメータから、密度、衝突周波数、電力分配、実ICPチャンバーは検証できない。",
    "B1_fixed_nominal": "公称一点の固定整合が負荷窓へ一般化できないことを示す負例であり、実負荷分布ではない。",
    "B2_limited_tuner": "離散容量状態は制御権限のモデルであり、制御則、追従速度、安定性を示さない。",
    "B3_full_tuner": "宣言した5負荷点と有限容量格子内の成立であり、連続負荷領域や実チューナ定格を保証しない。",
    "B4_independent_frequency_points": "3点は合成された独立入力であり、補間可能な連続帯域や実測周波数応答ではない。",
    "B5_high_drive_stress": "固定線形一周波数モデルであり、実プラズマ負荷やESRの振幅・温度依存、過渡、高調波、温度上昇、冷却、絶縁寿命を示さない。C1/C2は値を観測するが定格制約を持たない。",
    "B6_discrete_hardware_search": "有限3候補の完全列挙だけを示し、連続最適化や負荷窓ロバスト性を示さない。",
    "B7_role_factorial_search": "解析的に構成した2点の役割分離テストであり、実チャンバー窓や制御器モデルではない。",
    "B8_component_value_corner_stress": "合成±15%の8頂点であり、ベンダ公差、確率分布、歩留まり、温度・経時変化ではない。",
    "D1_reference_plane_explicit": "参照面ラベルだけでは、外部から与えたde-embedding変換の正しさを検証できない。",
    "D2_reference_plane_embedded": "合成R-L fixtureの等価性であり、実fixtureや一般S-parameter de-embeddingを資格付けしない。",
    "D3_reference_plane_double_counted": "未申告fixtureが任意の供給Zへ既に含まれるかを自動判定する機能は示さない。",
}


LITERATURE_STATIC = [
    {
        "id": "LEE21-T1",
        "title": "Lee 2021 Table-I bias-path回路閉包",
        "question": "Lee 2021 Table-Iの集中定数回路は、論文が示す補正後bias plasma-terminal Zを再現するか。",
        "purpose": "source fidelity",
        "input": "13.56 MHz、論文Table-IのCsh series [C0 parallel (Rp series Lp)]。corrected bias plasma terminal。",
        "candidate": "なし。論文回路値の再計算であり設計探索ではない。",
        "scenario": "Ar 200 sccm、5 mTorr、source 400 W、bias 100 Wの掲載1条件。",
        "control": "なし。",
        "method": "論文回路値から複素Zを独立計算し、掲載された補正後端子Zとの閉包を確認する。",
        "acceptance": "論文丸め値との差をR≤0.01 Ω、X≤0.05 Ω、独立式とNGSpiceとの差をR/X各≤1e-5 Ωとする。",
        "provenance": "Lee et al. 2021, DOI 10.1063/6.0000883, Table I。定義 `bench/literature/p0_lee2021_bias/source.yaml`、期待値 `expectations.yaml`。",
        "result_id": "lee2021_bias_table_i",
        "establishes": "Table-Iの集中定数回路算術、位相符号、corrected plasma-terminal Zの数値閉包。",
        "does_not_establish": "未掲載の720 mm経路分布、他動作条件、装置一般性。",
    },
    {
        "id": "GEC-S1",
        "title": "Hargis 1994 Tables III/IVの32行変換",
        "question": "Hargis 1994の公開V/I/phase全32行を、基準面を変えずに複素Zと電力へ変換し、一ポートとして再生できるか。",
        "purpose": "source fidelity",
        "input": "13.56 MHz、4圧力×4外部drive×2 empty-cell resonance群の公開V/I/phase 32行。powered-electrode surface。",
        "candidate": "なし。データ変換と一ポート再生。",
        "scenario": "32 published central rows。24/34 MHzは駆動周波数ではなく装置群ラベル。",
        "control": "なし。",
        "method": "|Z|=V/I、R=|Z|cosφ、X=|Z|sinφ、P1=0.5VIcosφを再計算し、派生表・電力閉包・NGSpice一ポートを確認する。",
        "acceptance": "32行キーを完全一致。派生viewは|差|≤max(2e-11×max(|actual|,|expected|),1e-8)、基本波と掲載電力の差≤5%かつ掲載spread内、NGSpiceは|ΔZ|≤1e-6 Ω+1e-8|Z|を要求する。",
        "provenance": "Hargis et al. 1994, DOI 10.1063/1.1144770, Tables III/IV。原表 `bench/literature/p1_gec_ccp/raw_tables_iii_iv.csv`、runner `run_all32_benchmark.py`。",
        "result_id": "hargis1994_tables_iii_iv",
        "establishes": "全32行の転記、V/I/phaseから端子R+jXへの変換、基本波電力閉包、一ポート再生。",
        "does_not_establish": "純粋なプラズマ内部Z、量産装置の統計分布、周波数窓。",
    },
    {
        "id": "LEE20-M1",
        "title": "Lee 2020 ICP変圧器式18/19の適合性",
        "question": "Lee 2020のICP変圧器終端式とpeak-current電力式は、直接式・実装・独立SPICEの三経路で一致するか。",
        "purpose": "model equation conformance",
        "input": "γ/ω={0.01、0.1、1、10、100}と追加周波数を含む6つの合成代数条件。",
        "candidate": "なし。終端式実装の検証。",
        "scenario": "6 equation test vectors。測定点ではない。",
        "control": "なし。",
        "method": "論文式の直接矩形式、pcd.rf_loads実装、PCD builderを使わない独立NGSpice変圧器の3経路を比較する。",
        "acceptance": "式対Pythonはrel/abs各1e-12、式対NGSpice Zはrel≤2e-5かつabs≤1e-7 Ω、電力はrel≤5e-5かつabs≤1e-10 W。受動性と減衰領域coverageも必須。",
        "provenance": "Lee et al. 2020, DOI 10.1063/1.5133862, Eqs. 18/19。test vectors `bench/literature/p1_lee2020_icp/cases.csv`、runner `run_benchmark.py`。",
        "result_id": "lee2020_icp_transformer_equations",
        "establishes": "採用したICP変圧器終端インピーダンス式とpeak-current電力式の数値実装。",
        "does_not_establish": "有限寸法プラズマ計算、密度推定、分布アンテナモード、実装置。",
    },
    {
        "id": "COLPO-F1",
        "title": "Colpo 1999 fixture共振とgraphite dummy",
        "question": "Colpo 1999のfixture topologyと丸め部品値は、掲載共振とgraphite dummyのglobal-terminal Zを再現するか。",
        "purpose": "source fidelity",
        "input": "論文の二区間L//C conductive-dummy回路、3共振ラベル、13.56 MHzのgraphite dummy 1.26+j57 Ω。",
        "candidate": "なし。fixture式と既知負荷の再生。",
        "scenario": "共振3点、off-resonance点、graphite dummy 1点。",
        "control": "なし。",
        "method": "閉形式、汎用PCD回路、独立NGSpiceを比較し、丸め部品値の共振と既知global-terminal Zを確認する。",
        "acceptance": "3共振の論文丸め差を順に≤3%、5%、3%、SPICE対閉形式とoff-resonanceを≤0.5%、graphite Zの相対誤差を≤2e-5、受動性を要求する。",
        "provenance": "Colpo et al. 1999, DOI 10.1063/1.369268。定義 `bench/literature/p1_colpo1999_icp/source.yaml`、runner `run.py`。",
        "result_id": "colpo1999_fixture_and_graphite",
        "establishes": "二区間fixture topologyの共振構造と、既知graphite dummyのglobal-terminal Z再生。",
        "does_not_establish": "plasma-on状態、量産planar ICP、寄生成分の装置一般性。",
    },
    {
        "id": "COLPO-S1",
        "title": "Colpo 1999 plasma-on図の15中心点と読取コーナー",
        "question": "Colpo 1999図の対応付き15中心点を再現可能に読取り、出典中心を変えずに60読取コーナーへ展開できるか。",
        "purpose": "digitized source fidelity",
        "input": "13.56 MHz、RFZ60 global ICP source terminal。2読者で対応確認した3圧力×5電力=15組のR/X中心。",
        "candidate": "なし。図読取と派生規則の検証。",
        "scenario": "15 parent conditions。各中心からR±6 Ω、X±7 Ωの4読取コーナーを派生。",
        "control": "なし。",
        "method": "pixel transform逆変換、marker対応、除外点、受動性、傾向、一ポート再生、15→60行の決定論的展開を確認する。",
        "acceptance": "pixel round-trip≤0.101 Ω、独立読者差R≤3.4 Ω/X≤3.3 Ω、SPICE相対複素誤差≤5e-5、受動性・傾向・15→60行完全一致を要求する。",
        "provenance": "Colpo et al. 1999, DOI 10.1063/1.369268, plasma-on figures。校正 `bench/literature/p1_colpo1999_icp/digitized/axis_calibration.yaml`、中心値 `qualified_impedance.csv`。",
        "result_id": "colpo1999_digitized_centers",
        "establishes": "対応付き15中心点の再現可能な図読取と、中心ごとの4読取コーナーへの決定論的展開。",
        "does_not_establish": "60独立観測、測定不確かさ、信頼区間、他装置の負荷分布。",
    },
]

STATIC_SOURCE_PATHS = {
    "LEE21-T1": "bench/literature/p0_lee2021_bias/source.yaml",
    "GEC-S1": "bench/literature/p1_gec_ccp/raw_tables_iii_iv.csv",
    "LEE20-M1": "bench/literature/p1_lee2020_icp/cases.csv",
    "COLPO-F1": "bench/literature/p1_colpo1999_icp/source.yaml",
    "COLPO-S1": "bench/literature/p1_colpo1999_icp/digitized/qualified_impedance.csv",
}


GEC_66_INPUT = (
    "13.56 MHz、powered-electrode-surface、66 Pa。empty-cell resonanceラベル24/34 MHz（駆動周波数ではない）"
    "×外部drive 75/100/150/200 Vppの8 published loads: "
    "98.461-j271.996、81.470-j271.555、64.226-j265.479、59.874-j270.075、"
    "148.332-j384.411、105.046-j321.390、69.845-j286.514、53.041-j278.049 Ω。"
)
GEC_FULL_CONTROL = "C1={400,450,500,550} pF、C2={65,70,75,80} pFの16状態。"


DESIGN_META: dict[str, dict[str, str]] = {
    "literature_p0_lee2021_matcher_output_probe": {
        "report_id": "LEE21-P1",
        "title": "Lee 2021 matcher-output planeでの整合再現",
        "input": "13.56 MHz。Lee 2021 Table-IのVI-probe/matcher-output Z=2.41-j7.45 Ωを、ideal π fixtureへ接続。",
        "candidate": "C1=1043.829 pF、L1=0.213013 µH、C2=1 pFの固定π fixture 1組。",
        "scenario": "Ar 200 sccm、5 mTorr、source 400 W、bias 100 Wのmatcher-output plane 1点。",
        "control": "なし。",
        "method": "独立π回路演算とngspiceの|Γ|を比較し、platformの10%反射基準を適用する。",
        "interpretation": "正しいfixture入力面での基準正例。論文matcherそのものの再現ではない。",
    },
    "literature_p0_lee2021_post_coax_plane_sensitivity": {
        "report_id": "LEE21-P2",
        "title": "Lee 2021 post-coax plane誤代入感度",
        "input": "13.56 MHz。Lee 2021 Table-Iの下流post-coax Z=2.47-j14.3 Ωを、基準面変換なしでideal π fixtureへ代入。",
        "candidate": "C1=1043.829 pF、L1=0.213013 µH、C2=1 pFの固定π fixture 1組。",
        "scenario": "Ar 200 sccm、5 mTorr、source 400 W、bias 100 Wのpost-coax plane 1点を誤接続する反実仮想。",
        "control": "なし。",
        "method": "基準面以外を同じに固定し、|Γ|と設計分類の変化を観測する。",
        "interpretation": "誤った平面のZを同一fixtureへ接続すると判断が変わる感度を示す。PCDが誤平面を自動検出する証拠ではない。",
    },
    "literature_p0_lee2021_plasma_terminal_plane_sensitivity": {
        "report_id": "LEE21-P3",
        "title": "Lee 2021 corrected plasma-terminal plane誤代入感度",
        "input": "13.56 MHz。Lee 2021 Table-Iのcorrected plasma-terminal Z=2.69-j12.8 Ωを、基準面変換なしでideal π fixtureへ代入。",
        "candidate": "C1=1043.829 pF、L1=0.213013 µH、C2=1 pFの固定π fixture 1組。",
        "scenario": "Ar 200 sccm、5 mTorr、source 400 W、bias 100 Wのplasma-terminal plane 1点を誤接続する反実仮想。",
        "control": "なし。",
        "method": "基準面以外を固定し、|Γ|と設計分類の変化を観測する。",
        "interpretation": "基準面変換を省いた代入の電気的影響を示す。published Z自体が誤りという意味ではない。",
    },
    "P1_CCP_fixed_central": {
        "report_id": "GEC-D1",
        "title": "66 Pa中心8点に対する固定整合回路",
        "input": GEC_66_INPUT,
        "candidate": "C1=485.6 pF、L1=1.5 µH、C2=68.97 pFの固定π回路。",
        "scenario": "powered-electrode surfaceの中心8点。",
        "control": "なし。",
        "method": "各点の|Γ|を計算し、8点すべてがplatformの10%反射基準内か判定する。",
        "interpretation": "1つの中心負荷近傍で決めた固定回路が全8点を覆えない負例。論文matcherの再現ではない。",
    },
    "P1_CCP_limited_central": {
        "report_id": "GEC-D2",
        "title": "66 Pa中心8点に対する限定Control",
        "input": GEC_66_INPUT,
        "candidate": "L1=1.5 µHの固定インダクタ1組。",
        "scenario": "中心8点。",
        "control": "C1={450,500} pF、C2={70} pFの2状態。",
        "method": "8×2=16評価から各Scenarioの最小|Γ|状態を選び、全8点の10%反射基準を確認する。",
        "interpretation": "限定Controlはworst reflectionを下げるが、empty-cell resonance 34 MHzラベル群（RF driveは13.56 MHz）の低drive 1点に届かない。",
    },
    "P1_CCP_full_central": {
        "report_id": "GEC-D3",
        "title": "66 Pa中心8点に対する4×4 Control",
        "input": GEC_66_INPUT,
        "candidate": "L1=1.5 µHの固定インダクタ1組。",
        "scenario": "中心8点。",
        "control": GEC_FULL_CONTROL,
        "method": "8×16=128評価から各Scenarioの最小|Γ|を選び、全8点の10%反射基準を確認する。",
        "interpretation": "明示した有限Control範囲で中心8点を覆える正例。制御網はplatform fixtureであり論文装置仕様ではない。",
    },
    "P1_CCP_reported_spread": {
        "report_id": "GEC-D4",
        "title": "Hargis報告装置ばらつき32角点",
        "input": GEC_66_INPUT + " 各中心から報告群内V/I/phase幅の4境界を作る32 deterministic corners。",
        "candidate": "固定L1=1.5 µH。",
        "scenario": "32 apparatus-spread corners。",
        "control": GEC_FULL_CONTROL,
        "method": "32×16=512評価。各角点で最小|Γ|を選び、全角点の10%反射基準を確認する。",
        "interpretation": "中心8点合格だけでは報告ばらつき全体を保証しない。角点は信頼区間や確率標本ではない。",
    },
    "P1_CCP_phase_minus6": {
        "report_id": "GEC-D5",
        "title": "共通位相-6°モデル感度",
        "input": GEC_66_INPUT + " 全8点のV/I位相を一括で-6°回転したcounterfactual dataset。",
        "candidate": "固定L1=1.5 µH。",
        "scenario": "8 transformed points。",
        "control": GEC_FULL_CONTROL,
        "method": "8×16=128評価でcoverageとworst |Γ|を求め、未回転baseline（8/8、worst |Γ|=0.080353）と比較する。",
        "interpretation": "位相モデル形式の共通方向感度を示す。8個の独立誤差や信頼区間ではない。",
    },
    "P1_CCP_phase_plus6": {
        "report_id": "GEC-D6",
        "title": "共通位相+6°モデル感度",
        "input": GEC_66_INPUT + " 全8点のV/I位相を一括で+6°回転したcounterfactual dataset。",
        "candidate": "固定L1=1.5 µH。",
        "scenario": "8 transformed points。",
        "control": GEC_FULL_CONTROL,
        "method": "8×16=128評価でcoverageとworst |Γ|を求め、未回転baseline（8/8、worst |Γ|=0.080353）と比較する。",
        "interpretation": "正方向の共通位相変化では8点を覆う。確率的な不確かさ評価ではない。",
    },
    "literature_colpo1999_fixed_digitization_corners": {
        "report_id": "COLPO-D1",
        "title": "Colpo読取コーナーに対する固定回路",
        "input": "15 parent条件から派生した60個のR±6 Ω、X±7 Ω読取コーナー、13.56 MHz、RFZ60 global terminal。",
        "candidate": "L1=1.0 µH、C1=575 pF、C2=235 pFの固定π回路。",
        "scenario": "15 parent×4 reading corners。",
        "control": "なし。",
        "method": "60評価に加え、各parentが4角すべてで合格するかを集約する。反射電力10%以下。",
        "interpretation": "固定回路は一部parentで読取幅に頑健でない。60角点を60独立運転条件として数えない。",
    },
    "literature_colpo1999_bounded_digitization_corners": {
        "report_id": "COLPO-D2",
        "title": "Colpo読取コーナーに対する有限チューナ",
        "input": "13.56 MHz、RFZ60 global terminal。Colpo 1999の3圧力×5電力=15図読取中心から、各々R±6 Ω・X±7 Ωで派生した60読取コーナー。",
        "candidate": "固定L1=1.0 µH。",
        "scenario": "15 parent×4 reading corners。",
        "control": "C1={450,525,600,675} pF、C2={200,215,230,245} pFの16状態。",
        "method": "60×16=960評価から各角点の最小|Γ|を選び、parent単位の4角robust性も集約する。",
        "interpretation": "宣言した有限tunerは全parentの全読取角を覆う。Colpo論文matcherやproduction hardwareの再現ではない。",
    },
}


DESIGN_QUESTIONS = {
    "literature_p0_lee2021_matcher_output_probe": "Lee 2021のmatcher-output基準面Zを固定platform fixtureへ接続した基準ケースは、反射電力10%以下になるか。",
    "literature_p0_lee2021_post_coax_plane_sensitivity": "matcher-output用に固定したπfixtureへ下流post-coax基準面Zを変換なしで誤代入すると、反射と設計分類はどの程度変わるか。",
    "literature_p0_lee2021_plasma_terminal_plane_sensitivity": "matcher-output用に固定したπfixtureへcorrected plasma-terminal Zを変換なしで誤代入すると、反射と設計分類はどの程度変わるか。",
    "P1_CCP_fixed_central": "66 Paの公開中心8負荷を、一つの固定π回路だけで全点整合できるか。",
    "P1_CCP_limited_central": "66 Paの公開中心8負荷を、固定Lと2つだけの可変容量状態で全点整合できるか。",
    "P1_CCP_full_central": "66 Paの公開中心8負荷を、固定Lと明示した4×4可変容量状態で全点整合できるか。",
    "P1_CCP_reported_spread": "中心8点で成立するチューナは、Hargisの報告幅から作った32装置ばらつき角も全て覆えるか。",
    "P1_CCP_phase_minus6": "全8負荷の位相を共通に-6°回転したモデル感度条件で、同じ回路とControlは全点を覆えるか。",
    "P1_CCP_phase_plus6": "全8負荷の位相を共通に+6°回転したモデル感度条件で、同じ回路とControlは全点を覆えるか。",
    "literature_colpo1999_fixed_digitization_corners": "Colpoの15 parent・60読取コーナーを、一つの固定π回路だけでparent単位に頑健に覆えるか。",
    "literature_colpo1999_bounded_digitization_corners": "Colpoの15 parent・60読取コーナーを、固定Lと有限4×4容量Controlでparent単位に頑健に覆えるか。",
}


DESIGN_PROVENANCE = {
    "literature_p0_lee2021_matcher_output_probe": "Lee 2021 DOI 10.1063/6.0000883, Table I; `bench/literature/p0_lee2021_bias/matcher_output_probe.yaml`",
    "literature_p0_lee2021_post_coax_plane_sensitivity": "Lee 2021 DOI 10.1063/6.0000883, Table I; `bench/literature/p0_lee2021_bias/post_coax_plane_sensitivity.yaml`",
    "literature_p0_lee2021_plasma_terminal_plane_sensitivity": "Lee 2021 DOI 10.1063/6.0000883, Table I; `bench/literature/p0_lee2021_bias/plasma_terminal_plane_sensitivity.yaml`",
    "P1_CCP_fixed_central": "Hargis 1994 DOI 10.1063/1.1144770, Tables III/IV; `bench/literature/p1_gec_ccp/match_fixed_central.yaml`",
    "P1_CCP_limited_central": "Hargis 1994 DOI 10.1063/1.1144770, Tables III/IV; `bench/literature/p1_gec_ccp/match_limited_central.yaml`",
    "P1_CCP_full_central": "Hargis 1994 DOI 10.1063/1.1144770, Tables III/IV; `bench/literature/p1_gec_ccp/match_full_central.yaml`",
    "P1_CCP_reported_spread": "Hargis 1994 DOI 10.1063/1.1144770, Tables III/IV; `bench/literature/p1_gec_ccp/match_full_reported_spread.yaml`",
    "P1_CCP_phase_minus6": "Hargis 1994 DOI 10.1063/1.1144770 + Sobolewski 1995 DOI 10.6028/jres.100.026; `bench/literature/p1_gec_ccp/match_full_phase_minus6.yaml`",
    "P1_CCP_phase_plus6": "Hargis 1994 DOI 10.1063/1.1144770 + Sobolewski 1995 DOI 10.6028/jres.100.026; `bench/literature/p1_gec_ccp/match_full_phase_plus6.yaml`",
    "literature_colpo1999_fixed_digitization_corners": "Colpo 1999 DOI 10.1063/1.369268; `bench/literature/p1_colpo1999_icp/digitized/match_fixed_uncertainty_corners.yaml`",
    "literature_colpo1999_bounded_digitization_corners": "Colpo 1999 DOI 10.1063/1.369268; `bench/literature/p1_colpo1999_icp/digitized/match_bounded_uncertainty_corners.yaml`",
}

DESIGN_SOURCE_PATHS = {
    "literature_p0_lee2021_matcher_output_probe": "bench/literature/p0_lee2021_bias/matcher_output_probe.yaml",
    "literature_p0_lee2021_post_coax_plane_sensitivity": "bench/literature/p0_lee2021_bias/post_coax_plane_sensitivity.yaml",
    "literature_p0_lee2021_plasma_terminal_plane_sensitivity": "bench/literature/p0_lee2021_bias/plasma_terminal_plane_sensitivity.yaml",
    "P1_CCP_fixed_central": "bench/literature/p1_gec_ccp/match_fixed_central.yaml",
    "P1_CCP_limited_central": "bench/literature/p1_gec_ccp/match_limited_central.yaml",
    "P1_CCP_full_central": "bench/literature/p1_gec_ccp/match_full_central.yaml",
    "P1_CCP_reported_spread": "bench/literature/p1_gec_ccp/match_full_reported_spread.yaml",
    "P1_CCP_phase_minus6": "bench/literature/p1_gec_ccp/match_full_phase_minus6.yaml",
    "P1_CCP_phase_plus6": "bench/literature/p1_gec_ccp/match_full_phase_plus6.yaml",
    "literature_colpo1999_fixed_digitization_corners": "bench/literature/p1_colpo1999_icp/digitized/match_fixed_uncertainty_corners.yaml",
    "literature_colpo1999_bounded_digitization_corners": "bench/literature/p1_colpo1999_icp/digitized/match_bounded_uncertainty_corners.yaml",
}

DESIGN_LIMITATIONS = {
    "literature_p0_lee2021_matcher_output_probe": "Leeのmatcher-output測定点をplatformのideal π fixtureへ接続した一条件の静的判定である。論文matcher、実ケーブル、部品定格、装置運転範囲は再現しない。",
    "literature_p0_lee2021_post_coax_plane_sensitivity": "異なる基準面Zを変換せず誤接続した反実仮想である。基盤が基準面誤りを自動検出することやde-embeddingの妥当性は示さない。",
    "literature_p0_lee2021_plasma_terminal_plane_sensitivity": "published plasma-terminal Zの誤りを意味しない。matcher-output用fixtureへ変換なしで代入した場合の感度だけを示す。",
    "P1_CCP_fixed_central": "75/100/150/200 Vppは文献条件ラベルであり、このplatform回路のsource振幅ではない。絶対電圧・電流・損失は評価しない。",
    "P1_CCP_limited_central": "有限格子の到達性だけを評価し、control reserve、連続可動域、追従速度、安定性を資格付けしない。Vppは文献条件ラベルである。",
    "P1_CCP_full_central": "全8点到達は宣言した有限格子内の静的結果である。観測grid margin=0であり、実チューナの予備量や端部回避は資格付けしていない。",
    "P1_CCP_reported_spread": "32角点は報告幅から作った決定論的包絡で、独立標本、信頼区間、発生確率ではない。control reserveも資格付けしない。",
    "P1_CCP_phase_minus6": "全点を一括回転した一つのcommon-modeモデル仮説であり、8個の独立誤差や測定信頼区間ではない。",
    "P1_CCP_phase_plus6": "全点を一括回転した一つのcommon-modeモデル仮説である。全点到達してもcontrol reserveと実装置の位相不確かさは資格付けしない。",
    "literature_colpo1999_fixed_digitization_corners": "60角は15 parent内の図読取幅であり、60の独立運転や統計標本ではない。fixed回路の結果を実ICP装置へ一般化しない。",
    "literature_colpo1999_bounded_digitization_corners": "有限格子が読取幅を覆う結果であり、論文matcher、production hardware、連続負荷領域、control reserveを資格付けしない。",
}

LITERATURE_POWER_NOTES = {
    "GEC-S1": "P1=0.5VIcosφは公開peak V/I基本波からの電気ポート量である。論文掲載powerは最初の5高調波を含むため、この比較は熱収支閉包ではない。",
    "LEE20-M1": "Pabsは既知paper-transformerパラメータとpeak coil currentを与えた式適合用補助量である。一般設計結果がプラズマ内部電力を同定・分配することを意味しない。",
    "LEE21-T1": "source 400 W / bias 100 Wは文献運転条件ラベルであり、この基盤が再計算したgenerator powerではない。",
    "COLPO-S1": "掲載RF powerは図の運転条件ラベルであり、この基盤がsource端子電力を再現した値ではない。",
}

DESIGN_EVIDENCE_GROUPS = {
    **{
        case_id: "公開数値＋platform基準面感度fixture"
        for case_id in DESIGN_META
        if case_id.startswith("literature_p0_lee2021")
    },
    **{
        case_id: "公開V/I/phase由来負荷＋platform設計fixture" for case_id in DESIGN_META if case_id.startswith("P1_CCP")
    },
    **{
        case_id: "二読者図読取＋決定論的読取角＋platform設計fixture"
        for case_id in DESIGN_META
        if case_id.startswith("literature_colpo1999")
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage_text(feasible_count: int, scenario_count: int) -> str:
    return f"{feasible_count}/{scenario_count}"


def _core_amplitude(case_id: str) -> str:
    if case_id == "B5_high_drive_stress":
        return "理想Vsrc端子 25/100 Vpeak（基本波）"
    return "正規化1 VpeakのAC小信号。絶対ストレスは評価しない"


def _core_limiting_constraints(case: dict[str, Any]) -> str:
    names = sorted({str(name) for scenario in case["scenarios"] for name in scenario.get("violated_constraints", [])})
    return "、".join(CONSTRAINT_LABELS.get(name, name) for name in names) if names else "なし"


def _criterion_status(feasible: bool) -> str:
    return "MET" if feasible else "NOT MET"


def _b5_stress_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = [
        ("L1 Irms", "max_component_L1_current_rms_A", "A rms"),
        ("L1固定Rseries損失", "max_component_L1_loss_W", "W"),
        ("理想源端子 Irms", "max_source_current_rms_A", "A rms"),
        ("理想源端子 VA", "max_source_apparent_power_VA", "VA"),
    ]
    rows: list[dict[str, Any]] = []
    for item in candidate["scenarios"]:
        selected = item["selected"]
        constraints = {row["name"]: row for row in selected["constraints"]}
        values = selected["metrics"]
        drive = float(item["scenario"]["values"]["drive_amplitude_V"])
        for label, constraint_name, unit in metrics:
            constraint = constraints[constraint_name]
            rows.append(
                {
                    "metric": label,
                    "scenario": f"{drive:g} Vpeak",
                    "drive_vpeak": drive,
                    "value": float(constraint["value"]),
                    "limit": float(constraint["limit"]),
                    "utilization": float(constraint["value"]) / float(constraint["limit"]),
                    "unit": unit,
                    "screen": "PASS" if constraint["satisfied"] else "FAIL",
                    "reference_plane": "ideal Vsrc / electrode_terminal",
                    "source_real_power_W": float(values["source_real_power_W"]),
                    "load_real_power_W": float(values["load_real_power_W"]),
                    "network_loss_W": float(values["network_loss_W"]),
                    "transfer_efficiency": float(values["transfer_efficiency"]),
                }
            )
    return rows


def _validate_b5_candidate_snapshot(core_cases: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    core_b5 = next(case for case in core_cases if case["benchmark_id"] == "B5_high_drive_stress")
    if candidate["candidate"]["candidate_id"] != core_b5["best_candidate_id"]:
        raise ValueError("B5 candidate detail does not match the core selected candidate id")
    if candidate["candidate"]["values"] != core_b5["best_candidate_values"]:
        raise ValueError("B5 candidate detail does not match the core selected component values")

    core_by_scenario = {row["scenario_id"]: row for row in core_b5["scenarios"]}
    detail_by_scenario = {row["scenario"]["scenario_id"]: row for row in candidate["scenarios"]}
    if set(core_by_scenario) != set(detail_by_scenario):
        raise ValueError("B5 candidate detail scenario ids do not match the core snapshot")
    for scenario_id, core_row in core_by_scenario.items():
        detail_gamma = float(detail_by_scenario[scenario_id]["selected"]["metrics"]["reflection_magnitude"])
        if not math.isclose(detail_gamma, float(core_row["reflection_magnitude"]), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"B5 candidate detail reflection does not match core scenario {scenario_id!r}")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_candidate_result(run_root: Path, case_id: str) -> Path:
    matches = []
    for path in sorted(run_root.glob("*/candidates/trial_*.json")):
        payload = _load(path)
        scenarios = payload.get("scenarios") or []
        if not scenarios:
            continue
        selected = scenarios[0].get("selected") or {}
        observations = (selected.get("raw") or {}).get("observations") or {}
        if observations.get("case_id") == case_id:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one candidate result for {case_id} under {run_root}, found {len(matches)}")
    return matches[0]


def _pct(value: float) -> float:
    return round(100.0 * value, 3)


def _gamma(value: float) -> str:
    return f"{value:.6g}"


def _relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _complex_text(resistance: float, reactance: float) -> str:
    return f"{resistance:.6g}{reactance:+.6g}j Ω"


def _component_map_text(values: dict[str, Any]) -> str:
    rendered = []
    for name, raw in values.items():
        value = float(raw)
        if name.startswith("C"):
            rendered.append(f"{name}={value * 1e12:.6g} pF")
        elif name.startswith("L"):
            rendered.append(f"{name}={value * 1e6:.6g} µH")
        else:
            rendered.append(f"{name}={value:.6g}")
    return "、".join(rendered) if rendered else "なし"


def _core_scenario_evidence(case: dict[str, Any]) -> str:
    oracle_by_scenario = case.get("expected", {}).get("input_impedance_ohm", {})
    show_margin = case["benchmark_id"] in {"B2_limited_tuner", "B3_full_tuner"}
    lines = []
    for scenario in case["scenarios"]:
        scenario_id = scenario["scenario_id"]
        parts = [
            f"Zin={_complex_text(scenario['input_resistance_ohm'], scenario['input_reactance_ohm'])}",
            f"|Γ|={_gamma(scenario['reflection_magnitude'])}",
        ]
        oracle = oracle_by_scenario.get(scenario_id)
        if oracle:
            component_error = max(
                abs(float(scenario["input_resistance_ohm"]) - float(oracle[0])),
                abs(float(scenario["input_reactance_ohm"]) - float(oracle[1])),
            )
            parts.extend(
                [
                    f"独立oracle={_complex_text(float(oracle[0]), float(oracle[1]))}",
                    f"max(|ΔR|,|ΔX|)={component_error:.3g} Ω",
                ]
            )
        if scenario["control"]:
            parts.append(f"選択Control={_component_map_text(scenario['control'])}")
        if show_margin:
            parts.append(f"正規化余裕={float(scenario['control_margin']):.6g}")
        violations = scenario["violated_constraints"]
        outcome = "合格" if scenario["feasible"] else "不合格"
        parts.append(f"宣言条件={outcome}")
        if violations:
            parts.append(f"違反={','.join(violations)}")
        lines.append(f"- `{scenario_id}`: " + "、".join(parts))
    return "\n".join(lines)


def _b5_stress_evidence(candidate: dict[str, Any]) -> str:
    lines = [
        "",
        "",
        "**合成電気スクリーニング値と宣言閾値**",
        "",
    ]
    constraint_labels = {
        "max_component_L1_current_rms_A": "L1 Irms",
        "max_component_L1_loss_W": "L1固定Rseries平均実電力",
        "max_source_current_rms_A": "理想源端子 Irms",
        "max_source_apparent_power_VA": "理想源端子 Vrms×Irms",
    }
    for item in candidate["scenarios"]:
        selected = item["selected"]
        metrics = selected["metrics"]
        drive = item["scenario"]["values"]["drive_amplitude_V"]
        constraints = {constraint["name"]: constraint for constraint in selected["constraints"]}
        screen_parts = []
        for constraint_name, label in constraint_labels.items():
            constraint = constraints[constraint_name]
            screen_parts.append(
                f"{label}={float(constraint['value']):.6g}/{float(constraint['limit']):.6g} "
                f"({'PASS' if constraint['satisfied'] else 'FAIL'})"
            )
        closure = constraints["max_component_loss_balance_fraction_of_source"]
        lines.append(f"- `{item['scenario']['scenario_id']}` ({drive:g} Vpeak): " + "、".join(screen_parts))
        lines.append(
            f"  電力ポート: source real={metrics['source_real_power_W']:.6g} W、"
            f"load-plane real={metrics['load_real_power_W']:.6g} W、"
            f"network loss={metrics['network_loss_W']:.6g} W、"
            f"電気伝送効率={100 * metrics['transfer_efficiency']:.6g}% 。"
            f"数値閉包残差率={float(closure['value']):.3g}/{float(closure['limit']):.3g} (PASS)。"
        )
        lines.append(
            f"  制約未設定の観測値: C1 Vpeak={metrics['component_C1_voltage_peak_V']:.6g} V、"
            f"C1 Irms={metrics['component_C1_current_rms_A']:.6g} A、"
            f"C2 Vpeak={metrics['component_C2_voltage_peak_V']:.6g} V、"
            f"C2 Irms={metrics['component_C2_current_rms_A']:.6g} A。"
        )
    lines.extend(
        [
            "",
            "閾値は合成benchmark assumptionであり、ベンダ定格ではない。源端VAは内部50 Ωを持たない理想Vsrc端子のVrms×Irmsで、generator forward powerではない。",
        ]
    )
    return "\n".join(lines)


def _widget_source(source: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Attach the exact snapshot-table query used by a native report widget."""
    widget_source = {**source, "query": dict(source["query"])}
    widget_source["query"].update(
        {
            "engine": "artifact-snapshot",
            "language": "sql",
            "sql": f'SELECT * FROM "{dataset}"',  # noqa: S608 - internal report identifier
            "tables_used": [dataset],
        }
    )
    return widget_source


def _core_result_sentence(case: dict[str, Any]) -> str:
    scenarios = len(case["scenarios"])
    covered = round(case["feasible_fraction"] * scenarios)
    benchmark = "PASS" if case["passed"] else "FAIL"
    status = _criterion_status(bool(case["feasible"]))
    text = (
        f"Benchmark reproduction: **{benchmark}**。Case-local declared criteria: **{status}**。"
        f"Scenario coverage={covered}/{scenarios}、worst |Γ|={_gamma(case['worst_reflection_magnitude'])}、"
        f"SPICE評価={case['n_evaluations']}。"
    )
    if case["benchmark_id"] == "B6_discrete_hardware_search":
        text += (
            f" 3 Candidate中{case['feasible_candidates']}件が合格し、"
            f"C1={case['best_candidate_values']['C1'] * 1e12:.3f} pFを選択。"
        )
    if case["benchmark_id"] == "B7_role_factorial_search":
        text += f" 2 Candidate中、全Scenarioを覆った唯一のL1={case['best_candidate_values']['L1'] * 1e6:.6g} µHを選択。"
    if case["benchmark_id"] == "D2_reference_plane_embedded":
        text += " fixture明示表現のoracleと同じ源側Zおよび設計分類を再現。"
    return text


def _core_block(
    case: dict[str, Any],
    b5_candidate: dict[str, Any] | None = None,
    *,
    core_result_path: str,
) -> dict[str, Any]:
    meta = CORE_META[case["benchmark_id"]]
    detail = _core_scenario_evidence(case)
    if case["benchmark_id"] == "B5_high_drive_stress":
        if b5_candidate is None:
            raise ValueError("B5 candidate detail is required for the report")
        detail += _b5_stress_evidence(b5_candidate)
    body = f"""## {case["benchmark_id"].split("_")[0]} — {meta["title"]}

**このケース単独の問い**
{CORE_QUESTIONS[case["benchmark_id"]]}

**問題設定と入力**
{meta["input"]}

- **ケース種別 / evidence:** {CORE_EVIDENCE_GROUPS[case["benchmark_id"]]}。
- **Machine-readable ID:** `{case["benchmark_id"]}`。
- **解析:** 基本波AC、評価周波数は入力記載どおり。
- **負荷reference plane:** {CORE_REFERENCE_PLANES[case["benchmark_id"]]}。
- **振幅定義:** {_core_amplitude(case["benchmark_id"])}。
- **Candidate:** {meta["candidate"]}
- **Scenario:** {meta["scenario"]}
- **Control:** {meta["control"]}
- **設計合格条件:** 50 Ω基準で |Γ|≤0.316228（反射電力10%以下）。追加条件がある場合は以下の評価方法に含む。
- **結果の用途:** {CORE_OUTCOME_SCOPES[case["benchmark_id"]]}。装置・工程・熱の資格判定ではない。
- **根拠ファイル:** 定義 `{_relative(case["case_path"])}`、結果 `{core_result_path}`。

**評価方法**
{meta["method"]}

**観測結果**
{_core_result_sentence(case)}

**Scenario別の判定根拠**
{detail}

**この結果から分かること**
{meta["interpretation"]}

**分からないこと**
{CORE_LIMITATIONS[case["benchmark_id"]]}
"""
    return {
        "id": f"case-{case['benchmark_id'].lower()}",
        "type": "markdown",
        "layout": "full",
        "body": body,
        "sourceId": f"source-{case['benchmark_id'].lower()}",
    }


def _literature_static_block(
    meta: dict[str, str], source_rows: dict[str, dict[str, Any]], model_rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = source_rows.get(meta["result_id"]) or model_rows[meta["result_id"]]
    benchmark = "PASS" if result["passed"] else "FAIL"
    power_note = LITERATURE_POWER_NOTES.get(meta["id"], "")
    if power_note:
        power_note = f"\n\n**電力・振幅の読み方**  \n{power_note}"
    body = f"""## {meta["id"]} — {meta["title"]}

**このケース単独の問い**
{meta["question"]}

**問題設定と入力**
{meta["input"]}

- **Candidate:** {meta["candidate"]}
- **Scenario / test vector:** {meta["scenario"]}
- **Control:** {meta["control"]}
- **Evidence / 解析:** {result["evidence_class"]}。文献値または合成代数vectorを用いる決定論的conformance。
- **Case-local design criteria:** N/A。これは設計可否ではなく{meta["purpose"]}の検証。
- **装置・工程・熱資格:** NOT EVALUATED。
- **Benchmark合格条件:** {meta["acceptance"]}
- **出典と根拠ファイル:** {meta["provenance"]}
- **結果artifact:** `{_relative(result["result"])}`。

**評価方法**
{meta["method"]}

**観測結果**
Benchmark reproduction: **{benchmark}**。{result["observed"]}。{power_note}

**この結果から分かること**
{meta["establishes"]}

**分からないこと**
{meta["does_not_establish"]}
"""
    return {
        "id": f"case-{meta['id'].lower()}",
        "type": "markdown",
        "layout": "full",
        "body": body,
        "sourceId": f"source-{meta['id'].lower()}",
    }


def _design_amplitude(result_id: str) -> str:
    if result_id.startswith("P1_CCP"):
        return "75/100/150/200 VppはHargis公開条件のラベル。platform matcherのsource振幅は未指定で、絶対V/I/損失は評価しない"
    if result_id.startswith("literature_p0_lee2021"):
        return "論文のsource/bias powerは条件ラベル。platform fixtureは正規化ACで絶対ストレスを評価しない"
    return "論文のRF powerは条件ラベル。platform fixtureは正規化ACで絶対ストレスを評価しない"


def _design_block(row: dict[str, Any], *, literature_result_path: str) -> dict[str, Any]:
    meta = DESIGN_META[row["id"]]
    benchmark = "PASS" if row["regression_passed"] else "FAIL"
    status = _criterion_status(bool(row["design_feasible"]))
    margin = "Control reserve: N/A（Controlなし）。"
    if meta["control"] != "なし。":
        if row.get("control_margin") is None:
            margin = "Control reserve: 集約値なし・資格未実施。"
        else:
            margin = (
                f"観測された離散grid margin={float(row['control_margin']):.6g}。"
                "reserveの合格条件は宣言しておらず、予備量は資格未実施。"
            )
    parent = ""
    if row.get("parent_condition_count"):
        parent = f" parent robust={row['robust_parent_count']}/{row['parent_condition_count']}。"
    body = f"""## {meta["report_id"]} — {meta["title"]}

**このケース単独の問い**
{DESIGN_QUESTIONS[row["id"]]}

**問題設定と入力**
{meta["input"]}

- **ケース種別 / evidence:** {DESIGN_EVIDENCE_GROUPS[row["id"]]}。
- **解析:** 13.56 MHzの基本波ACによる静的一ポート設計challenge。
- **振幅定義:** {_design_amplitude(row["id"])}。
- **Candidate:** {meta["candidate"]}
- **Scenario:** {meta["scenario"]}
- **Control:** {meta["control"]}
- **合格条件:** 50 Ω基準で反射電力10%以下。これはplatform criterionであり論文の普遍基準ではない。
- **Benchmark回帰条件:** 期待する設計分類、coverage、worst |Γ|、評価数を凍結expectationと照合する。
- **装置・工程・熱資格:** NOT EVALUATED。
- **出典と根拠ファイル:** {DESIGN_PROVENANCE[row["id"]]}。統合結果 `{literature_result_path}`。

**評価方法**
{meta["method"]}

**観測結果**
Benchmark reproduction: **{benchmark}**。Case-local declared criteria: **{status}**。coverage={row["feasible_count"]}/{row["scenario_count"]}、worst |Γ|={_gamma(row["worst_reflection_magnitude"])}、評価数={row["n_evaluations"]}。{margin}{parent}

結果scope: {row["scope"]}。

**この結果から分かること**
{meta["interpretation"]}

**分からないこと**
{DESIGN_LIMITATIONS[row["id"]]}
"""
    return {
        "id": f"case-{meta['report_id'].lower()}",
        "type": "markdown",
        "layout": "full",
        "body": body,
        "sourceId": f"source-{meta['report_id'].lower()}",
    }


def _hardware_family_block(rows: list[dict[str, Any]], *, literature_result_path: str) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    families = [
        ("central_operating_conditions", "32 published central conditions"),
        ("reported_apparatus_spread", "32 deterministic apparatus-spread corners"),
        ("phase_model_minus6", "8-point common-mode -6° sensitivity"),
        ("phase_model_plus6", "8-point common-mode +6° sensitivity"),
    ]
    table_lines = [
        "| Evidence family | L1=1.4 µH | L1=1.5 µH | L1=1.6 µH |",
        "|---|---:|---:|---:|",
    ]
    for family, label in families:
        cells = []
        for inductance in ("1.4", "1.5", "1.6"):
            row = by_id[f"{family}__L1_{inductance}uH"]
            cells.append(
                f"{row['feasible_count']}/{row['scenario_count']}; Γmax={_gamma(row['worst_reflection_magnitude'])}"
            )
        table_lines.append(f"| {label} | {' | '.join(cells)} |")
    benchmark = "PASS" if all(row["regression_passed"] for row in rows) else "FAIL"
    criteria_met = [row["id"] for row in rows if row["design_feasible"]]
    body = f"""## GEC-D7 — 4 evidence family × 3固定L候補

**このケース単独の問い**
L1={{1.4、1.5、1.6}} µHの固定ハード候補を、性質の異なる4つのevidence familyで別々に評価すると、設計判断はどの程度変わるか。

**問題設定と入力**

- **回路と基準面:** 13.56 MHz、powered-electrode-surfaceのR+jXを接続するπ-match。
- **Candidate:** L1={{1.4、1.5、1.6}} µHの3固定候補。
- **Scenario:** published central 32点、reported apparatus spread 32角、phase -6° 8点、phase +6° 8点を別familyとして扱う。
- **Control:** 各ScenarioでC1={{400、450、500、550}} pF、C2={{65、70、75、80}} pFの16状態。
- **振幅定義:** Hargisの75/100/150/200 Vppは公開条件ラベルであり、platform回路のsource振幅ではない。絶対電圧・電流・損失は評価しない。
- **評価数:** 3候補×(32+32+8+8)×16=3,840。
- **合格条件:** 各family内の全Scenarioで反射電力10%以下。family間を合算しない。
- **Control reserve:** 合格条件を宣言していないため資格未実施。grid端使用を装置余裕と解釈しない。
- **装置・工程・熱資格:** NOT EVALUATED。
- **出典と根拠ファイル:** Hargis 1994 DOI 10.1063/1.1144770 + Sobolewski 1995 DOI 10.6028/jres.100.026; `bench/literature/p1_gec_ccp/hardware_family_spec.yaml`、結果 `{literature_result_path}`。

**評価方法**
各Candidateについて、4つのevidence familyを独立に全列挙する。各Scenarioでは16 Control中の最小|Γ|を選び、family内の全Scenario合格可否を判定する。family間の行数を重みとして合算せず、family別ベクトルで比較する。

**観測結果**

{chr(10).join(table_lines)}

12 Candidate-family回帰のBenchmark reproductionは **{benchmark}**。Case-local declared criteriaを満たす組は{len(criteria_met)}/12で、`{", ".join(criteria_met) if criteria_met else "なし"}`。familyをまたぐ単一coverageや総合winnerは定義しない。

**この結果から分かること**
候補の優劣はevidence familyに依存するため、行数を合算した総合winnerは論理的に正当化できない。実装置の優先familyまたは目的関数が必要である。

**分からないこと**
各familyの発生確率、相対重要度、実装置に対する最適L1は与えられていない。
"""
    return {
        "id": "case-gec-d7",
        "type": "markdown",
        "layout": "full",
        "body": body,
        "sourceId": "source-gec-d7",
    }


def build_artifact(  # noqa: C901
    core: dict[str, Any],
    literature: dict[str, Any],
    *,
    core_path: Path = DEFAULT_CORE,
    literature_path: Path = DEFAULT_LITERATURE,
    b5_candidate_path: Path | None = None,
) -> dict[str, Any]:
    core_cases = core["cases"]
    if b5_candidate_path is None:
        b5_candidate_path = _find_candidate_result(core_path.parent, "benchmark_match_high_drive_stress")
    b5_candidate = _load(b5_candidate_path)
    core_result_path = _relative(core_path)
    literature_result_path = _relative(literature_path)
    if set(CORE_META) != {case["benchmark_id"] for case in core_cases}:
        raise ValueError("Core case metadata does not match the benchmark result")
    _validate_b5_candidate_snapshot(core_cases, b5_candidate)

    source_rows = {row["id"]: row for row in literature["source_fidelity"]}
    model_rows = {row["id"]: row for row in literature["model_conformance"]}
    design_rows = literature["design_challenges"]
    design_by_id = {row["id"]: row for row in design_rows}
    focused_design_ids = list(DESIGN_META)
    missing = set(focused_design_ids) - set(design_by_id)
    if missing:
        raise ValueError(f"Missing literature design results: {sorted(missing)}")

    core_index = []
    for case in core_cases:
        meta = CORE_META[case["benchmark_id"]]
        scenario_count = len(case["scenarios"])
        feasible_count = round(float(case["feasible_fraction"]) * scenario_count)
        core_index.append(
            {
                "case_id": case["benchmark_id"].split("_")[0],
                "case_name": meta["title"],
                "purpose": meta["purpose"],
                "evidence_group": CORE_EVIDENCE_GROUPS[case["benchmark_id"]],
                "reference_plane": CORE_REFERENCE_PLANES[case["benchmark_id"]],
                "outcome_scope": CORE_OUTCOME_SCOPES[case["benchmark_id"]],
                "scenario_count": scenario_count,
                "candidate_count": case["n_candidates"],
                "evaluation_count": case["n_evaluations"],
                "coverage": _coverage_text(feasible_count, scenario_count),
                "coverage_pct": _pct(case["feasible_fraction"]),
                "worst_gamma": case["worst_reflection_magnitude"],
                "limiting_constraint": _core_limiting_constraints(case),
                "benchmark": "PASS" if case["passed"] else "FAIL",
                "declared_criteria": _criterion_status(bool(case["feasible"])),
                "engineering_raw": "FEASIBLE" if case["feasible"] else "INFEASIBLE",
            }
        )

    literature_checks = []
    for meta in LITERATURE_STATIC:
        result = source_rows.get(meta["result_id"]) or model_rows[meta["result_id"]]
        literature_checks.append(
            {
                "case_id": meta["id"],
                "case_name": meta["title"],
                "purpose": meta["purpose"],
                "evidence": result["evidence_class"],
                "benchmark": "PASS" if result["passed"] else "FAIL",
                "declared_criteria": "N/A",
                "observed": result["observed"],
            }
        )

    # Colpo center fidelity has a second, inseparable deterministic-derivation
    # assertion.  It remains evidence inside COLPO-S1 rather than a fake extra
    # physical case.
    colpo_derivation = source_rows["colpo1999_digitization_derivation"]

    literature_design_index = []
    for result_id in focused_design_ids:
        row = design_by_id[result_id]
        meta = DESIGN_META[result_id]
        literature_design_index.append(
            {
                "case_id": meta["report_id"],
                "case_name": meta["title"],
                "scenario_count": row["scenario_count"],
                "scenario_scope": str(row["scenario_count"]),
                "evaluation_count": row["n_evaluations"],
                "coverage": _coverage_text(row["feasible_count"], row["scenario_count"]),
                "coverage_pct": _pct(row["feasible_count"] / row["scenario_count"]),
                "worst_gamma": row["worst_reflection_magnitude"],
                "evidence_group": DESIGN_EVIDENCE_GROUPS[result_id],
                "reserve_status": (
                    "N/A（Controlなし）" if meta["control"] == "なし。" else "未資格（離散grid到達性のみ）"
                ),
                "null_reason": "—",
                "benchmark": "PASS" if row["regression_passed"] else "FAIL",
                "declared_criteria": _criterion_status(bool(row["design_feasible"])),
                "engineering_raw": "FEASIBLE" if row["design_feasible"] else "INFEASIBLE",
            }
        )

    hardware_rows = [row for row in design_rows if "__L1_" in row["id"]]
    expected_hardware_ids = {
        f"{family}__L1_{inductance}uH"
        for family in (
            "central_operating_conditions",
            "reported_apparatus_spread",
            "phase_model_minus6",
            "phase_model_plus6",
        )
        for inductance in ("1.4", "1.5", "1.6")
    }
    if {row["id"] for row in hardware_rows} != expected_hardware_ids:
        raise ValueError("GEC-D7 requires exactly 12 unique candidate-family rows")
    literature_design_index.append(
        {
            "case_id": "GEC-D7",
            "case_name": "4 evidence family × 3固定L候補",
            "scenario_count": None,
            "scenario_scope": "32 / 32 / 8 / 8（別family）",
            "evaluation_count": sum(row["n_evaluations"] for row in hardware_rows),
            "coverage": "N/A—family別集約しない",
            "coverage_pct": None,
            "worst_gamma": None,
            "evidence_group": "4つの決定論的evidence family×3固定L候補",
            "reserve_status": "未資格（離散grid到達性のみ）",
            "null_reason": "coverage/worst |Γ|はfamily間で集約しない",
            "benchmark": "PASS" if all(row["regression_passed"] for row in hardware_rows) else "FAIL",
            "declared_criteria": "FAMILY-DEPENDENT",
            "engineering_raw": "FAMILY-DEPENDENT",
        }
    )

    inventory: list[dict[str, Any]] = [
        {
            "category": "コア：回路エンジン適合",
            "case_count": 3,
            "scope": "A1-A3",
            "evidence_kind": "独立解析式＋合成入力",
        },
        {
            "category": "コア：CCP/ICP端子式適合",
            "case_count": 2,
            "scope": "A4-A5",
            "evidence_kind": "独立端子式＋合成入力",
        },
        {
            "category": "コア：設計workflow",
            "case_count": 8,
            "scope": "B1-B8",
            "evidence_kind": "合成・決定論的設計fixture",
        },
        {
            "category": "コア：負荷評価面",
            "case_count": 3,
            "scope": "D1-D3",
            "evidence_kind": "解析式＋境界fixture",
        },
        {
            "category": "文献：公開値の忠実再現",
            "case_count": 4,
            "scope": "Lee/Hargis/Colpo",
            "evidence_kind": "公開表または図読取",
        },
        {
            "category": "文献：モデル式適合",
            "case_count": 1,
            "scope": "Lee 2020",
            "evidence_kind": "論文式＋合成代数vector",
        },
        {
            "category": "文献：platform設計challenge",
            "case_count": 12,
            "scope": "Lee/GEC/Colpo",
            "evidence_kind": "文献負荷＋platform fixture",
        },
    ]
    if sum(row["case_count"] for row in inventory) != 33:
        raise AssertionError("Logical benchmark inventory must contain 33 cases")

    generated_at = datetime.now(timezone.utc).isoformat()
    static_passes = sum(
        bool((source_rows.get(meta["result_id"]) or model_rows[meta["result_id"]])["passed"])
        and (meta["id"] != "COLPO-S1" or bool(colpo_derivation["passed"]))
        for meta in LITERATURE_STATIC
    )
    focused_design_passes = sum(bool(design_by_id[result_id]["regression_passed"]) for result_id in focused_design_ids)
    hardware_family_pass = int(all(row["regression_passed"] for row in hardware_rows))
    reproduced_questions = (
        sum(bool(case["passed"]) for case in core_cases) + static_passes + focused_design_passes + hardware_family_pass
    )
    b5_stress = _b5_stress_rows(b5_candidate)
    core_reproduced = sum(bool(case["passed"]) for case in core_cases)
    core_met_ids = [case["benchmark_id"].split("_")[0] for case in core_cases if case["feasible"]]
    conformance_met_ids = [case_id for case_id in core_met_ids if case_id.startswith(("A", "D"))]
    workflow_met_ids = [case_id for case_id in core_met_ids if case_id.startswith("B")]
    b5_screen_by_scenario = {
        scenario: "PASS" if all(row["screen"] == "PASS" for row in b5_stress if row["scenario"] == scenario) else "FAIL"
        for scenario in {row["scenario"] for row in b5_stress}
    }
    b5_high_fail_count = sum(row["screen"] == "FAIL" for row in b5_stress if row["scenario"] == "100 Vpeak")
    summary = [
        {
            "logical_cases": 33,
            "reproduced_questions": reproduced_questions,
            "core_cases": len(core_cases),
            "core_evaluations": core["n_evaluations"],
            "core_feasible_cases": sum(bool(case["feasible"]) for case in core_cases),
            "literature_logical_cases": 17,
            "literature_runners": len(literature["execution_regressions"]),
            "literature_runners_passed": sum(bool(row["passed"]) for row in literature["execution_regressions"]),
            "literature_design_rows": len(design_rows),
            "apparatus_qualification": "未実施（対象装置未定）",
        }
    ]
    if reproduced_questions > 33:
        raise AssertionError("Reproduced logical question count exceeds the inventory")

    decision_ladder = [
        {
            "stage": "1. Regression reproduction",
            "question": "実装は凍結期待値、独立式、列挙数、期待分類を再現したか",
            "current_evidence": f"{reproduced_questions}/33 logical questions reproduced",
            "status": "PASS" if reproduced_questions == 33 else "FAIL",
            "meaning": "ソフトウェア・変換・式の回帰。装置成立性ではない",
        },
        {
            "stage": "2. Static match reachability",
            "question": "宣言Scenarioで、固定Candidateと許可Controlが反射基準へ到達するか",
            "current_evidence": "core B1-B4/B6-B8、文献由来design challenges",
            "status": "ケース別",
            "meaning": "有限・決定論的入力集合内の静的一ポート判断",
        },
        {
            "stage": "3. Declared electrical screen",
            "question": "振幅・固定Rseries・合成閾値を与えたとき端子V/I/損失制約を満たすか",
            "current_evidence": (
                "B5のみ。"
                f"25 Vpeak {b5_screen_by_scenario.get('25 Vpeak', 'N/A')}、"
                f"100 Vpeak {b5_screen_by_scenario.get('100 Vpeak', 'N/A')}"
            ),
            "status": "限定的",
            "meaning": "合成電気閾値のscreen。部品定格・温度・寿命ではない",
        },
        {
            "stage": "4. Apparatus qualification",
            "question": "校正済み保持データと実部品・fixture・制御・工程制約を満たすか",
            "current_evidence": "対象装置と保持データなし",
            "status": "NOT EVALUATED",
            "meaning": "実チャンバー、工程、熱、寿命の承認には使用不可",
        },
    ]
    practitioner_guide = [
        {
            "reader": "半導体製造/RF装置",
            "usable_now": "指定reference planeのR+jX、固定matcher、有限tunerによる静的基本波スクリーニング",
            "not_supported": "点火、pulsed matching、非線形高調波、process outcome、実generator/coupler資格",
            "next_evidence": "装置ID、校正済み複素V/I、周波数・振幅面、fixture変換、matcher範囲、保持条件",
        },
        {
            "reader": "伝熱/部品設計",
            "usable_now": "明示した固定Rseriesへ帰属する部品別平均実電力、Vpeak、Irmsの受け渡し",
            "not_supported": "温度、hot spot、冷却、熱抵抗、温度依存ESR、寿命、絶縁margin",
            "next_evidence": "実部品、損失モデル、周囲・冷却・実装、duty、材料/定格/derating",
        },
        {
            "reader": "データサイエンス",
            "usable_now": "決定論的scenario/candidate/control完全列挙、出典追跡、case-local worst-case",
            "not_supported": "coverageからの確率・歩留まり・信頼区間推定、family行数による暗黙重み付け",
            "next_evidence": "母集団定義、sampling設計、重み、同定/保持分割、測定不確かさ、欠測規則",
        },
    ]
    model_boundary = [
        {
            "input_model": "複素インピーダンス点/表",
            "electrical_meaning": "指定reference planeでのqualified fundamental one-port R+jX",
            "best_use": "実測/外部解析済み負荷の直接screen、周波数・条件別window",
            "do_not_infer": "プラズマ内部状態、電力分配、未測定条件の補間",
            "minimum_qualification": "周波数、peak/RMS規約、reference plane、fixture、複素位相、出典",
        },
        {
            "input_model": "CCP effective series R-L-C",
            "electrical_meaning": "基本波端子応答を近似するeffective one-port",
            "best_use": "周波数継続、matcher感度、合成回帰fixture",
            "do_not_infer": "シース/イオン/電子/壁への電力内訳、非線形高調波、点火",
            "minimum_qualification": "適用周波数・状態範囲、パラメータ由来、評価面、独立端子Z比較",
        },
        {
            "input_model": "ICP effective transformer",
            "electrical_meaning": "coil側から見た反射負荷を含むeffective one-port",
            "best_use": "coil-feedthrough負荷、matcher感度、論文式conformance",
            "do_not_infer": "密度、衝突、分布アンテナmode、capacitive/inductive power split",
            "minimum_qualification": "coil/secondary parameter由来、周波数範囲、reference plane、端子Z比較",
        },
    ]
    data_grain = [
        {
            "order": 1,
            "grain": "logical question",
            "definition": "他の問いと区別して記述・反証できる一つの検証目的",
            "counting": "レポートの33件",
            "statistical_role": "統計的独立標本ではない",
        },
        {
            "order": 2,
            "grain": "scenario",
            "definition": "設計側がその場で選べない外生条件",
            "counting": "case内のn/N分母",
            "statistical_role": "既定集合。確率は未付与",
        },
        {
            "order": 3,
            "grain": "candidate",
            "definition": "scenarioをまたいで固定するhardware/BOM候補",
            "counting": "全候補を列挙",
            "statistical_role": "標本ではない",
        },
        {
            "order": 4,
            "grain": "control",
            "definition": "各scenarioで許可された可変設定",
            "counting": "scenarioごとに最良許可状態を選択",
            "statistical_role": "制御探索格子。確率ではない",
        },
        {
            "order": 5,
            "grain": "evaluation",
            "definition": "candidate×scenario×controlの一回の回路計算",
            "counting": "コア411ほか",
            "statistical_role": "計算行。独立観測ではない",
        },
        {
            "order": 6,
            "grain": "published row",
            "definition": "文献表から転記した一条件",
            "counting": "source-fidelity内",
            "statistical_role": "原著の設計に従う。自動的に独立とはしない",
        },
        {
            "order": 7,
            "grain": "derived corner",
            "definition": "中心値・読取幅・公差から決定論的に派生した境界",
            "counting": "parent内にネスト",
            "statistical_role": "CI・分布・追加観測ではない",
        },
        {
            "order": 8,
            "grain": "evidence family",
            "definition": "central/spread/common-mode sensitivity等の異質な証拠群",
            "counting": "family別ベクトル",
            "statistical_role": "重みなしで合算しない",
        },
        {
            "order": 9,
            "grain": "runner",
            "definition": "一まとまりの実行入口",
            "counting": (
                f"{sum(bool(row['passed']) for row in literature['execution_regressions'])}/"
                f"{len(literature['execution_regressions'])}実行成功"
            ),
            "statistical_role": "case数・証拠強度ではない",
        },
    ]
    thermal_handoff = [
        {
            "output": "component loss W",
            "correct_meaning": "固定等価直列抵抗へ帰属した平均実電力",
            "usable_for": "下流熱解析の発熱入力候補",
            "not_provided": "温度、hot spot、寿命",
        },
        {
            "output": "network loss W",
            "correct_meaning": "理想源端と負荷面の実電力差",
            "usable_for": "回路損失の電気的閉包",
            "not_provided": "チャンバー全体の熱損失",
        },
        {
            "output": "load real power W",
            "correct_meaning": "宣言した負荷reference planeで受け入れた実電力",
            "usable_for": "電気ポートの電力移送",
            "not_provided": "電子・イオン・壁・ガス・waferへの配分",
        },
        {
            "output": "source apparent VA",
            "correct_meaning": "内部50 Ωを持たない理想源端子のVrms×Irms",
            "usable_for": "同じモデル内の源端負荷比較",
            "not_provided": "generator forward power、発熱量",
        },
        {
            "output": "transfer efficiency",
            "correct_meaning": "load-plane real power / source-terminal real power",
            "usable_for": "宣言二平面間の電気実電力効率",
            "not_provided": "プロセス効率、熱効率",
        },
    ]
    snapshot_provenance = [
        {
            "suite": "core",
            "path": _relative(core_path),
            "schema": core.get("schema", "unknown"),
            "platform_version": core.get("platform_version", "not recorded"),
            "as_of": core.get("generated_at", "unknown"),
            "sha256": _sha256(core_path),
            "status": "PASS" if core.get("passed") else "FAIL",
            "solver": core.get("solver", "not recorded"),
        },
        {
            "suite": "literature",
            "path": _relative(literature_path),
            "schema": literature.get("schema", "unknown"),
            "platform_version": literature.get("platform_version", "not recorded"),
            "as_of": literature.get("generated_at", "unknown"),
            "sha256": _sha256(literature_path),
            "status": "PASS" if literature.get("benchmark_integrity_passed") else "FAIL",
            "solver": "runner別。solver binary versionは統合snapshotに未記録",
        },
        {
            "suite": "B5 candidate detail",
            "path": _relative(b5_candidate_path),
            "schema": b5_candidate.get("schema", "unknown"),
            "platform_version": core.get("platform_version", "not recorded"),
            "as_of": "artifact内に時刻記録なし",
            "sha256": _sha256(b5_candidate_path),
            "status": f"READ ({len(b5_candidate['scenarios'])} scenarios)",
            "solver": "core suiteの選択candidate詳細。solver binary version未記録",
        },
    ]

    sources = [
        {
            "id": "core-suite",
            "label": "Core benchmark definitions and frozen results",
            "path": core_result_path,
            "query": {
                "description": "Core case YAML, frozen expectations, and completed ngspice benchmark result",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    f"core = json.loads(Path({core_result_path!r}).read_text(encoding='utf-8'))\n"
                    "core['cases']"
                ),
                "tables_used": [
                    "bench/cases/*.yaml",
                    "bench/expectations.yaml",
                    core_result_path,
                    _relative(b5_candidate_path),
                ],
                "filters": ["all 16 declared core cases", "complete final suite snapshot"],
                "metric_definitions": [
                    "benchmark PASS = observed classification and frozen invariants reproduced",
                    "raw engineering FEASIBLE = one candidate satisfies every declared scenario and case-local constraint; not apparatus qualification",
                    "coverage = feasible declared scenarios / declared scenarios; n/N is primary and is not probability or yield",
                    "selected control = feasibility-first, then violation, then declared objectives; when a control-margin limit exists, electrical feasibility/violation is ranked before margin violation",
                    "candidate selection = complete feasibility, success coverage, scenario coverage, violation, objectives, then control margin",
                ],
            },
        },
        {
            "id": "literature-suite",
            "label": "Literature benchmark definitions and final evaluation",
            "path": literature_result_path,
            "query": {
                "description": "Published-source fidelity, paper-equation conformance, and platform design challenge results",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    f"literature = json.loads(Path({literature_result_path!r}).read_text(encoding='utf-8'))\n"
                    "(literature['source_fidelity'], literature['model_conformance'], literature['design_challenges'])"
                ),
                "tables_used": [
                    "bench/literature/**",
                    literature_result_path,
                ],
                "filters": ["executable evidence only", "reference-only sources excluded from benchmark counts"],
                "metric_definitions": [
                    "literature benchmark PASS = source/equation reproduction and expected decision classification",
                    "reading corners remain nested within parent conditions",
                    "evidence families are not combined into an implicit weighted score",
                    "Vpp and reported RF powers are literature condition labels unless a platform source amplitude is explicitly declared",
                ],
            },
        },
        {
            "id": "report-method",
            "label": "Logical benchmark case mapping",
            "path": "bench/reports/generate_case_report.py",
            "query": {
                "description": "Report-only mapping that groups test vectors, scenario rows, and candidate rows under a distinct validation question",
                "language": "python",
                "sql": "from bench.reports.generate_case_report import DEFAULT_CORE, DEFAULT_LITERATURE, _load, build_artifact\nartifact = build_artifact(_load(DEFAULT_CORE), _load(DEFAULT_LITERATURE))\nartifact['snapshot']['datasets']",
                "tables_used": ["bench/README.md", "bench/literature/README.md"],
                "metric_definitions": [
                    "logical case = one separately stated, falsifiable benchmark question; the 33 questions are not statistically independent samples",
                    "scenario and candidate rows are evidence inside a logical case unless they change the question",
                    "B5 utilization = observed value / synthetic declared threshold; 1.0 is the case-local boundary",
                ],
            },
        },
        {"id": "lee2021-paper", "label": "Lee, Kwon & Chung 2021", "href": "https://doi.org/10.1063/6.0000883"},
        {"id": "hargis1994-paper", "label": "Hargis et al. 1994", "href": "https://doi.org/10.1063/1.1144770"},
        {"id": "sobolewski1995-paper", "label": "Sobolewski 1995", "href": "https://doi.org/10.6028/jres.100.026"},
        {"id": "lee2020-paper", "label": "Lee et al. 2020", "href": "https://doi.org/10.1063/1.5133862"},
        {"id": "colpo1999-paper", "label": "Colpo, Ernst & Rossi 1999", "href": "https://doi.org/10.1063/1.369268"},
    ]
    for case in core_cases:
        benchmark_id = case["benchmark_id"]
        case_tables = [
            _relative(case["case_path"]),
            core_result_path,
        ]
        if benchmark_id == "B5_high_drive_stress":
            case_tables.append(_relative(b5_candidate_path))
        sources.append(
            {
                "id": f"source-{benchmark_id.lower()}",
                "label": f"{benchmark_id}: definition and frozen result",
                "path": core_result_path,
                "query": {
                    "description": "Select this benchmark case from the frozen core result and pair it with its exact YAML definition",
                    "language": "python",
                    "sql": (
                        "import json\nfrom pathlib import Path\n"
                        f"data = json.loads(Path({core_result_path!r}).read_text(encoding='utf-8'))\n"
                        f"next(row for row in data['cases'] if row['benchmark_id'] == '{benchmark_id}')"
                    ),
                    "tables_used": case_tables,
                    "filters": [f"benchmark_id = {benchmark_id}"],
                },
            }
        )
    for meta in LITERATURE_STATIC:
        result = source_rows.get(meta["result_id"]) or model_rows[meta["result_id"]]
        sources.append(
            {
                "id": f"source-{meta['id'].lower()}",
                "label": f"{meta['id']}: source/model conformance evidence",
                "path": _relative(result["result"]),
                "query": {
                    "description": "Select the exact source/model conformance row and its executable evidence",
                    "language": "python",
                    "sql": (
                        "import json\nfrom pathlib import Path\n"
                        f"data = json.loads(Path({literature_result_path!r}).read_text(encoding='utf-8'))\n"
                        f"next(row for row in data['source_fidelity'] + data['model_conformance'] if row['id'] == '{meta['result_id']}')"
                    ),
                    "tables_used": [STATIC_SOURCE_PATHS[meta["id"]], _relative(result["result"])],
                    "filters": [f"result id = {meta['result_id']}"],
                },
            }
        )
    for result_id in focused_design_ids:
        meta = DESIGN_META[result_id]
        sources.append(
            {
                "id": f"source-{meta['report_id'].lower()}",
                "label": f"{meta['report_id']}: design definition and result",
                "path": literature_result_path,
                "query": {
                    "description": "Select this literature-derived design challenge from the integrated evaluation",
                    "language": "python",
                    "sql": (
                        "import json\nfrom pathlib import Path\n"
                        f"data = json.loads(Path({literature_result_path!r}).read_text(encoding='utf-8'))\n"
                        f"next(row for row in data['design_challenges'] if row['id'] == '{result_id}')"
                    ),
                    "tables_used": [DESIGN_SOURCE_PATHS[result_id], literature_result_path],
                    "filters": [f"design challenge id = {result_id}"],
                },
            }
        )
    sources.append(
        {
            "id": "source-gec-d7",
            "label": "GEC-D7: hardware candidate by evidence family",
            "path": literature_result_path,
            "query": {
                "description": "Select all 12 candidate-family rows without combining family weights",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    f"data = json.loads(Path({literature_result_path!r}).read_text(encoding='utf-8'))\n"
                    "[row for row in data['design_challenges'] if '__L1_' in row['id']]"
                ),
                "tables_used": [
                    "bench/literature/p1_gec_ccp/hardware_family_spec.yaml",
                    literature_result_path,
                ],
                "filters": ["four evidence families retained separately", "three L1 candidates"],
            },
        }
    )
    source_by_id = {source["id"]: source for source in sources}

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {TITLE}"},
        {
            "id": "technical-summary",
            "type": "markdown",
            "layout": "full",
            "body": f"""## 技術要約 — 33件の異なる検証質問を、装置資格と分離して読む

現行ベンチマークは、**33件の論理上区別した検証質問**として整理できる。これらは統計的に独立な33標本ではない。今回のsnapshotでは**{reproduced_questions}/33件が期待結果を再現**した。再現PASSはソフトウェア・回路式・データ変換・期待分類の回帰を意味し、対象装置が成立したことを意味しない。

コア16件のraw `FEASIBLE`は{len(core_met_ids)}件だが、内訳は同質でない。conformance fixtureで期待分類が成立したのは`{"/".join(conformance_met_ids) if conformance_met_ids else "なし"}`、設計workflowでcase-local criteriaを満たしたのは`{"/".join(workflow_met_ids) if workflow_met_ids else "なし"}`である。これらを装置性能件数や成功率として集計しない。期待負例がcase-local criteriaを満たさないことも正しい回帰結果である。

文献側17件は、公開表・式・図読取の忠実度と、文献由来負荷をplatform側fixtureへ接続した設計challengeを分離する。Scenario行、Lee 2020の6代数vector、Colpoの60読取角、Hargisの12 candidate-family結果を独立物理観測として水増ししない。

**結論:** 本基盤は、負荷を定義する評価面で与えた基本波複素負荷に対し、同じ整合回路が複数負荷を覆うか、可変範囲で追従できるか、与えた振幅・等価損失で電気閾値を超えないかを比較する回路スクリーニング基盤である。特定chamber、process、generator、実部品、熱、寿命、制御安定性の資格判定は**未実施**である。
""",
            "sourceId": "report-method",
        },
        {
            "id": "headline-metrics",
            "type": "metric-strip",
            "layout": "full",
            "cardIds": ["logical-cases", "reproduced-questions", "literature-questions"],
        },
        {
            "id": "platform-position",
            "type": "markdown",
            "layout": "full",
            "body": """## 何を入力し、何を返す基盤か

最小入力は、(1) 周波数と**負荷を定義する評価面（reference plane）**でのR+jX、または**端子から見た有効一端子負荷（effective one-port）**、(2) **固定回路候補（Candidate）**、(3) **設計側で選べない外部条件（Scenario）**、(4) 必要な場合だけ**条件ごとの調整値（Control）**、(5) ケース内合格条件である。絶対電気ストレスを評価するときだけ、信号源振幅、固定等価直列抵抗、制約値を追加する。

処理は決定論的である。固定回路候補×外部条件×調整値を完全列挙し、各外部条件で許可された調整値を選び、その選択後の入力インピーダンスZin、反射係数|Γ|、制約違反を求める。固定回路候補は全外部条件の成立性を優先して比較する。出力は、条件別選択状態、合格数/総数、最悪|Γ|、電圧・電流・電力などの電気量、根拠artifactである。プラズマ状態や温度場を内部で推定するsolverではない。
""",
            "sourceId": "report-method",
        },
        {
            "id": "reading-contract",
            "type": "markdown",
            "layout": "full",
            "body": """## 読み方 — 役割、反射、集約、二つの合否

- **Candidate（固定回路候補）:** Scenarioが変わっても交換しない固定回路・ハードウェア候補。
- **Scenario（外部条件）:** 周波数、負荷、振幅、実現部品値など、設計側がその場で選べない条件。
- **Control（調整値）:** 各Scenarioで変更できるチューナ設定。
- **Control margin（離散格子内の幾何余裕）:** 数値軸ごとに `m=2×min(u−u_min, u_max−u)/(u_max−u_min)` とし、多軸の最小値を使う。端=0、範囲中央=1、単一値軸=1、categorical/bool軸は無視する。実actuatorの機械余裕、連続可動域、追従余裕ではない。
- **Benchmark reproduction（回帰再現）:** 実装が凍結期待値、独立oracle、列挙数、不変条件、期待分類を再現したか。
- **Case-local declared criteria（ケース内条件）:** Candidateが全Scenarioで、許されたControlを使い、そのケースで宣言した全制約を満たしたか。raw artifactのFEASIBLE/INFEASIBLEをMET/NOT METと表示する。実機成立性ではない。
- **Coverage:** 宣言したScenario集合内の合格数/総数をn/Nで示す。分母の異なるケース間で割合を順位付けせず、確率・稼働率・歩留まりにも変換しない。
- **Reference plane（負荷評価面）:** 複素負荷Zを定義する電気的境界。fixtureは明示または埋込みのどちらか一度だけ扱う。

共通反射指標は `Γ=(Zin−50 Ω)/(Zin+50 Ω)`、反射不整合率は `|Γ|²` である。50 Ωはmatcher上流/source評価面の計算基準で、回路図に置かれた直列抵抗ではない。directional couplerやgeneratorのforward/reflected powerを直接再現した値でもない。共通基準 `|Γ|≤√0.1=0.316228` はplatformの工学基準で、文献や全装置の普遍基準ではない。

Control選択は単純な最小|Γ|とは限らない。各Scenarioで、まずケース内制約の成立性、次に違反量、最後に宣言した目的値で選ぶ。Control-margin制約がある場合は、電気的成立性・電気違反、margin違反の順で優先する。margin制約のない文献designでは目的値が|Γ|なので最小|Γ|状態となる。選択後に`worst |Γ| = max_s |Γ_selected,s|`と全制約を集約し、Candidateは全Scenarioの成立性を優先して比較する。Control格子端まで到達できることと、実運用reserveを持つことは別である。
""",
        },
        {
            "id": "decision-ladder-intro",
            "type": "markdown",
            "layout": "full",
            "body": """## 判定の4段階 — 下段を上段のPASSから推論しない

回帰再現、静的到達性、電気スクリーニング、実機資格は別の問いである。現行証拠がどこまで届くかを先に固定する。""",
            "sourceId": "report-method",
        },
        {"id": "decision-ladder-table-block", "type": "table", "layout": "full", "tableId": "decision-ladder-table"},
        {
            "id": "practitioner-guide-intro",
            "type": "markdown",
            "layout": "full",
            "body": """## 三分野の利用判断 — 使える出力と次工程を分ける

各専門分野は同じPASSを別の承認として読まない。現在利用できる判断、未対応領域、次に必要な証拠を以下にまとめる。""",
        },
        {
            "id": "practitioner-guide-table-block",
            "type": "table",
            "layout": "full",
            "tableId": "practitioner-guide-table",
        },
        {
            "id": "model-boundary-intro",
            "type": "markdown",
            "layout": "full",
            "body": """## 負荷モデルの使い分け — 端子一ポートの責務に限定する

最も直接的な入力は、評価面が明示された複素インピーダンス点または表である。CCP/ICPモデルは、厳密なプラズマ内部解析ではなく、端子応答を説明・継続するためのeffective one-portとして使う。""",
        },
        {"id": "model-boundary-table-block", "type": "table", "layout": "full", "tableId": "model-boundary-table"},
        {
            "id": "inventory-intro",
            "type": "markdown",
            "layout": "full",
            "body": """## 33ケースの役割分担 — 行数ではなく異なる検証質問で数える

下図はレポート内のlogical question数を目的別に示すinventoryである。棒の長さは重要度、成熟度、証拠強度、統計標本数ではない。scenario、candidate、controlの行数を加えず、問いが変わる単位だけを数えた。
""",
            "sourceId": "report-method",
        },
        {"id": "inventory-chart-block", "type": "chart", "layout": "full", "chartId": "case-inventory-chart"},
        {
            "id": "core-index-intro",
            "type": "markdown",
            "layout": "full",
            "body": f"""## コア16ケース一覧 — 回帰結果とcase-local criteriaを分離する

今回の凍結期待結果を再現したのは{core_reproduced}/16件である。raw FEASIBLE {len(core_met_ids)}件は同質な装置性能ではなく、conformance fixture {len(conformance_met_ids)}件とdesign-workflow fixture {len(workflow_met_ids)}件である。表のcoverageはn/Nを主表示し、評価数はCandidate×Scenario×Controlの計算行数であって証拠の重みではない。以下の個別節は、この表や前ケースを読まなくても問題設定が分かるよう条件を再掲する。
""",
            "sourceId": "core-suite",
        },
        {"id": "core-index-table-block", "type": "table", "layout": "full", "tableId": "core-index-table"},
        {
            "id": "b5-stress-intro",
            "type": "markdown",
            "layout": "full",
            "body": f"""## B5の横断可視化 — 整合、合成電気screen、数値閉包を混ぜない

下図は4指標の`計算値/合成閾値`で、1.0がcase-local境界である。13.56 MHzの固定線形AC、理想Vsrc端子25/100 Vpeakという2条件だけを比較する。閾値は部品データシート由来ではなく、温度・寿命予測でもない。振幅4倍に対して電流は4倍、固定抵抗損失とVAは16倍となる。今回の100 Vpeak条件では整合が良好なまま{b5_high_fail_count}/4指標が閾値を超えた。""",
            "sourceId": "source-b5_high_drive_stress",
        },
        {"id": "b5-stress-chart-block", "type": "chart", "layout": "full", "chartId": "b5-stress-chart"},
        {"id": "thermal-handoff-table-block", "type": "table", "layout": "full", "tableId": "thermal-handoff-table"},
    ]
    blocks.extend(
        _core_block(
            case,
            b5_candidate if case["benchmark_id"] == "B5_high_drive_stress" else None,
            core_result_path=core_result_path,
        )
        for case in core_cases
    )
    blocks.extend(
        [
            {
                "id": "literature-intro",
                "type": "markdown",
                "layout": "full",
                "body": """## 文献17ケース — 出典忠実度、式適合、platform設計challengeを分離する

文献値そのものを確認するケースではcase-local design criteriaをN/Aとする。文献由来負荷をplatform matcherへ接続する設計ケースでは、論文由来の負荷・条件ラベルと、platform側で追加した回路・Control格子・10%反射基準を明確に分ける。これらは論文装置matcherの再現や装置資格ではない。
""",
                "sourceId": "literature-suite",
            },
            {
                "id": "literature-check-table-block",
                "type": "table",
                "layout": "full",
                "tableId": "literature-check-table",
            },
            {
                "id": "literature-design-index-intro",
                "type": "markdown",
                "layout": "full",
                "body": """## 文献設計ケースの数値索引 — n/Nはケース内だけで解釈する

この索引は、Leeの基準面感度、GECのfocused control cases、Colpoのcorner challengesを横断して数値を引くためのもの。分母と証拠生成規則が異なるためcoverage比をケース間で順位付けしない。GEC-D7は4family×3候補の複合比較なので単一coverageを定義せず、個別節のfamily別ベクトルを正とする。Controlを持つ文献ケースは静的grid到達性であり、reserveは未資格である。
""",
                "sourceId": "literature-suite",
            },
            {
                "id": "literature-design-table-block",
                "type": "table",
                "layout": "full",
                "tableId": "literature-design-table",
            },
        ]
    )

    static_by_id = {item["id"]: item for item in LITERATURE_STATIC}
    blocks.append(_literature_static_block(static_by_id["LEE21-T1"], source_rows, model_rows))
    for result_id in (
        "literature_p0_lee2021_matcher_output_probe",
        "literature_p0_lee2021_post_coax_plane_sensitivity",
        "literature_p0_lee2021_plasma_terminal_plane_sensitivity",
    ):
        blocks.append(_design_block(design_by_id[result_id], literature_result_path=literature_result_path))
    blocks.append(_literature_static_block(static_by_id["GEC-S1"], source_rows, model_rows))
    for result_id in (
        "P1_CCP_fixed_central",
        "P1_CCP_limited_central",
        "P1_CCP_full_central",
        "P1_CCP_reported_spread",
        "P1_CCP_phase_minus6",
        "P1_CCP_phase_plus6",
    ):
        blocks.append(_design_block(design_by_id[result_id], literature_result_path=literature_result_path))
    blocks.append(_hardware_family_block(hardware_rows, literature_result_path=literature_result_path))
    blocks.append(_literature_static_block(static_by_id["LEE20-M1"], source_rows, model_rows))
    blocks.append(_literature_static_block(static_by_id["COLPO-F1"], source_rows, model_rows))
    colpo_block = _literature_static_block(static_by_id["COLPO-S1"], source_rows, model_rows)
    colpo_block["body"] += (
        f"\n**派生規則の追加確認**  \nBenchmark reproduction: "
        f"**{'PASS' if colpo_derivation['passed'] else 'FAIL'}**。{colpo_derivation['observed']}。\n"
    )
    blocks.append(colpo_block)
    for result_id in (
        "literature_colpo1999_fixed_digitization_corners",
        "literature_colpo1999_bounded_digitization_corners",
    ):
        blocks.append(_design_block(design_by_id[result_id], literature_result_path=literature_result_path))
    blocks.extend(
        [
            {
                "id": "data-grain-intro",
                "type": "markdown",
                "layout": "full",
                "body": """## データ粒度辞書 — 計算行を観測標本として数えない

本レポートは推測統計を行わない。B8は決定論的全因子角点、Hargis spreadは決定論的包絡、±6°はcommon-modeモデル感度、Colpo角はparent内の読取幅、Lee 2020は合成代数vectorである。""",
                "sourceId": "report-method",
            },
            {"id": "data-grain-table-block", "type": "table", "layout": "full", "tableId": "data-grain-table"},
            {
                "id": "snapshot-provenance-intro",
                "type": "markdown",
                "layout": "full",
                "body": """## 評価snapshot — 時刻・schema・内容hashを固定する

以下のSHA-256は、このレポートが読んだcore統合結果、文献統合結果、B5選択candidate詳細そのものに対する値である。基盤versionは統合結果から記録する。solver binaryのversionやB5 artifact時刻は記録されていないため、推測して補わない。""",
                "sourceId": "report-method",
            },
            {
                "id": "snapshot-provenance-table-block",
                "type": "table",
                "layout": "full",
                "tableId": "snapshot-provenance-table",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": """## 立証範囲 — 完了している電気的責務と、意図的に未資格の領域

**完了している責務**は、公開トポロジ配線、CCP/ICP有効端子式、有限Candidate/Control列挙、参照面の一回表現、文献表・式・限定図読取、宣言されたScenario集合でのworst-case設計判断である。

**資格付けしていない領域**は、特定production chamber、工程性能、プラズマ密度・シース・化学、非線形高調波、点火・settling、分布アンテナモード、部品温度・寿命、実制御器の安定性である。これらを既存ケースのPASSから推論してはならない。

**ロバスト性上の注意:** B8の8隅は歩留まりではない。Hargisの4 evidence familyは重み付け合算しない。±6°は共通モデル感度であり確率区間ではない。Colpoの60角は15 parent内の読取幅である。D1-D3は個別実行できるが、等価・二重計上の主張は比較不変条件による。
""",
            },
            {
                "id": "reference-only-intro",
                "type": "markdown",
                "layout": "full",
                "body": """## 実行ケースにしなかった文献 — 不足証拠をgolden値へ変換しない

次の文献は有用な背景だが、位相付き複素端子Z、責任境界、または現コードの定常一ポート責務に必要な情報が不足する。未実装残件ではなく、現在の証拠では厳密goldenにしないという設計判断である。
""",
                "sourceId": "literature-suite",
            },
            {"id": "reference-only-table-block", "type": "table", "layout": "full", "tableId": "reference-only-table"},
            {
                "id": "next-steps",
                "type": "markdown",
                "layout": "full",
                "body": """## 推奨する次の利用手順 — 目的に対応するケースだけを選ぶ

1. 回路実装変更ではA1-A5とD1-D3を回帰させる。
2. Candidate/Scenario/Control処理変更ではB1-B8を回帰させる。
3. 文献データ取込み変更ではsource-fidelity casesを先に通し、その後に設計challengeを実行する。
4. 新しい実機対象が決まった場合は、校正済み・位相付き複素Z、周波数、振幅、reference plane、fixture transform、実matcher範囲、部品定格を一つの装置資格ケースとして追加する。
5. 既存の合成・文献ケースを一つの「普遍負荷窓」へ結合しない。
""",
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "layout": "full",
                "body": """## 次に答えるべき問い — 実機資格へ進むときだけ追加する

- 対象装置の設計基準面はgenerator、matcher output、electrode、coil feedthroughのどこか。
- 保持データは同定用と独立評価用に分離できるか。
- source amplitudeはideal source node、forward wave、VI-probeのどれとして定義されるか。
- 反射以外に、電圧、電流、損失、熱、絶縁、tuner reserveのどれを必須制約とするか。
- 定常ACを越えて過渡、高調波、pulsed matchingを本基盤の責務へ含める必要があるか。
""",
            },
        ]
    )

    case_block_ids = [block["id"] for block in blocks if block["id"].startswith("case-")]
    if len(case_block_ids) != 33 or len(set(case_block_ids)) != 33:
        raise AssertionError("Report must contain exactly 33 unique logical case blocks")

    cards = [
        {
            "id": "logical-cases",
            "dataset": "summary",
            "source": _widget_source(source_by_id["report-method"], "summary"),
            "description": "Scenario行や候補行を重複計上しない異なる検証質問。統計的独立標本ではない",
            "metrics": [{"label": "検証質問", "field": "logical_cases", "format": "number"}],
        },
        {
            "id": "reproduced-questions",
            "dataset": "summary",
            "source": _widget_source(source_by_id["report-method"], "summary"),
            "description": "凍結期待結果を再現したlogical question。装置資格ではない（33件中）",
            "metrics": [{"label": "回帰再現", "field": "reproduced_questions", "format": "number"}],
        },
        {
            "id": "literature-questions",
            "dataset": "summary",
            "source": _widget_source(source_by_id["literature-suite"], "summary"),
            "description": "公開値・式の再現と文献由来負荷のplatform設計challengeを分けた質問数",
            "metrics": [{"label": "文献由来の検証質問", "field": "literature_logical_cases", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "case-inventory-chart",
            "title": "目的別の論理ベンチマークケース数",
            "subtitle": "合計33件の異なる検証質問。統計的独立標本・証拠強度・成熟度ではない",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "Which distinct validation questions are covered by the current benchmark inventory?",
            "rationale": "Seven categories with long labels are most legible as horizontal bars; exact definitions remain in the adjacent narrative and source data.",
            "dataset": "case_inventory",
            "source": _widget_source(source_by_id["report-method"], "case_inventory"),
            "encodings": {
                "x": {"field": "category", "type": "nominal", "label": "目的"},
                "y": {"field": "case_count", "type": "quantitative", "label": "ケース数", "format": "number"},
                "tooltip": [
                    {"field": "scope", "type": "text", "label": "対象"},
                    {"field": "evidence_kind", "type": "text", "label": "証拠種別"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "all"},
            "settings": {"sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": {"surface": "card", "viewMode": "both", "showControls": False},
        },
        {
            "id": "b5-stress-chart",
            "title": "B5 合成電気スクリーニング閾値の利用率",
            "subtitle": "計算値/合成閾値。1.0がcase-local境界；13.56 MHz固定線形AC、温度・寿命予測ではない",
            "type": "bar",
            "intent": "comparison",
            "question": "Which declared synthetic electrical limits are exceeded when ideal-source amplitude rises from 25 to 100 Vpeak?",
            "rationale": "A grouped bar chart preserves the four unlike metrics only after normalization by their own declared thresholds and keeps the two deterministic drive scenarios separate.",
            "comparisonContext": {
                "baseline": "25 Vpeak",
                "denominator": "metric-specific synthetic declared threshold",
                "grain": "metric × deterministic source-amplitude scenario",
                "normalization": "observed value / declared threshold",
                "semanticFamily": "case-local electrical screening",
                "unit": "ratio",
            },
            "dataset": "b5_stress",
            "source": _widget_source(source_by_id["source-b5_high_drive_stress"], "b5_stress"),
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "電気指標"},
                "y": {"field": "utilization", "type": "quantitative", "label": "計算値 / 合成閾値", "format": "number"},
                "color": {"field": "scenario", "type": "nominal", "label": "理想源端子振幅"},
                "tooltip": [
                    {"field": "value", "type": "quantitative", "label": "計算値"},
                    {"field": "limit", "type": "quantitative", "label": "合成閾値"},
                    {"field": "unit", "type": "text", "label": "単位"},
                    {"field": "screen", "type": "text", "label": "case-local判定"},
                    {"field": "load_real_power_W", "type": "quantitative", "label": "負荷面受入実電力 W"},
                    {"field": "network_loss_W", "type": "quantitative", "label": "回路網等価損失 W"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "drive-scenarios"},
            "legend": {"position": "bottom", "sort": "spec", "title": "理想Vsrc端子"},
            "labels": {"values": "all"},
            "referenceLines": [
                {"axis": "y", "color": "red", "label": "合成閾値 1.0", "lineStyle": "dashed", "value": 1.0}
            ],
            "settings": {"groupMode": "grouped", "showValues": True, "categoryLabelPolicy": "wrap"},
            "surface": {"surface": "card", "viewMode": "both", "showControls": False},
        },
    ]

    tables = [
        {
            "id": "decision-ladder-table",
            "title": "判定段階と現行証拠",
            "subtitle": "上段のPASSは下段の資格を代替しない",
            "dataset": "decision_ladder",
            "source": _widget_source(source_by_id["report-method"], "decision_ladder"),
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "stage", "direction": "asc"},
            "columns": [
                {"field": "stage", "label": "段階", "type": "text"},
                {"field": "question", "label": "問うこと", "type": "text"},
                {"field": "current_evidence", "label": "現行証拠", "type": "text"},
                {"field": "status", "label": "状態", "type": "text"},
                {"field": "meaning", "label": "正しい意味", "type": "text"},
            ],
        },
        {
            "id": "practitioner-guide-table",
            "title": "専門分野別の利用境界",
            "subtitle": "現在使える判断、未対応領域、次に必要な証拠",
            "dataset": "practitioner_guide",
            "source": _widget_source(source_by_id["report-method"], "practitioner_guide"),
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "reader", "direction": "asc"},
            "columns": [
                {"field": "reader", "label": "利用者", "type": "text"},
                {"field": "usable_now", "label": "現在利用できる判断", "type": "text"},
                {"field": "not_supported", "label": "未対応・推論不可", "type": "text"},
                {"field": "next_evidence", "label": "次工程に必要な証拠", "type": "text"},
            ],
        },
        {
            "id": "model-boundary-table",
            "title": "負荷入力モデルの責務と最小qualification",
            "subtitle": "プラズマ内部solverではなく、指定面のeffective one-portとして利用",
            "dataset": "model_boundary",
            "source": _widget_source(source_by_id["report-method"], "model_boundary"),
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "input_model", "direction": "asc"},
            "columns": [
                {"field": "input_model", "label": "入力モデル", "type": "text"},
                {"field": "electrical_meaning", "label": "電気的意味", "type": "text"},
                {"field": "best_use", "label": "適する用途", "type": "text"},
                {"field": "do_not_infer", "label": "推論しないもの", "type": "text"},
                {"field": "minimum_qualification", "label": "最小qualification情報", "type": "text"},
            ],
        },
        {
            "id": "snapshot-provenance-table",
            "title": "入力snapshot provenance",
            "subtitle": "基盤version、as-of、ファイル内容hash。solver binary版は推測しない",
            "dataset": "snapshot_provenance",
            "source": _widget_source(source_by_id["report-method"], "snapshot_provenance"),
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "suite", "direction": "asc"},
            "columns": [
                {"field": "suite", "label": "Suite", "type": "text"},
                {"field": "path", "label": "入力artifact", "type": "text"},
                {"field": "schema", "label": "Schema", "type": "text"},
                {"field": "platform_version", "label": "PCD version", "type": "text"},
                {"field": "as_of", "label": "Snapshot時刻", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
                {"field": "status", "label": "結果", "type": "text"},
                {"field": "solver", "label": "Solver記録", "type": "text"},
            ],
        },
        {
            "id": "data-grain-table",
            "title": "データ粒度と統計的意味",
            "subtitle": "本レポートは決定論的検証で、推測統計を実施しない",
            "dataset": "data_grain",
            "source": _widget_source(source_by_id["report-method"], "data_grain"),
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "順", "type": "number"},
                {"field": "grain", "label": "粒度", "type": "text"},
                {"field": "definition", "label": "定義", "type": "text"},
                {"field": "counting", "label": "集計方法", "type": "text"},
                {"field": "statistical_role", "label": "統計的意味", "type": "text"},
            ],
        },
        {
            "id": "core-index-table",
            "title": "コア16ケースの結果索引",
            "subtitle": "Coverageはcase内n/N、worst |Γ|はscenario別Control選択後の最悪点",
            "dataset": "core_index",
            "source": _widget_source(source_by_id["core-suite"], "core_index"),
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "case_id", "direction": "asc"},
            "columns": [
                {"field": "case_id", "label": "ID", "type": "text"},
                {"field": "case_name", "label": "ケース", "type": "text"},
                {"field": "benchmark", "label": "回帰再現", "type": "text"},
                {"field": "declared_criteria", "label": "ケース内条件", "type": "text"},
                {"field": "evidence_group", "label": "Evidence / 役割", "type": "text"},
                {"field": "reference_plane", "label": "負荷面", "type": "text"},
                {"field": "scenario_count", "label": "Scenario", "type": "number"},
                {"field": "candidate_count", "label": "Candidate", "type": "number"},
                {"field": "evaluation_count", "label": "評価数", "type": "number"},
                {"field": "coverage", "label": "Coverage n/N", "type": "text"},
                {"field": "worst_gamma", "label": "worst |Γ|", "type": "number"},
                {"field": "limiting_constraint", "label": "制約違反", "type": "text"},
            ],
        },
        {
            "id": "literature-check-table",
            "title": "文献source/model conformanceケース",
            "subtitle": "設計可否ではなく、公開証拠の再現可能性を評価",
            "dataset": "literature_checks",
            "source": _widget_source(source_by_id["literature-suite"], "literature_checks"),
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "case_id", "direction": "asc"},
            "columns": [
                {"field": "case_id", "label": "ID", "type": "text"},
                {"field": "case_name", "label": "ケース", "type": "text"},
                {"field": "benchmark", "label": "回帰再現", "type": "text"},
                {"field": "declared_criteria", "label": "ケース内条件", "type": "text"},
                {"field": "purpose", "label": "目的", "type": "text"},
                {"field": "observed", "label": "観測結果", "type": "text"},
            ],
        },
        {
            "id": "literature-design-table",
            "title": "文献由来負荷に対する設計ケース索引",
            "subtitle": "n/Nはcase内だけで解釈し、GEC-D7の4familyは単一coverageへ合算しない",
            "dataset": "literature_design_index",
            "source": _widget_source(source_by_id["literature-suite"], "literature_design_index"),
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "case_id", "direction": "asc"},
            "columns": [
                {"field": "case_id", "label": "ID", "type": "text"},
                {"field": "case_name", "label": "ケース", "type": "text"},
                {"field": "benchmark", "label": "回帰再現", "type": "text"},
                {"field": "declared_criteria", "label": "ケース内条件", "type": "text"},
                {"field": "evidence_group", "label": "Evidence", "type": "text"},
                {"field": "scenario_scope", "label": "Scenario範囲", "type": "text"},
                {"field": "evaluation_count", "label": "評価数", "type": "number"},
                {"field": "coverage", "label": "Coverage n/N", "type": "text"},
                {"field": "worst_gamma", "label": "worst |Γ|", "type": "number"},
                {"field": "reserve_status", "label": "Control reserve", "type": "text"},
            ],
        },
        {
            "id": "thermal-handoff-table",
            "title": "電気出力から伝熱・部品設計への受け渡し",
            "subtitle": "電気ポート量を温度・process powerへ読み替えない",
            "dataset": "thermal_handoff",
            "source": _widget_source(source_by_id["source-b5_high_drive_stress"], "thermal_handoff"),
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "output", "direction": "asc"},
            "columns": [
                {"field": "output", "label": "出力", "type": "text"},
                {"field": "correct_meaning", "label": "正しい意味", "type": "text"},
                {"field": "usable_for", "label": "利用可能", "type": "text"},
                {"field": "not_provided", "label": "ここでは得られないもの", "type": "text"},
            ],
        },
        {
            "id": "reference-only-table",
            "title": "Reference-only文献",
            "subtitle": "現在の証拠または責任境界では厳密な実行goldenにしない項目",
            "dataset": "reference_only",
            "source": _widget_source(source_by_id["literature-suite"], "reference_only"),
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "source", "direction": "asc"},
            "columns": [
                {"field": "source", "label": "文献", "type": "text"},
                {"field": "status", "label": "状態", "type": "text"},
                {"field": "reason", "label": "理由", "type": "text"},
            ],
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "半導体製造、伝熱、データ分析の第三者が、回路benchmarkの問題設定、結果、証拠粒度、利用境界を独立に判断するための技術レポート",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "summary": summary,
            "decision_ladder": decision_ladder,
            "practitioner_guide": practitioner_guide,
            "model_boundary": model_boundary,
            "snapshot_provenance": snapshot_provenance,
            "data_grain": data_grain,
            "case_inventory": inventory,
            "core_index": core_index,
            "b5_stress": b5_stress,
            "thermal_handoff": thermal_handoff,
            "literature_checks": literature_checks,
            "literature_design_index": literature_design_index,
            "reference_only": literature["reference_inventory"],
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-result", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--literature-result", type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument(
        "--b5-candidate-result",
        type=Path,
        help="optional B5 candidate detail; auto-discovered from --core-result when omitted",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    core = _load(args.core_result)
    literature = _load(args.literature_result)

    artifact = build_artifact(
        core,
        literature,
        core_path=args.core_result,
        literature_path=args.literature_result,
        b5_candidate_path=args.b5_candidate_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "logical_cases": 33,
                "blocks": len(artifact["manifest"]["blocks"]),
                "datasets": {key: len(value) for key, value in artifact["snapshot"]["datasets"].items()},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
