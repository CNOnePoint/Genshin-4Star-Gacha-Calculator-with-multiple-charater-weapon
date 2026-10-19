# -*- coding: utf-8 -*-
r"""
原神武器池「定轨 / 命定值」抽取策略最优化求解器
================================================

问题
----
武器池每出一金都在 UP 武器 A、B 和「非 UP 武器」之间三选一：

    P(A) = 37.5%    P(B) = 37.5%    P(非UP) = 25%

另有两条保底规则：

1. 大保底 g：出非 UP 武器后，下一金必为 UP 武器（A / B 各 50%）。
2. 定轨命定值 f：玩家可随时把 A 或 B 设为「定轨」目标。
   * 出金 ≠ 定轨目标（非 UP 或另一把 UP）→ f += 1
   * 出金 = 定轨目标                    → f 归 0
   * 取消 / 切换定轨                     → f 归 0（大保底 g 不受影响）
   * f 达到上限 F 时，下一金必为定轨目标

目标：凑齐 a 把 A 和 b 把 B，求**期望抽数最小**的策略。

建模
----
状态   (a_还差, b_还差, g, f, s)        s ∈ {A, B, 不定轨}
动作   定轨 A / 定轨 B / 不定轨
求解   随机最短路径的 Bellman 最优方程。

同一个 (a, b) 块内存在自环（出了非 UP、或出了已经不需要的武器），
所以块内要解一个小型方程组；块之间按 a+b 递增天然可解。

「不定轨」这个动作：当 f>0 且 g=1 时，取消定轨能把下一金从
「100% 定轨目标」换成「100% UP，A/B 各 50%」。理论上它看起来有用，
但**计算结果表明它永远不是严格最优**——因为「切换定轨到另一把」给出完全
相同的分布，却额外把命定值记到新目标头上。见 --check-none。

命令
----
    python 定轨最优抽取策略.py                 # 用户口径 F=1
    python 定轨最优抽取策略.py --real           # 官方真实口径 F=2
    python 定轨最优抽取策略.py --validate       # 蒙特卡洛校验
    python 定轨最优抽取策略.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys

P_A, P_B, P_OFF = 0.375, 0.375, 0.25
NAMES = {"A": "武器A", "B": "武器B", None: "不定轨"}
ACT_NAME = {"A": "定轨A", "B": "定轨B", None: "不定轨"}


# ==========================================================================
# 1. 模型
# ==========================================================================
class Model:
    """一套抽卡规则。

    F            命定值上限；f == F 时下一金必为定轨目标。
                 F = 1 → 用户描述口径（歪 1 次即保底）
                 F = 2 → 官方真实口径（命定值 0/1/2，满 2 才保底）
    G            是否存在「出非 UP 后下一金必为 UP」的大保底
    inc          'non_selected' 出金 ≠ 定轨目标就 +1（含另一把 UP）
                 'non_up'       只有出非 UP 才 +1
    switch_resets 切换定轨目标是否把命定值清零（官方为 True）
    """

    def __init__(self, F: int = 1, G: bool = True,
                 inc: str = "non_selected", switch_resets: bool = True):
        assert F >= 0
        assert inc in ("non_selected", "non_up")
        self.F = F
        self.G = G
        self.inc = inc
        self.switch_resets = switch_resets

    # -- 状态 ------------------------------------------------------------
    def canon(self, g: int, f: int, s):
        """不定轨时命定值无意义，一律归 0。"""
        return (g, 0 if s is None else f, s)

    def states(self):
        out = []
        for g in (0, 1):
            out.append((g, 0, None))
            for s in ("A", "B"):
                for f in range(0, self.F + 1):
                    out.append((g, f, s))
        return out

    def apply_action(self, f: int, s, act):
        """执行动作后（还没抽）命定值的取值。"""
        if act == s:
            return f
        if act is None:
            return 0
        return 0 if self.switch_resets else f

    # -- 出金分布 --------------------------------------------------------
    def distribution(self, g: int, f2: int, s2):
        if s2 is not None and self.F > 0 and f2 >= self.F:
            return ((s2, 1.0),)                       # 命定值满 → 必为定轨目标
        if self.G and g == 1:
            return (("A", 0.5), ("B", 0.5))           # 大保底 → UP，A/B 均分
        return (("A", P_A), ("B", P_B), ("OFF", P_OFF))

    def step_state(self, g: int, f2: int, s2, w: str):
        """出金结果 w 之后的状态。"""
        if w == "OFF":
            ng = 1 if self.G else 0
            nf = min(f2 + 1, self.F) if self.F > 0 else 0
        elif w == s2:
            ng, nf = 0, 0
        else:                                          # 出了另一把 UP
            ng = 0
            nf = (min(f2 + 1, self.F) if self.F > 0
                  and self.inc == "non_selected" else 0)
        return self.canon(ng, nf, s2)

    # -- 单步转移（含落地状态） -------------------------------------------
    def transitions(self, a: int, b: int, st, act):
        """返回 [(概率, 是否前进, 落地块内状态)]。"""
        g, f, s = st
        f2 = self.apply_action(f, s, act)
        s2 = act
        out = []
        for w, p in self.distribution(g, f2, s2):
            if p == 0.0:
                continue
            if w == "A":
                na, nb = (a - 1 if a > 0 else 0), b
            elif w == "B":
                na, nb = a, (b - 1 if b > 0 else 0)
            else:
                na, nb = a, b
            nxt = self.step_state(g, f2, s2, w)
            out.append((p, (na, nb) != (a, b), (na, nb) + nxt))
        return out


# ==========================================================================
# 2. 精确求解（最优）与策略评估（给定策略）
# ==========================================================================
def _block_order(maxa, maxb):
    return sorted(((a, b) for a in range(maxa + 1) for b in range(maxb + 1)),
                  key=lambda t: (t[0] + t[1], t[0]))


def _is_dead(model, a, b, st, act):
    """该动作是否**必然**原地打转：全部概率质量落在「没前进且状态不变」上。

    注意不能简单判「没有前进概率」——例如 a=0 时被命定值逼着出一把 A，
    这一次抽数被浪费掉，但状态从 (g,1,'A') 变成 (g,0,'A')，块内方程组依然可解。
    """
    p_self = 0.0
    for p, moved, key in model.transitions(a, b, st, act):
        if not moved and key[2:] == st:
            p_self += p
    return p_self >= 1.0 - 1e-15


def _iterate(sts, rhs):
    """块内值迭代。rhs(st, v) 返回该状态下一步的代价。"""
    v = {st: 0.0 for st in sts}
    for _ in range(20000):
        diff = 0.0
        for st in sts:
            new = rhs(st, v)
            diff = max(diff, abs(new - v[st]))
            v[st] = new
        if diff < 1e-13:
            return v, True
    return v, False


def _bellman(model, a, b, st, act, v, V):
    """给定动作的单步代价；死循环动作为 inf。"""
    if _is_dead(model, a, b, st, act):
        return math.inf
    acc = 1.0
    for p, moved, key in model.transitions(a, b, st, act):
        acc += p * (V[key] if moved else v[key[2:]])
    return acc


def solve(model: Model, maxa: int, maxb: int):
    """最优解。返回 (V, POL)。"""
    V, POL = {}, {}
    for a, b in _block_order(maxa, maxb):
        sts = model.states()
        if a == 0 and b == 0:
            for st in sts:
                V[(a, b) + st], POL[(a, b) + st] = 0.0, None
            continue

        def rhs(st, v, a=a, b=b):
            return min(_bellman(model, a, b, st, act, v, V)
                       for act in ("A", "B", None))

        v, ok = _iterate(sts, rhs)
        assert ok, f"块 ({a},{b}) 值迭代未收敛"
        for st in sts:
            V[(a, b) + st] = v[st]
            # 收敛后重新取一次 argmin，保证策略与值严格一致
            cand = [( _bellman(model, a, b, st, act, v, V), act)
                    for act in ("A", "B", None)]
            best = min(c for c, _ in cand)
            assert best < math.inf, f"状态 ({a},{b}){st} 无可行动作"
            POL[(a, b) + st] = next(act for c, act in cand if c <= best + 1e-12)
    return V, POL


def eval_policy(model: Model, action_fn, maxa: int, maxb: int):
    """评估给定策略 action_fn(a, b, state) -> 动作。返回 V 字典。

    若某状态在该策略下永远打转，其值记为 inf。
    """
    V = {}
    for a, b in _block_order(maxa, maxb):
        sts = model.states()
        if a == 0 and b == 0:
            for st in sts:
                V[(a, b) + st] = 0.0
            continue

        def rhs(st, v, a=a, b=b):
            return _bellman(model, a, b, st, action_fn(a, b, st), v, V)

        v, _ = _iterate(sts, rhs)
        for st in sts:
            V[(a, b) + st] = v[st]
    return V


# ==========================================================================
# 3. 蒙特卡洛校验
# ==========================================================================
def simulate(model, POL, a0, b0, init, rng, n=200_000):
    total = sq = 0
    for _ in range(n):
        a, b, st = a0, b0, init
        pulls = 0
        while a > 0 or b > 0:
            act = POL[(a, b) + st]
            g, f, s = st
            f2 = model.apply_action(f, s, act)
            dist = model.distribution(g, f2, act)
            r, cum, w = rng.random(), 0.0, dist[-1][0]
            for wc, p in dist:
                cum += p
                if r <= cum:
                    w = wc
                    break
            pulls += 1
            if w == "A" and a > 0:
                a -= 1
            elif w == "B" and b > 0:
                b -= 1
            st = model.step_state(g, f2, act, w)
        total += pulls
        sq += pulls * pulls
    mean = total / n
    var = max(sq / n - mean * mean, 0.0)
    return mean, math.sqrt(var / n)


# ==========================================================================
# 4. 策略规则
# ==========================================================================
def rule_greedy(a, b, st):
    """规则：定轨「还差得更多」的那把；两边一样则保持当前定轨。"""
    g, f, s = st
    if a > b:
        return "A"
    if b > a:
        return "B"
    return s if s is not None else "A"


def rule_refined(a, b, st, F=1):
    """改进规则：缺口大的一边；若两边缺口相等、且处于「大保底 + 命定值已满」，
    则把定轨切到另一把（把「必中指定武器」换成「必中UP 50/50」——失手时
    还能顺手拿到对方那把的命定值）。"""
    g, f, s = st
    if a > b:
        return "A"
    if b > a:
        return "B"
    if s is None:
        return "A"
    if g == 1 and f >= F:
        return "B" if s == "A" else "A"
    return s


# ==========================================================================
# 5. 报告
# ==========================================================================
def print_value_table(V, a_range, b_range, init, title):
    g, f, s = init
    print(f"  {title}")
    print(f"    （初始：大保底={g}  命定值={f}  已定轨={NAMES[s]}）")
    head = "    a\\b |" + "".join(f"{b:>8}" for b in b_range)
    print(head)
    print("    " + "-" * (len(head) - 4))
    for a in a_range:
        print(f"{a:>7} |" + "".join(f"{V[(a, b, g, f, s)]:>8.3f}" for b in b_range))
    print()


def print_policy_table(POL, a_range, b_range, init, title):
    g, f, s = init
    sym = {"A": "A", "B": "B", None: "-"}
    print(f"  {title}  （A=定轨A  B=定轨B  -=不定轨）")
    head = "    a\\b |" + "".join(f"{b:>4}" for b in b_range)
    print(head)
    for a in a_range:
        print(f"{a:>7} |" + "".join(f"{sym[POL[(a, b, g, f, s)]]:>4}"
                                    for b in b_range))
    print()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="官方真实口径：命定值上限 F=2")
    ap.add_argument("--nonup", action="store_true",
                    help="命定值只在出非 UP 时 +1（另一把 UP 不清零也不加）")
    ap.add_argument("--keep-fate", action="store_true",
                    help="切换定轨目标不清空命定值（非官方，敏感性检验用）")
    ap.add_argument("--maxa", type=int, default=6)
    ap.add_argument("--maxb", type=int, default=6)
    ap.add_argument("--asym", action="store_true", help="对角线渐近速率")
    ap.add_argument("--validate", action="store_true", help="蒙特卡洛校验")
    ap.add_argument("--check-none", action="store_true",
                    help="检查「不定轨」是否可能最优")
    ap.add_argument("--csv", default=None, help="输出 CSV")
    args = ap.parse_args()

    F = 2 if args.real else 1
    inc = "non_up" if args.nonup else "non_selected"
    model = Model(F=F, G=True, inc=inc, switch_resets=not args.keep_fate)

    bar = "=" * 78
    print(bar)
    print("原神武器池「定轨」最优抽取策略 —— 精确动态规划")
    print(bar)
    print(f"  出金分布     ：A {P_A:.1%} / B {P_B:.1%} / 非UP {P_OFF:.1%}")
    print(f"  大保底       ：{'开' if model.G else '关'}（出非UP后下一金必为UP，A/B各50%）")
    print(f"  命定值上限 F ：{F}"
          f"{'  ← 官方真实口径（命定值满 2 才保底）' if F == 2 else '  ← 用户口径（歪 1 次即保底）'}")
    print(f"  命定值递增   ：{'出金≠定轨目标即 +1' if inc == 'non_selected' else '仅出非UP时 +1'}")
    print(f"  切换定轨     ：{'清空命定值（官方）' if model.switch_resets else '保留命定值（非官方）'}")
    print()

    maxa, maxb = args.maxa, args.maxb
    V, POL = solve(model, maxa, maxb)
    ar, br = list(range(maxa + 1)), list(range(maxb + 1))

    # ---- 一、单武器基准 -------------------------------------------------
    print("─" * 78)
    print("一、只要一把武器（凑 a 把 A、不要 B）")
    print("─" * 78)
    for Fv in (1, 2):
        if Fv == F:
            e1 = V[(1, 0, 0, 0, None)]
        else:
            e1 = solve(Model(F=Fv, G=True, inc=inc,
                             switch_resets=model.switch_resets), 1, 0)[0][
                (1, 0, 0, 0, None)]
        tag = "官方 F=2" if Fv == 2 else "用户 F=1"
        print(f"  {tag}：定轨后每把期望 {e1:.4f} 抽（n 把 = {e1:.4f}·n + 收尾项）；"
              f"不定轨裸抽 = {1/P_A:.4f} 抽/把")
    print()

    # ---- 二、期望抽数表 -------------------------------------------------
    print("─" * 78)
    print("二、最优期望抽数表 E[a][b]（单位：抽）")
    print("─" * 78)
    fresh = (0, 0, None)
    big = (1, 1, "A")       # 上一金歪了：大保底 + 命定值满（已定轨A）
    offb = (0, 1, "A")      # 上一金是另一把UP：无大保底 + 命定值满
    big0 = (1, 0, "A")      # 有大保底但命定值已清零（例如刚切换过定轨）
    cases = [("① 全新开始（上一金就是定轨目标）", fresh),
             ("② 上一金歪了、未获取到定轨武器", big),
             ("③ 上一金是另一把UP（命定值已满）", offb),
             ("④ 有大保底但命定值已清零", big0)]
    print_value_table(V, ar, br, fresh, cases[0][0])
    print_value_table(V, ar, br, big, cases[1][0])
    print_value_table(V, ar, br, offb, cases[2][0])
    print_value_table(V, ar, br, big0, cases[3][0])
    print("  注：情形②与④的 a=0 行、情形③与①的 a=0 行完全一致（目标里没有那把武器时，"
          "该方向的定轨/命定值无意义）。")
    print()

    # ---- 三、最优首动作 -------------------------------------------------
    print("─" * 78)
    print("三、最优首动作")
    print("─" * 78)
    for title, init in cases:
        print_policy_table(POL, ar, br, init, title)
    print("  读法：先定轨哪一把。抽到东西后重新查表即可。")
    print()

    # ---- 四、规则化：与贪心规则的差距 ------------------------------------
    print("─" * 78)
    print("四、简化规则的损失")
    print("─" * 78)

    def worst_loss(action_fn):
        Vr = eval_policy(model, action_fn, maxa, maxb)
        worst, wkey = 0.0, None
        for a in ar:
            for b in br:
                for st in model.states():
                    opt = V[(a, b) + st]
                    gr = Vr[(a, b) + st]
                    if math.isinf(gr):
                        continue
                    if gr - opt > worst:
                        worst, wkey = gr - opt, (a, b) + st
        return Vr, worst, wkey

    Vg, w_g, k_g = worst_loss(rule_greedy)
    Vf, w_f, k_f = worst_loss(lambda a, b, st: rule_refined(a, b, st, F))
    for name, w, k in (("规则甲 定轨缺口大的那把（平局保持）", w_g, k_g),
                       ("规则乙 甲 + 平局且大保底&命定值满时切到另一把", w_f, k_f)):
        loc = (f"a={k[0]} b={k[1]} g={k[2]} f={k[3]} s={NAMES[k[4]]}"
               if k else "无")
        print(f"  {name:<44} 最大损失 {w:.4f} 抽（{loc}）")
    print()
    print(f"  {'目标':<9}{'最优':>9}{'甲':>9}{'乙':>9}{'乙损失%':>9}"
          f"{'全程定轨A':>11}{'从不定轨':>10}")
    for a, b in [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4), (5, 5),
                 (6, 6), (2, 4), (4, 2)]:
        if a > maxa or b > maxb:
            continue
        opt = V[(a, b, 0, 0, None)]
        vg = Vg[(a, b, 0, 0, None)]
        vf = Vf[(a, b, 0, 0, None)]
        fa = eval_policy(model, lambda x, y, st: "A", a, b)[(a, b, 0, 0, None)]
        fn = eval_policy(model, lambda x, y, st: None, a, b)[(a, b, 0, 0, None)]
        print(f"  A{a}+B{b:<6}{opt:>9.3f}{vg:>9.3f}{vf:>9.3f}"
              f"{100*(vf-opt)/opt:>8.2f}%{fa:>11.3f}{fn:>10.3f}")
    print()

    # ---- 五、对角线渐近 --------------------------------------------------
    if args.asym:
        print("─" * 78)
        print("五、平衡目标 A n + B n 的期望抽数（渐近速率）")
        print("─" * 78)
        N = 20
        Vn = V if (maxa >= N and maxb >= N) else solve(model, N, N)[0]
        prev = 0.0
        for n in range(0, N + 1):
            v = Vn[(n, n, 0, 0, None)]
            print(f"  n={n:>3}  E={v:>9.4f}  边际={v-prev:>7.4f}  "
                  f"每把武器={v/(2*n) if n else 0:>7.4f}")
            prev = v
        print()

    # ---- 六、蒙特卡洛 ----------------------------------------------------
    if args.validate:
        print("─" * 78)
        print("六、蒙特卡洛校验（每项 20 万次，按最优策略）")
        print("─" * 78)
        rng = random.Random(20261013)
        for (a, b), init in [((1, 1), fresh), ((2, 2), fresh), ((3, 3), fresh),
                             ((4, 4), fresh), ((2, 2), big), ((1, 2), offb)]:
            dp = V[(a, b) + init]
            mc, se = simulate(model, POL, a, b, init, rng)
            flag = "OK" if abs(dp - mc) < 4 * se else "*** 不一致 ***"
            print(f"  A{a}+B{b}  init={str(init):<14} DP={dp:8.4f}  "
                  f"MC={mc:8.4f} ± {1.96*se:.4f}  {flag}")
        print()

    # ---- 七、「不定轨」检查 ----------------------------------------------
    if args.check_none:
        print("─" * 78)
        print("七、检查：「不定轨」有没有可能最优")
        print("─" * 78)
        cnt_none = 0
        for a in ar:
            for b in br:
                for st in model.states():
                    if a == 0 and b == 0:
                        continue
                    if POL[(a, b) + st] is None:
                        cnt_none += 1

        def solve_restricted(m):
            """把动作空间砍成 {定轨A, 定轨B} 后的最优解。"""
            V2 = {}
            for a, b in _block_order(maxa, maxb):
                sts = m.states()
                if a == 0 and b == 0:
                    for st in sts:
                        V2[(a, b) + st] = 0.0
                    continue

                def rhs(st, v, a=a, b=b):
                    return min(_bellman(m, a, b, st, act, v, V2)
                               for act in ("A", "B"))

                v, _ = _iterate(sts, rhs)
                for st in sts:
                    V2[(a, b) + st] = v[st]
            return V2

        Vr = solve_restricted(model)
        mx = 0.0
        for a in ar:
            for b in br:
                for st in model.states():
                    mx = max(mx, Vr[(a, b) + st] - V[(a, b) + st])
        print(f"  最优策略中取「不定轨」的状态数 = {cnt_none}")
        print(f"  把「不定轨」从动作空间删掉后，最大价值变化 = {mx:.6f} 抽")
        print("  → 「取消定轨」永远不是严格最优：想清空命定值就直接切到另一把，"
              "出金分布完全相同，还额外把命定值记在新目标上。")
        print()

    # ---- CSV -------------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "初始状态", "大保底g", "命定值f", "已定轨",
                        "期望抽数", "最优首动作"])
            for a in ar:
                for b in br:
                    for label, (g, f, s) in [(c[0][2:], c[1]) for c in cases]:
                        w.writerow([a, b, label, g, f, NAMES[s],
                                    f"{V[(a,b,g,f,s)]:.6f}",
                                    ACT_NAME[POL[(a, b, g, f, s)]]])
        print(f"结果已写入 {args.csv}")


if __name__ == "__main__":
    main()
