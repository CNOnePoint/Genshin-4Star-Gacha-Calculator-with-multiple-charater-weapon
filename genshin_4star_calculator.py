#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原神四星抽卡概率计算器（马尔可夫链精确解）。

模型
----
每个卡池维护一条马尔可夫链，状态为::

    (pity, guarantee, c1, ..., cn)

* ``pity``      —— 距离上一个四星已经抽了多少发（0 ~ n_pity-1）。第 ``pity+1`` 抽出四星的
                   概率由概率表给出；最后一档的概率 >100%，视为必出，所以 pity 不会越界。
* ``guarantee`` —— 四星大保底标记（0/1）。未处于大保底时，出四星有 ``up_rate`` 的概率
                   为 UP、其余为非 UP 并进入大保底（角色池 50%、**武器池 75%**）；
                   处于大保底时，出四星 100% 为 UP 并解除大保底。
* ``ci``        —— 每个 UP 物品各自已经抽到的数量，在目标数处饱和（抽多了不影响结果）。

概率表（第 n 抽出四星）—— 两个池子的保底档数不同
    | 抽数 | 角色池 | 武器池 |
    | 1~7  | 5.1%   | 6.0%   |
    | 8    | 5.1%   | 66.0%  |
    | 9    | 56.1%  | 126.0% |   <- 武器池第 9 抽必出
    | 10   | 107.1% |   -    |   <- 角色池第 10 抽必出

变成 UP 的四星，会在该池的 UP 物品中等概率随机取一个：
角色池 3 个 UP（不歪 50%，每个 50%÷3）、武器池 5 个 UP（不歪 75%，每个 75%÷5）。

抽卡顺序（优先池）
    先在一个池子里抽到「该池计划全部达成」，再切到另一个池子；抽数用尽即结束。
    于是：P(全部达成) = Σ_k P(第一池恰好在第 k 抽完成) × P(第二池在 N-k 抽内完成)。

用法
----
命令行::

    python genshin_4star_calculator.py --order weapon --pulls 200 ^
        --char-pity 3 --weapon-pity 5 --weapon-guarantee ^
        --char-targets 2 1 0 --weapon-targets 1 0 1

交互式::

    python genshin_4star_calculator.py -i

蒙特卡洛自检（和上面的精确解对比）::

    python genshin_4star_calculator.py -i --verify 20000

作者: 由 AI 生成 | 纯标准库实现，无需安装 numpy。
"""

from __future__ import annotations

import argparse
import json
import operator
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# 中文输出更稳：管道/重定向时按 UTF-8 写出（真实控制台本就使用 UTF-8，不受影响）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - 极端环境下忽略
            pass


# ============================================================================
# 一、卡池与概率表
# ============================================================================

#: 角色池：第 1~8 抽 5.1%，第 9 抽 56.1%，第 10 抽 107.1%（>100% 即必出）—— 10 档保底
CHARACTER_TABLE: Tuple[float, ...] = (0.051,) * 8 + (0.561, 1.071)
#: 武器池：第 1~7 抽 6.0%，第 8 抽 66.0%，第 9 抽 126.0%（>100% 即必出）—— 9 档保底
WEAPON_TABLE: Tuple[float, ...] = (0.060,) * 7 + (0.660, 1.260)

MAX_PITY = max(len(CHARACTER_TABLE), len(WEAPON_TABLE))   # 最长的保底档数（角色池 10 档）
UP_RATE = 0.50       # 「没歪」的默认概率（角色池）；武器池是 0.75，见各自卡池定义
CHARACTER_UP_ITEMS = 3   # 角色池有 3 个 UP 四星
WEAPON_UP_ITEMS = 5      # 武器池有 5 个 UP 四星（中 UP 时 5 选 1，每个 1/5）


@dataclass(frozen=True)
class Banner:
    """一个卡池的静态定义。"""

    key: str                            # 'character' / 'weapon'
    name: str                           # '角色池' / '武器池'
    table: Tuple[float, ...]            # 原始概率表（第 1..n_pity 抽）
    up_names: Tuple[str, ...]           # UP 物品的名字（角色池 3 个，武器池 5 个）
    up_rate: float = UP_RATE            # 未处于大保底时，出四星「不歪」的概率

    @property
    def n_up(self) -> int:
        """这个池子有几个 UP 四星。"""
        return len(self.up_names)

    @property
    def n_pity(self) -> int:
        """保底档数 = 最多几抽必出四星（角色池 10、武器池 9）。"""
        return len(self.table)

    @property
    def lose_rate(self) -> float:
        """歪的概率（进大保底）。"""
        return 1.0 - self.up_rate

    @property
    def probs(self) -> Tuple[float, ...]:
        """真正参与运算的概率：超过 100% 的一律按 100%（必出）处理。"""
        return tuple(min(1.0, p) for p in self.table)


#: 角色池：不歪概率 50%，3 个 UP 角色 -> 非大保底时指定某个角色 50% ÷ 3
CHARACTER = Banner("character", "角色池", CHARACTER_TABLE,
                   ("UP角色1", "UP角色2", "UP角色3"), up_rate=0.50)
#: 武器池：不歪概率 75%，5 个 UP 武器 -> 非大保底时指定某把武器 75% ÷ 5
WEAPON = Banner("weapon", "武器池", WEAPON_TABLE,
                ("UP武器1", "UP武器2", "UP武器3", "UP武器4", "UP武器5"), up_rate=0.75)
BANNERS: Dict[str, Banner] = {"character": CHARACTER, "weapon": WEAPON}

ORDER_LABEL = {"character-first": "先角色池，后武器池", "weapon-first": "先武器池，后角色池"}


def mean_interval(probs: Sequence[float]) -> float:
    """从「刚出过四星」开始，抽出下一个四星所需抽数的期望（平均出货间隔）。"""
    e = 0.0
    miss = 1.0
    for n, q in enumerate(probs, start=1):
        if q >= 1.0:
            e += n * miss
            break
        e += n * miss * q
        miss *= 1.0 - q
    return e


# ============================================================================
# 二、单卡池的马尔可夫链
# ============================================================================


class FourStarChain:
    """一个卡池的四星马尔可夫链（分布 = 20 × 累计获得数格子 的概率向量）。

    状态编号: ``((pity * 2) + guarantee) * size + count_index``
    其中 ``count_index`` 是各个 UP 计数在各自的 ``target+1`` 进制下的行主序下标。
    UP 的个数由卡池决定：角色池 3 个，武器池 5 个。
    """

    def __init__(
        self,
        banner: Banner,
        targets: Sequence[int],
        start_pity: int = 0,
        start_guarantee: bool = False,
    ) -> None:
        self.banner = banner
        self.n_up = banner.n_up
        self.n_pity = banner.n_pity                 # 角色池 10 档、武器池 9 档
        self.up_rate = banner.up_rate               # 角色池 0.50、武器池 0.75
        self.targets: Tuple[int, ...] = tuple(int(t) for t in targets)
        if len(self.targets) != self.n_up or any(t < 0 for t in self.targets):
            raise ValueError(f"{banner.name}的目标数量必须是 {self.n_up} 个非负整数")
        if not 0 <= int(start_pity) < self.n_pity:
            raise ValueError(
                f"{banner.name}已垫的抽数必须在 0~{self.n_pity - 1} 之间"
                f"（垫满 {self.n_pity} 抽时第 {self.n_pity} 抽已经必出四星）"
            )

        self.q = banner.probs                       # q[p] = 第 p+1 抽出四星的概率
        self.sizes: Tuple[int, ...] = tuple(t + 1 for t in self.targets)
        self.size = 1
        for s in self.sizes:
            self.size *= s

        # 每个维度在行主序下的步长
        strides = [0] * self.n_up
        acc = 1
        for i in range(self.n_up - 1, -1, -1):
            strides[i] = acc
            acc *= self.sizes[i]
        self.strides = tuple(strides)
        self.corner = self.size - 1                 # 所有目标都达成（饱和）的格子

        # 初始分布：pity 确定、大保底确定、所有计数均为 0（下标 0）
        self.dist: List[float] = [0.0] * (2 * self.n_pity * self.size)
        self.dist[((int(start_pity) * 2) + (1 if start_guarantee else 0)) * self.size] = 1.0

        # 预计算：第 i 个 UP 计数 +1（在目标处饱和）后的格子下标
        self._inc: List[List[int]] = []
        for i in range(self.n_up):
            st, sz, tgt = self.strides[i], self.sizes[i], self.targets[i]
            self._inc.append([ci + st if (ci // st) % sz < tgt else ci for ci in range(self.size)])

        # 预计算：第 i 个 UP 已达标的格子区间（目标为 0 时恒为 1.0，特殊处理）
        self._reached_ranges: List[List[Tuple[int, int]]] = []
        for i in range(self.n_up):
            st, sz, tgt = self.strides[i], self.sizes[i], self.targets[i]
            ranges: List[Tuple[int, int]] = []
            if tgt > 0:
                outer = sz * st
                for base in range(0, self.size, outer):
                    start = base + tgt * st
                    ranges.append((start, start + st))
            self._reached_ranges.append(ranges)

        # 预计算：第 i 个 UP 的「已获得数量」（用于算期望）
        self._weights: List[List[int]] = [
            [(ci // self.strides[i]) % self.sizes[i] for ci in range(self.size)]
            for i in range(self.n_up)
        ]

    def counts_of(self, index: int) -> Tuple[int, ...]:
        """把数量格子的扁平下标还原成每个 UP 的数量。"""
        return tuple((index // self.strides[i]) % self.sizes[i] for i in range(self.n_up))

    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """推进一抽。"""
        dist = self.dist
        S = self.size
        q = self.q
        new = [0.0] * (2 * self.n_pity * S)

        # 1) 没出四星：保底计数 +1（最后一档 q=1，不可能走这条路）
        for p in range(self.n_pity - 1):
            rem = 1.0 - q[p]
            if rem <= 0.0:
                continue
            src = p * 2 * S
            dst = (p + 1) * 2 * S
            new[dst:dst + S] = [v * rem for v in dist[src:src + S]]
            new[dst + S:dst + 2 * S] = [v * rem for v in dist[src + S:src + 2 * S]]

        # 2) 出了四星：保底计数归零，把两种大保底状态的质量分别汇总
        up_unguarded = [0.0] * S   # 来自「无大保底」的四星到达质量
        up_guarded = [0.0] * S     # 来自「大保底」的四星到达质量
        for p in range(self.n_pity):
            qp = q[p]
            if qp <= 0.0:
                continue
            base = p * 2 * S
            up_unguarded = [x + y * qp for x, y in zip(up_unguarded, dist[base:base + S])]
            up_guarded = [x + y * qp for x, y in zip(up_guarded, dist[base + S:base + 2 * S])]

        # 2a) 命中 UP：n_up 个 UP 等概率，之后不再是大保底（guarantee=0，即 new[0:S] 段）
        up_rate = self.up_rate
        coef_unguarded = up_rate / self.n_up           # 角色 0.5/3、武器 0.75/5
        coef_guarded = 1.0 / self.n_up                 # 大保底：1.0/n_up
        for i in range(self.n_up):
            for j, v0, v1 in zip(self._inc[i], up_unguarded, up_guarded):
                if v0:
                    new[j] += v0 * coef_unguarded
                if v1:
                    new[j] += v1 * coef_guarded

        # 2b) 未命中 UP（只在无大保底时可能）：进入大保底（guarantee=1，即 new[S:2S] 段）
        new[S:2 * S] = [v * (1.0 - up_rate) for v in up_unguarded]

        self.dist = new

    # ------------------------------------------------------------------ #
    def marginal_counts(self) -> List[float]:
        """把 (pity, guarantee) 合并掉，得到「累计获得数量」的边缘分布（长度 size）。"""
        dist = self.dist
        S = self.size
        marg = dist[0:S]
        for k in range(1, 2 * self.n_pity):
            base = k * S
            marg = [x + y for x, y in zip(marg, dist[base:base + S])]
        return marg

    def pity_guarantee_marginal(self) -> List[List[float]]:
        """(保底计数, 大保底) 的边缘分布：``out[p][g]``，用于校验概率表/保底机制。"""
        S = self.size
        out: List[List[float]] = []
        for p in range(self.n_pity):
            row = []
            for g in range(2):
                base = (p * 2 + g) * S
                row.append(sum(self.dist[base:base + S]))
            out.append(row)
        return out

    def complete_prob(self, marg: Sequence[float] | None = None) -> float:
        """这个池子的所有 UP 目标都达成的概率。"""
        if marg is None:
            marg = self.marginal_counts()
        return marg[self.corner]

    def item_reached_probs(self, marg: Sequence[float] | None = None) -> List[float]:
        """每个 UP 各自「已达成目标」的概率。"""
        if marg is None:
            marg = self.marginal_counts()
        out: List[float] = []
        for i in range(self.n_up):
            if self.targets[i] == 0:
                out.append(1.0)                       # 目标为 0，永远算达成
            else:
                out.append(sum(sum(marg[a:b]) for a, b in self._reached_ranges[i]))
        return out

    def item_expectations(self, marg: Sequence[float] | None = None) -> List[float]:
        """每个 UP 的期望获得数量（在目标处封顶，即 E[min(数量, 目标)]）。"""
        if marg is None:
            marg = self.marginal_counts()
        return [sum(map(operator.mul, self._weights[i], marg)) for i in range(self.n_up)]


# ============================================================================
# 三、按抽数展开的一条曲线：F(k)、各物品达成概率、期望
# ============================================================================


@dataclass
class BannerCurve:
    """某个卡池在 0..N 抽预算下的完整结果。"""

    banner: Banner
    targets: Tuple[int, ...]
    start_pity: int
    start_guarantee: bool
    max_pulls: int
    complete: List[float]                 # complete[k] = P(该池计划在 <= k 抽内完成)
    item_prob: List[List[float]]          # item_prob[k][i] = P(第 i 个 UP 在 k 抽内达标)
    item_expect: List[List[float]]        # item_expect[k][i] = E[min(数量, 目标)]
    count_snapshots: Dict[int, List[float]] = field(default_factory=dict)
    # count_snapshots[k] = 第 k 抽时该池「已获得数量」的分布快照（结果分布要用）
    weighted_mixture: Optional[List[float]] = None
    # weighted_mixture = 按外部给定权重混合后的数量分布 Σ_k w[k]·dist_k（结果分布要用）

    # ------------------------------------------------------------------ #
    @property
    def n_up(self) -> int:
        """这个池子有几个 UP（角色 3、武器 5）。"""
        return self.banner.n_up

    @property
    def final(self) -> float:
        """该池计划在「曲线算到的最大抽数 max_pulls」内完成的概率。

        注意：报告里「单独给满 N 抽」用的是 ``complete[N]``（N = 当前预算），
        当曲线算得更远时 ``final``（= complete[max_pulls]）会更大。
        """
        return self.complete[self.max_pulls]

    def exact_completion(self) -> List[float]:
        """「恰好在第 k 抽完成」的概率（首次达标时间分布）。"""
        out: List[float] = []
        prev = 0.0
        for v in self.complete:
            out.append(v - prev)
            prev = v
        return out

    def expected_pulls_used(self, budget: int | None = None) -> float:
        """E[min(T, budget)]：预计消耗抽数（用尽预算则按 budget 计）。

        依据 E[min(T, m)] = Σ_{j<m} P(T > j) = Σ_{j<m} (1 - F(j))。
        """
        m = self.max_pulls if budget is None else int(budget)
        return sum(1.0 - v for v in self.complete[:m])


def analyze_banner(
    banner: Banner,
    targets: Sequence[int],
    start_pity: int,
    start_guarantee: bool,
    max_pulls: int,
    want_expect: bool = True,
    snapshots: Sequence[int] = (),
    mixture_weights: Optional[Sequence[float]] = None,
) -> BannerCurve:
    """把某个卡池的马尔可夫链从 0 抽一直推到 max_pulls 抽，记录每一步的结果。

    ``snapshots`` 里的抽数会额外保存当时的「已获得数量」分布；
    ``mixture_weights`` 给出权重 w[k] 时，额外累加 Σ_k w[k]·dist_k（都是顺手做的，
    不会多跑一遍动态规划）。
    """
    chain = FourStarChain(banner, targets, start_pity, start_guarantee)
    n = int(max_pulls)
    n_up = banner.n_up
    complete: List[float] = [0.0] * (n + 1)
    item_prob: List[List[float]] = [[0.0] * n_up for _ in range(n + 1)]
    item_expect: List[List[float]] = [[0.0] * n_up for _ in range(n + 1)]
    snap_set = {int(k) for k in snapshots}
    count_snapshots: Dict[int, List[float]] = {}
    mixture: Optional[List[float]] = [0.0] * chain.size if mixture_weights is not None else None

    for k in range(n + 1):
        if k:
            chain.step()
        marg = chain.marginal_counts()
        complete[k] = marg[chain.corner]
        probs = chain.item_reached_probs(marg)
        for i in range(n_up):
            item_prob[k][i] = probs[i]
        if want_expect:
            exps = chain.item_expectations(marg)
            for i in range(n_up):
                item_expect[k][i] = exps[i]
        if k in snap_set:
            count_snapshots[k] = marg
        if mixture is not None:
            w = mixture_weights[k] if k < len(mixture_weights) else 0.0  # type: ignore[index]
            if w:
                for idx, v in enumerate(marg):
                    if v:
                        mixture[idx] += w * v

    return BannerCurve(
        banner=banner,
        targets=tuple(int(t) for t in targets),  # type: ignore[arg-type]
        start_pity=int(start_pity),
        start_guarantee=bool(start_guarantee),
        max_pulls=n,
        complete=complete,
        item_prob=item_prob,
        item_expect=item_expect,
        count_snapshots=count_snapshots,
        weighted_mixture=mixture,
    )


# ============================================================================
# 四、把两个池子按抽卡顺序组合起来
# ============================================================================


@dataclass
class OrderOutcome:
    """在某个抽卡顺序下的最终结论。"""

    order: str                                # 'weapon-first' / 'character-first'
    first: BannerCurve
    second: BannerCurve
    budget: int                               # 本次使用的抽数预算
    success: float                            # 全部计划完成
    second_complete: float                    # 轮到第二个池子时能完成的概率
    expected_pulls_used: float                # 平均消耗抽数（含失败时耗尽预算）
    item_prob: Dict[str, List[float]] = field(default_factory=dict)
    item_expect: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return ORDER_LABEL[self.order]

    @property
    def first_complete(self) -> float:
        """第一池在本次预算内完成的概率（第一个池子总能拿满全部预算）。"""
        return self.first.complete[self.budget]


def combine_curves(
    first: BannerCurve, second: BannerCurve, order: str, budget: int
) -> OrderOutcome:
    """先抽 first 池、达标后转 second 池，计算 budget 抽内的成功率与各物品达成概率。

    ``first`` / ``second`` 的曲线可以算到比 budget 更远的抽数（用于画成功率曲线），
    这里只取前 budget 抽的部分。
    """
    n = int(budget)
    assert 0 <= n <= first.max_pulls and 0 <= n <= second.max_pulls
    exact = first.exact_completion()          # 第一池恰好在第 k 抽完成
    fail_first = 1.0 - first.complete[n]      # 预算耗尽仍未完成第一池

    # 总成功率：第一池恰在第 k 抽完成 × 第二池在 N-k 抽内完成
    success = sum(exact[k] * second.complete[n - k] for k in range(n + 1))

    # 第二池的完成概率：注意第一池失败时它只有 0 抽预算
    second_complete = success + fail_first * second.complete[0]

    # 平均消耗抽数 = 第一池用时 + 第二池用时（第一池失败则耗尽全部预算）
    expected_used = sum(
        exact[k] * (k + second.expected_pulls_used(n - k)) for k in range(n + 1)
    ) + fail_first * n

    # 各物品达成概率 / 期望获得数（两个池子的 UP 个数不同：角色 3 个、武器 5 个）
    item_prob: Dict[str, List[float]] = {first.banner.key: [], second.banner.key: []}
    item_expect: Dict[str, List[float]] = {first.banner.key: [], second.banner.key: []}
    for i in range(first.n_up):
        # 第一池：无论如何都会一直抽到完成或用尽，所以直接看 k=N 的边际
        item_prob[first.banner.key].append(first.item_prob[n][i])
        item_expect[first.banner.key].append(first.item_expect[n][i])
    for i in range(second.n_up):
        # 第二池：按第一池的完成时间混合
        mixed_p = sum(exact[k] * second.item_prob[n - k][i] for k in range(n + 1))
        mixed_p += fail_first * second.item_prob[0][i]
        mixed_e = sum(exact[k] * second.item_expect[n - k][i] for k in range(n + 1))
        mixed_e += fail_first * second.item_expect[0][i]
        item_prob[second.banner.key].append(mixed_p)
        item_expect[second.banner.key].append(mixed_e)

    return OrderOutcome(
        order=order,
        first=first,
        second=second,
        budget=n,
        success=success,
        second_complete=second_complete,
        expected_pulls_used=expected_used,
        item_prob=item_prob,
        item_expect=item_expect,
    )


# ============================================================================
# 五、抽卡计划与总入口
# ============================================================================


@dataclass
class OutcomeDistribution:
    """把全部抽数花完之后，最终获得情况的分布（按出现概率从高到低排序）。

    ``entries`` 里每一项是 ``(角色各 UP 的最终数量, 武器各 UP 的最终数量, 概率)``；
    数量在目标处封顶（抽多了不再计入），所以 ``缺口 = Σ(目标 - 获得)`` 恒 ≥ 0，
    缺口为 0 就等于「计划全部达成」。
    """

    plan: Plan
    first_key: str
    second_key: str
    total_targets: int
    entries: List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]
    gap_probs: List[float]        # gap_probs[g] = P(还差 g 个目标物品)

    @property
    def total_outcomes(self) -> int:
        """可能出现的结果种数（概率大于 0）。"""
        return len(self.entries)

    @property
    def success(self) -> float:
        return self.gap_probs[0]

    @property
    def expected_gap(self) -> float:
        return sum(g * p for g, p in enumerate(self.gap_probs))

    def top(self, k: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]:
        return self.entries[:max(0, int(k))]

    def omitted(self, k: int) -> Tuple[int, float]:
        """(没显示出来的结果种数, 它们合计的概率)。"""
        rest = self.entries[max(0, int(k)):]
        return len(rest), sum(e[2] for e in rest)

    def item_marginal(self, key: str, index: int) -> float:
        """某个 UP 物品最终达到目标的概率（用于和分池结果交叉验证）。"""
        target = self.plan.target_of(key)[index]
        total = 0.0
        for char_counts, weapon_counts, p in self.entries:
            counts = char_counts if key == "character" else weapon_counts
            if counts[index] >= target:
                total += p
        return total


def build_outcome_distribution(plan: Plan, result: PlanResult) -> OutcomeDistribution:
    """由已经算好的内容组合出「最终结果分布」。

    设优先池单独抽完需要 T1 抽（首次达标时间），则：

    * ``T1 <= N``（优先池在预算内完成）：优先池一定是满目标（目标处饱和），
      另一池的分布 = 按 P(T1 = N - k) 混合的「k 抽后的数量分布」；
    * ``T1 > N``（优先池没完成）：优先池就是 N 抽后的数量分布（去掉已完成的满目标部分），
      另一池一抽没抽，数量全为 0。

    两种情况在优先池的维度上互斥，加起来正好是 1。
    """
    plan_key = plan.order
    first_key, second_key = {
        "character-first": ("character", "weapon"),
        "weapon-first": ("weapon", "character"),
    }[plan_key]
    first = result.curves[first_key]
    second = result.curves[second_key]
    budget = plan.pulls

    if budget not in first.count_snapshots or second.weighted_mixture is None:
        raise ValueError("请用 solve(plan, want_distribution=True) 计算后再取结果分布")

    marg_first = first.count_snapshots[budget]          # 优先池 N 抽后的数量分布
    mixed_second = second.weighted_mixture              # Σ_k P(T1=N-k)·(k 抽后的数量分布)
    s1, s2 = len(marg_first), len(mixed_second)
    corner1, corner2 = s1 - 1, s2 - 1

    # 数量格子下标 -> 每个 UP 的数量（按行主序的通用还原）
    def coords(index: int, curve: BannerCurve) -> Tuple[int, ...]:
        acc = index
        out = [0] * curve.banner.n_up
        for i in range(curve.banner.n_up - 1, -1, -1):
            size = curve.targets[i] + 1
            out[i] = acc % size
            acc //= size
        return tuple(out)

    char_first = first_key == "character"
    entries: List[Tuple[Tuple[int, ...], Tuple[int, ...], float]] = []

    def add(first_index: int, second_index: int, prob: float) -> None:
        if prob <= 0.0:
            return
        first_counts = coords(first_index, first)
        second_counts = coords(second_index, second)
        if char_first:
            entries.append((first_counts, second_counts, prob))
        else:
            entries.append((second_counts, first_counts, prob))

    # 情况 A：优先池在预算内完成 -> 优先池满目标，另一池吃剩下的抽数
    for b in range(s2):
        add(corner1, b, mixed_second[b])
    # 情况 B：优先池没完成 -> 另一池 0 抽（数量全 0，即下标 0）
    for a in range(s1):
        if a != corner1:
            add(a, 0, marg_first[a])

    total_targets = sum(plan.char_targets) + sum(plan.weapon_targets)
    gap_probs = [0.0] * (total_targets + 1)
    for char_counts, weapon_counts, p in entries:
        gap = _gap_of(plan, char_counts, weapon_counts)
        gap_probs[gap] += p

    def sort_key(item: Tuple[Tuple[int, ...], Tuple[int, ...], float]):
        char_counts, weapon_counts, p = item
        gap = _gap_of(plan, char_counts, weapon_counts)
        # 概率高的在前；概率相同时「拿到的更多」的在前
        return (-p, gap, tuple(-x for x in char_counts), tuple(-x for x in weapon_counts))

    entries.sort(key=sort_key)
    return OutcomeDistribution(
        plan=plan,
        first_key=first_key,
        second_key=second_key,
        total_targets=total_targets,
        entries=entries,
        gap_probs=gap_probs,
    )


@dataclass
class Plan:
    order: str                       # 'character-first' / 'weapon-first'
    pulls: int                       # 可用抽数
    char_pity: int
    char_guarantee: bool
    weapon_pity: int
    weapon_guarantee: bool
    char_targets: Tuple[int, ...]    # 角色池 3 个 UP 的目标数量
    weapon_targets: Tuple[int, ...]  # 武器池 5 个 UP 的目标数量

    def target_of(self, key: str) -> Tuple[int, ...]:
        return self.char_targets if key == "character" else self.weapon_targets

    def pity_of(self, key: str) -> int:
        return self.char_pity if key == "character" else self.weapon_pity

    def guarantee_of(self, key: str) -> bool:
        return self.char_guarantee if key == "character" else self.weapon_guarantee


@dataclass
class SuccessCurve:
    """「达成抽卡计划的成功率」随抽数变化的函数。

    ``prob[m]`` = 给 m 抽时把角色池 + 武器池的计划**全部**完成的概率。
    ``prob`` 单调不减，``prob[-1]`` 是在 ``max_pulls`` 抽内的成功率。
    """

    max_pulls: int
    prob: List[float]
    curves: Dict[str, BannerCurve]
    order: str

    def at(self, pulls: int) -> float:
        """给 pulls 抽时的成功率（pulls 不能超过 max_pulls）。"""
        m = int(pulls)
        if m < 0 or m > self.max_pulls:
            raise ValueError(f"抽数必须在 0 ~ {self.max_pulls} 之间")
        return self.prob[m]

    def needed_for(self, level: float, start: int = 0) -> Optional[int]:
        """最少需要多少抽才能让成功率达到 level；在范围内达不到则返回 None。"""
        for m in range(max(0, int(start)), self.max_pulls + 1):
            if self.prob[m] >= level - 1e-12:
                return m
        return None

    def gains(self, current: int, steps: Sequence[int] = (10, 20, 30, 50)) -> List[Tuple[int, float]]:
        """从 current 抽再额外投入若干抽后的成功率（超出 max_pulls 的会被跳过）。"""
        out: List[Tuple[int, float]] = []
        for step in steps:
            m = int(current) + int(step)
            if m <= self.max_pulls:
                out.append((step, self.prob[m]))
        return out

    def as_pairs(self) -> List[Tuple[int, float]]:
        """[(抽数, 成功率)]，方便导出/画图。"""
        return list(enumerate(self.prob))


def build_success_curve(
    curves: Dict[str, BannerCurve], order: str, max_pulls: int
) -> SuccessCurve:
    """由两个池子的曲线做「总消耗 T = T角色 + T武器」的卷积，得到成功率函数。

    因为两池独立，P(T ≤ m) = Σ_k P(T_first = k) · F_second(m − k)，
    与先抽哪个池子无关（两种顺序只是同一个卷积的不同展开）。
    """
    n = int(max_pulls)
    first_key, second_key = {
        "character-first": ("character", "weapon"),
        "weapon-first": ("weapon", "character"),
    }[order]
    exact = curves[first_key].exact_completion()
    cumulative = curves[second_key].complete

    prob: List[float] = [0.0] * (n + 1)
    for m in range(n + 1):
        a = exact[:m + 1]
        b = cumulative[m::-1]
        prob[m] = sum(map(operator.mul, a, b))
    # 兜底：浮点误差可能让值略微越界
    for m in range(n + 1):
        prob[m] = min(1.0, max(0.0, prob[m]))
    return SuccessCurve(max_pulls=n, prob=prob, curves=curves, order=order)


@dataclass
class PlanResult:
    plan: Plan
    curves: Dict[str, BannerCurve]
    outcomes: Dict[str, OrderOutcome]      # 两种顺序都算，方便对比
    curve: SuccessCurve                    # 成功率随抽数变化的函数
    state_space: int                       # 单池马尔可夫链格子数（诊断用）
    distribution: Optional[OutcomeDistribution] = None   # 花光抽数后的结果分布

    @property
    def chosen(self) -> OrderOutcome:
        return self.outcomes[self.plan.order]

    @property
    def other(self) -> OrderOutcome:
        other_order = "character-first" if self.plan.order == "weapon-first" else "weapon-first"
        return self.outcomes[other_order]

    @property
    def success_at_current_pulls(self) -> float:
        """当前抽数下达成整个计划的成功率。"""
        return self.curve.at(self.plan.pulls)


def solve(plan: Plan, max_pulls: int | None = None, want_distribution: bool = False) -> PlanResult:
    """核心入口。

    ``max_pulls`` 只影响「成功率随抽数变化」这条曲线算到多远，不影响 ``plan.pulls``
    处的结论；省略时曲线只算到 ``plan.pulls``。
    ``want_distribution=True`` 时额外给出「花光全部抽数后的结果分布」。
    """
    if plan.pulls < 0:
        raise ValueError("抽数不能为负")
    if len(plan.char_targets) != CHARACTER.n_up or len(plan.weapon_targets) != WEAPON.n_up:
        raise ValueError(f"角色池要 {CHARACTER.n_up} 个目标、武器池要 {WEAPON.n_up} 个目标")
    if any(t < 0 for t in plan.char_targets) or any(t < 0 for t in plan.weapon_targets):
        raise ValueError("目标数量不能为负")
    horizon = max(plan.pulls, int(max_pulls) if max_pulls is not None else plan.pulls)

    orders = {
        "character-first": ("character", "weapon"),
        "weapon-first": ("weapon", "character"),
    }
    first_key, second_key = orders[plan.order]

    curves: Dict[str, BannerCurve] = {}
    # 先算优先池：结果分布要用它的「首次达标时间」给另一池的分布配权重
    curves[first_key] = analyze_banner(
        BANNERS[first_key],
        plan.target_of(first_key),
        plan.pity_of(first_key),
        plan.guarantee_of(first_key),
        horizon,
        snapshots=(plan.pulls,) if want_distribution else (),
    )
    mixture_weights: Optional[List[float]] = None
    if want_distribution:
        exact_first = curves[first_key].exact_completion()
        # 另一池抽 k 抽的权重 = P(优先池恰好在 N-k 抽完成)；k > N 时不可能
        mixture_weights = [
            exact_first[plan.pulls - k] if k <= plan.pulls else 0.0 for k in range(horizon + 1)
        ]
    curves[second_key] = analyze_banner(
        BANNERS[second_key],
        plan.target_of(second_key),
        plan.pity_of(second_key),
        plan.guarantee_of(second_key),
        horizon,
        mixture_weights=mixture_weights,
    )

    outcomes: Dict[str, OrderOutcome] = {}
    for order, (fk, sk) in orders.items():
        outcomes[order] = combine_curves(curves[fk], curves[sk], order, plan.pulls)

    # 诊断用：单池马尔可夫链的格子数 = 2 × 保底档数 × Π(目标+1)（含 pity 与大保底两种标记）
    state_space = 0
    for key in ("character", "weapon"):
        size = 1
        for t in plan.target_of(key):
            size *= t + 1
        state_space += 2 * BANNERS[key].n_pity * size

    curve = build_success_curve(curves, plan.order, horizon)
    result = PlanResult(
        plan=plan, curves=curves, outcomes=outcomes, curve=curve, state_space=state_space
    )
    if want_distribution:
        result.distribution = build_outcome_distribution(plan, result)
    return result


# ============================================================================
# 六、蒙特卡洛模拟（用于自检 / 对照）
# ============================================================================


def _simulate_pull(banner: Banner, rng: random.Random, state: List) -> int | None:
    """模拟一抽。state = [pity, guarantee, c1, ..., cn]；返回命中的 UP 序号或 None。"""
    pity = state[0]
    q = banner.probs[pity]
    if rng.random() < q:
        state[0] = 0                       # 出四星，保底计数归零
        if state[1]:
            is_up = True
        else:
            is_up = rng.random() < banner.up_rate
        state[1] = 1 if not is_up else 0   # 歪了就进大保底，中了就解除
        if is_up:
            idx = rng.randrange(banner.n_up)
            state[2 + idx] += 1
            return idx
    else:
        state[0] = pity + 1
    return None


def monte_carlo(plan: Plan, trials: int, seed: int | None = None) -> Dict[str, object]:
    """按抽卡顺序模拟整个抽卡过程，返回与精确解相同口径的统计量。"""
    rng = random.Random(seed)
    banners = {"character": CHARACTER, "weapon": WEAPON}
    first_key, second_key = {
        "character-first": ("character", "weapon"),
        "weapon-first": ("weapon", "character"),
    }[plan.order]

    success = 0
    met_first = 0
    met_second = 0
    item_met = {key: [0] * BANNERS[key].n_up for key in ("character", "weapon")}
    total_used = 0
    outcome_hist: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}

    for _ in range(trials):
        states = {
            key: [plan.pity_of(key), 1 if plan.guarantee_of(key) else 0]
                 + [0] * BANNERS[key].n_up
            for key in ("character", "weapon")
        }
        remaining = plan.pulls
        used = 0

        def done(key: str) -> bool:
            t = plan.target_of(key)
            return all(states[key][2 + i] >= t[i] for i in range(len(t)))

        for key in (first_key, second_key):
            while not done(key) and remaining > 0:
                _simulate_pull(banners[key], rng, states[key])
                remaining -= 1
                used += 1
        total_used += used

        first_ok = done(first_key)
        second_ok = done(second_key)
        met_first += first_ok
        met_second += second_ok
        if first_ok and second_ok:
            success += 1
        for key in ("character", "weapon"):
            t = plan.target_of(key)
            for i in range(len(t)):
                if states[key][2 + i] >= t[i]:
                    item_met[key][i] += 1

        # 最终结果（数量在目标处封顶，和精确解口径一致）
        final_key = (
            tuple(min(states["character"][2 + i], plan.char_targets[i])
                  for i in range(len(plan.char_targets))),
            tuple(min(states["weapon"][2 + i], plan.weapon_targets[i])
                  for i in range(len(plan.weapon_targets))),
        )
        outcome_hist[final_key] = outcome_hist.get(final_key, 0) + 1

    return {
        "trials": trials,
        "success": success / trials,
        "first_complete": met_first / trials,
        "second_complete": met_second / trials,
        "item_prob": {k: [c / trials for c in v] for k, v in item_met.items()},
        "expected_pulls_used": total_used / trials,
        "outcome_hist": outcome_hist,
    }


# ============================================================================
# 七、结果输出
# ============================================================================


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _targets_text(banner: Banner, targets: Sequence[int]) -> str:
    parts = [f"{banner.up_names[i]}x{targets[i]}" for i in range(banner.n_up)]
    return "  ".join(parts)


def _disp_width(text: str) -> int:
    """显示宽度：全角/宽字符算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    padding = " " * max(0, width - _disp_width(text))
    return text + padding if align == "left" else padding + text


def _pool_actual(outcome: OrderOutcome, key: str) -> float:
    """在某个顺序下，某个池子「最终真的完成了计划」的概率。"""
    if outcome.first.banner.key == key:
        return outcome.first_complete
    return outcome.second_complete


# --------------------------------------------------------------------------- #
# 成功率随抽数变化：表格 / 分位点 / CSV / SVG
# --------------------------------------------------------------------------- #

#: 「多少抽能有几成把握」里用到的把握度（低把握 = 最少要准备多少抽）
CONFIDENCE_LEVELS: Tuple[float, ...] = (0.10, 0.25, 0.30, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99)


def _nice_step(span: int, target_rows: int = 25) -> int:
    """把 span 分成不超过 target_rows 行，取 1/2/5/10/20/25/50/... 这类好看的步长。"""
    if span <= target_rows:
        return 1
    raw = span / target_rows
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if step >= raw:
            return step
    return int(raw) + 1


def format_success_table(
    curve: SuccessCurve, current_pulls: Optional[int] = None, bar_width: int = 50
) -> str:
    """成功率随抽数变化的柱状表（纯文本，不需要任何第三方库）。"""
    step = _nice_step(curve.max_pulls)
    rows: List[int] = sorted(set(range(0, curve.max_pulls + 1, step)) | {curve.max_pulls})
    if current_pulls is not None and 0 <= current_pulls <= curve.max_pulls:
        rows = sorted(set(rows) | {current_pulls})

    lines = [
        f"  抽数   成功率（在 N 抽内完成全部计划）",
        "  " + "-" * (bar_width + 14),
    ]
    for m in rows:
        p = curve.prob[m]
        filled = int(round(p * bar_width))
        bar = "#" * filled + "." * (bar_width - filled)
        mark = ""
        if current_pulls is not None and m == current_pulls:
            mark = "  <-- 当前抽数"
        lines.append(f"  {m:>4}  {p * 100:>7.2f}% |{bar}|{mark}")
    return "\n".join(lines)


def format_confidence(curve: SuccessCurve, current_pulls: Optional[int] = None) -> str:
    """「想有几成把握，需要多少抽」的汇总。

    低把握（10%/25%/30%）回答的是「最少要准备多少抽才不至于空手而归」，
    高把握（75% 以上）回答的是「要留多少余量才稳」。
    """
    lines = ["  " + _pad("把握", 8) + _pad("所需抽数", 12, "right")
             + _pad("该抽数下的成功率", 20, "right") + "   说明"]
    for level in CONFIDENCE_LEVELS:
        need = curve.needed_for(level)
        note = "最少" if level <= 0.30 else ("稳" if level >= 0.75 else "")
        if need is None:
            lines.append("  " + _pad(f"{level:.0%}", 8) + _pad(f">{curve.max_pulls}", 12, "right")
                         + _pad("（超出计算范围）", 20, "right") + f"   {note}")
        else:
            lines.append("  " + _pad(f"{level:.0%}", 8) + _pad(str(need), 12, "right")
                         + _pad(_pct(curve.prob[need]), 20, "right") + f"   {note}")
    if current_pulls is not None and 0 <= current_pulls <= curve.max_pulls:
        lines.append("")
        lines.append(f"  当前 {current_pulls} 抽的成功率 : {_pct(curve.prob[current_pulls])}")
        gains = curve.gains(current_pulls)
        if gains:
            parts = [f"再抽 {step} 抽 → {_pct(p)}" for step, p in gains]
            lines.append("  追加投入的效果        : " + "，".join(parts))
    return "\n".join(lines)


def write_csv_curve(curve: SuccessCurve, path: str, current_pulls: Optional[int] = None) -> str:
    """把成功率曲线导出成 CSV（utf-8-sig，Excel 直接打开不乱码）。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("pulls,success_probability,is_current\n")
        for m, p in curve.as_pairs():
            flag = 1 if (current_pulls is not None and m == current_pulls) else 0
            fh.write(f"{m},{p:.10f},{flag}\n")
    return path


def write_svg_curve(
    curve: SuccessCurve,
    path: str,
    current_pulls: Optional[int] = None,
    title: str = "达成抽卡计划的成功率 随抽数变化",
) -> str:
    """把成功率曲线画成 SVG 折线图（纯字符串拼接，不需要 matplotlib）。"""
    width, height = 980, 540
    left, right, top, bottom = 96, 46, 64, 74
    pw, ph = width - left - right, height - top - bottom
    span = max(curve.max_pulls, 1)

    def sx(m: int) -> float:
        return left + m / span * pw

    def sy(p: float) -> float:
        return top + (1.0 - p) * ph

    out: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="34" font-size="20" text-anchor="middle" fill="#222">'
        f'{_xml_escape(title)}</text>',
    ]

    # Y 轴网格与刻度
    for i in range(11):
        p = i / 10
        y = sy(p)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" '
                   f'stroke="{"#cccccc" if i else "#666666"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="13" text-anchor="end" '
                   f'fill="#444">{p * 100:.0f}%</text>')

    # X 轴刻度
    step = _nice_step(curve.max_pulls, 20)
    ticks = sorted(set(range(0, curve.max_pulls + 1, step)) | {curve.max_pulls})
    for m in ticks:
        x = sx(m)
        out.append(f'<line x1="{x:.1f}" y1="{top + ph}" x2="{x:.1f}" y2="{top + ph + 6}" '
                   f'stroke="#666666" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + ph + 24}" font-size="13" text-anchor="middle" '
                   f'fill="#444">{m}</text>')
    out.append(f'<text x="{left + pw / 2:.0f}" y="{height - 22}" font-size="14" '
               f'text-anchor="middle" fill="#333">可用抽数 N</text>')
    out.append(f'<text x="24" y="{top + ph / 2:.0f}" font-size="14" text-anchor="middle" '
               f'fill="#333" transform="rotate(-90 24 {top + ph / 2:.0f})">全部计划完成概率</text>')

    # 曲线
    points = " ".join(f"{sx(m):.1f},{sy(p):.1f}" for m, p in curve.as_pairs())
    out.append(f'<polyline points="{points}" fill="none" stroke="#1a73e8" stroke-width="2.4"/>')

    # 当前抽数
    if current_pulls is not None and 0 <= current_pulls <= curve.max_pulls:
        p = curve.prob[current_pulls]
        x, y = sx(current_pulls), sy(p)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ph}" '
                   f'stroke="#e8710a" stroke-width="1.8" stroke-dasharray="6 5"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#e8710a"/>')
        anchor, dx = ("start", 10) if current_pulls < curve.max_pulls * 0.72 else ("end", -10)
        out.append(f'<text x="{x + dx:.1f}" y="{y - 12:.1f}" font-size="14" text-anchor="{anchor}" '
                   f'fill="#b35c00">当前 {current_pulls} 抽: {p * 100:.2f}%</text>')

    # 50% 参考线
    y50 = sy(0.5)
    out.append(f'<line x1="{left}" y1="{y50:.1f}" x2="{left + pw}" y2="{y50:.1f}" '
               f'stroke="#999999" stroke-width="1" stroke-dasharray="3 4"/>')
    out.append('</svg>')

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return path


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------- #
# 花光抽数后的「结果分布」：文本表 / CSV / SVG
# --------------------------------------------------------------------------- #


def _outcome_text(char_counts: Sequence[int], weapon_counts: Sequence[int]) -> str:
    return ("角色 " + "/".join(str(c) for c in char_counts)
            + "  武器 " + "/".join(str(c) for c in weapon_counts))


def _gap_of(plan: Plan, char_counts: Sequence[int], weapon_counts: Sequence[int]) -> int:
    gap = sum(t - c for t, c in zip(plan.char_targets, char_counts))
    gap += sum(t - c for t, c in zip(plan.weapon_targets, weapon_counts))
    return max(0, gap)


def format_outcome_distribution(
    dist: OutcomeDistribution, top_k: int = 12, bar_width: int = 26
) -> str:
    """结果分布表：按出现概率从高到低排列（最可能的结果在最上面）。"""
    plan = dist.plan
    if not dist.entries:
        return "  （没有任何可能出现的结果）"
    max_p = dist.entries[0][2] or 1.0
    lines: List[str] = []
    lines.append(f"【花光全部 {plan.pulls} 抽后的结果分布】"
                 f"按出现概率从高到低，共 {dist.total_outcomes} 种可能结果")
    slots_char = "/".join(chr(ord("a") + i) for i in range(CHARACTER.n_up))
    slots_weapon = "/".join(chr(ord("a") + i) for i in range(WEAPON.n_up))
    lines.append(f"  结果写法：角色 {slots_char} = UP角色1~{CHARACTER.n_up} 的最终数量；"
                 f"武器 {slots_weapon} = UP武器1~{WEAPON.n_up} 的最终数量")
    lines.append(f"  本次目标：角色 {'/'.join(str(t) for t in plan.char_targets)}、"
                 f"武器 {'/'.join(str(t) for t in plan.weapon_targets)}")
    header = ("  " + _pad("排名", 6, "right") + _pad("概率", 10, "right")
              + _pad("累计", 10, "right") + "  " + _pad("概率条", bar_width + 2)
              + "  " + _pad("最终结果", 30) + _pad("缺口", 12))
    lines.append(header)
    lines.append("  " + "-" * (_disp_width(header) - 2))

    shown = dist.top(top_k)
    cumulative = 0.0
    for rank, (char_counts, weapon_counts, p) in enumerate(shown, start=1):
        cumulative += p
        gap = _gap_of(plan, char_counts, weapon_counts)
        filled = int(round(p / max_p * bar_width))
        bar = "|" + "#" * filled + "." * (bar_width - filled) + "|"
        gap_text = "全部达成" if gap == 0 else f"还差 {gap} 个"
        lines.append("  " + _pad(str(rank), 6, "right") + _pad(_pct(p), 10, "right")
                     + _pad(_pct(cumulative), 10, "right") + "  " + _pad(bar, bar_width + 2)
                     + "  " + _pad(_outcome_text(char_counts, weapon_counts), 30)
                     + _pad(gap_text, 12))
    rest_count, rest_p = dist.omitted(top_k)
    if rest_count:
        lines.append(f"  （其余 {rest_count} 种结果合计 {_pct(rest_p)}）")
    lines.append(f"  分布条按最高概率 {_pct(max_p)} 等比缩放。")

    lines.append("")
    lines.append("【缺口分布】距离「全部达成」还差几个目标物品")
    gap_bar = 30
    for gap, p in enumerate(dist.gap_probs):
        if p <= 0.0:
            continue
        filled = int(round(p * gap_bar))
        bar = "|" + "#" * filled + "." * (gap_bar - filled) + "|"
        label = "全部达成" if gap == 0 else f"还差 {gap} 个"
        lines.append(f"  {_pad(label, 12)}{_pct(p):>9}  {bar}")
    lines.append(f"  平均还差 {dist.expected_gap:.3f} 个目标物品"
                 f"（合计目标 {dist.total_targets} 个）")
    return "\n".join(lines)


def write_csv_outcome(dist: OutcomeDistribution, path: str) -> str:
    """把完整的结果分布导出成 CSV（按概率从高到低）。"""
    plan = dist.plan
    char_cols = [f"char_up{i + 1}" for i in range(CHARACTER.n_up)]
    weapon_cols = [f"weapon_up{i + 1}" for i in range(WEAPON.n_up)]
    header = ["rank", "probability", "cumulative"] + char_cols + weapon_cols \
             + ["gap", "all_completed"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(",".join(header) + "\n")
        cumulative = 0.0
        for rank, (char_counts, weapon_counts, p) in enumerate(dist.entries, start=1):
            cumulative += p
            gap = _gap_of(plan, char_counts, weapon_counts)
            row = [str(rank), f"{p:.10f}", f"{cumulative:.10f}"]
            row += [str(c) for c in char_counts]
            row += [str(c) for c in weapon_counts]
            row += [str(gap), "1" if gap == 0 else "0"]
            fh.write(",".join(row) + "\n")
    return path


def write_svg_outcome(
    dist: OutcomeDistribution,
    path: str,
    top_k: int = 12,
    title: Optional[str] = None,
) -> str:
    """把结果分布画成横向柱状图（最可能的结果在最上面）。"""
    shown = dist.top(top_k)
    if not shown:
        shown = []
    row_h = 30
    width = 1000
    left, right, top = 300, 210, 78
    height = top + max(len(shown), 1) * row_h + 46
    max_p = shown[0][2] if shown else 1.0
    max_p = max_p or 1.0
    bar_max = width - left - right
    plan = dist.plan

    if title is None:
        title = (f"花光 {plan.pulls} 抽后的结果分布（按概率从高到低，前 {len(shown)} 种）")

    out: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="34" font-size="20" text-anchor="middle" fill="#222">'
        f'{_xml_escape(title)}</text>',
        f'<text x="{width / 2:.0f}" y="58" font-size="13" text-anchor="middle" fill="#666">'
        f'目标：角色 {_xml_escape("/".join(str(t) for t in plan.char_targets))}'
        f'　武器 {_xml_escape("/".join(str(t) for t in plan.weapon_targets))}'
        f'　（达成整个计划的概率 {dist.success * 100:.2f}%）</text>',
    ]

    for i, (char_counts, weapon_counts, p) in enumerate(shown):
        y = top + i * row_h
        gap = _gap_of(plan, char_counts, weapon_counts)
        color = "#188038" if gap == 0 else "#1a73e8"
        w = max(1.0, p / max_p * bar_max)
        label = _outcome_text(char_counts, weapon_counts)
        out.append(f'<text x="{left - 12}" y="{y + 19}" font-size="13" text-anchor="end" '
                   f'fill="#333">{_xml_escape(label)}</text>')
        out.append(f'<rect x="{left}" y="{y + 5}" width="{w:.1f}" height="{row_h - 12}" '
                   f'fill="{color}" rx="2"/>')
        pct = f"{p * 100:.2f}%" + ("  ← 全部达成" if gap == 0 else f"  (还差 {gap})")
        out.append(f'<text x="{left + w + 8:.1f}" y="{y + 19}" font-size="13" '
                   f'fill="#444">{_xml_escape(pct)}</text>')

    out.append('</svg>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return path


def format_report(result: PlanResult, show_curve: bool = False, dist_top_k: int = 12) -> str:
    plan = result.plan
    chosen = result.chosen
    other = result.other
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append(" 原神四星抽卡概率计算器 · 马尔可夫链精确解")
    lines.append("=" * 78)

    lines.append("")
    lines.append("【输入参数】")
    lines.append(f"  抽卡顺序    : {chosen.label}")
    lines.append(f"  可用抽数    : {plan.pulls}")
    for key in ("character", "weapon"):
        banner = BANNERS[key]
        guarantee = "大保底中" if plan.guarantee_of(key) else "无大保底"
        lines.append(f"  {banner.name}      : 已垫 {plan.pity_of(key)} 抽 / {guarantee}")
        lines.append(f"                目标 -> {_targets_text(banner, plan.target_of(key))}")

    lines.append("")
    lines.append("【总体结论】")
    lines.append(f"  当前抽数 {plan.pulls} 抽的成功率   : {_pct(chosen.success)}"
                 f"   （把两池计划全部完成）")
    lines.append(f"  平均消耗抽数            : {chosen.expected_pulls_used:.2f} / {plan.pulls}"
                 f"（平均剩余 {plan.pulls - chosen.expected_pulls_used:.2f} 抽）")
    lines.append("  说明: 两池的四星保底互相独立，按优先池抽完再抽另一个，总消耗必然等于")
    lines.append("        「两池各自所需抽数之和」，所以全部完成的概率与平均消耗跟先抽哪个")
    lines.append("        池子无关；抽卡顺序只决定预算不够时先保住哪个池子（见下表）。")

    lines.append("")
    lines.append("【分池完成率】")
    lines.append("  " + _pad("池子", 12) + _pad(f"单独给满{plan.pulls}抽", 16, "right")
                 + _pad(chosen.label, 20, "right") + _pad(other.label, 20, "right"))
    for key in ("character", "weapon"):
        row = "  " + _pad(BANNERS[key].name, 12)
        row += _pad(_pct(result.curves[key].complete[plan.pulls]), 16, "right")
        row += _pad(_pct(_pool_actual(chosen, key)), 20, "right")
        row += _pad(_pct(_pool_actual(other, key)), 20, "right")
        lines.append(row)

    lines.append("")
    lines.append("【各 UP 物品达成概率】")
    lines.append("  " + _pad("物品", 12) + _pad("目标", 6, "right")
                 + _pad(chosen.label, 20, "right") + _pad(other.label, 20, "right")
                 + _pad("期望获得*", 12, "right"))
    for key in ("character", "weapon"):
        banner = BANNERS[key]
        targets = plan.target_of(key)
        for i in range(banner.n_up):
            row = "  " + _pad(banner.up_names[i], 12) + _pad(str(targets[i]), 6, "right")
            row += _pad(_pct(chosen.item_prob[key][i]), 20, "right")
            row += _pad(_pct(other.item_prob[key][i]), 20, "right")
            row += _pad(f"{chosen.item_expect[key][i]:.2f}", 12, "right")
            lines.append(row)
    lines.append(f"  * 期望获得按当前顺序（{chosen.label}）计算，并在目标数量处封顶。")
    lines.append(f"  * 角色池 {CHARACTER.n_up} 个 UP（不歪 {CHARACTER.up_rate:.0%}）、"
                 f"武器池 {WEAPON.n_up} 个 UP（不歪 {WEAPON.up_rate:.0%}）；"
                 f"非大保底时指定某一个的概率分别是 "
                 f"{CHARACTER.up_rate / CHARACTER.n_up:.2%} / "
                 f"{WEAPON.up_rate / WEAPON.n_up:.2%}。")

    if result.distribution is not None and dist_top_k > 0:
        lines.append("")
        lines.append(format_outcome_distribution(result.distribution, top_k=dist_top_k))

    if show_curve:
        lines.append("")
        lines.append(f"【成功率随抽数变化】P(在 N 抽内完成全部计划)，已算到 {result.curve.max_pulls} 抽")
        lines.append(format_success_table(result.curve, plan.pulls))

    lines.append("")
    lines.append(f"【达成把握与所需抽数】（曲线算到 {result.curve.max_pulls} 抽）")
    lines.append(format_confidence(result.curve, plan.pulls))

    lines.append("")
    lines.append("【模型说明】")
    for key in ("character", "weapon"):
        banner = BANNERS[key]
        lines.append(
            f"  {banner.name}: 概率表 {'/'.join(f'{p:.1%}' for p in banner.table)}"
            f"  ->  四星平均间隔 {mean_interval(banner.probs):.3f} 抽"
        )
        lines.append(
            f"      不歪概率 {banner.up_rate:.0%}，{banner.n_up} 个 UP 等概率随机"
            f"（非大保底时指定某一个 {banner.up_rate / banner.n_up:.2%}，"
            f"大保底时 {1 / banner.n_up:.2%}）"
        )
    lines.append("  >100% 按必出处理；出四星时先判「歪不歪」，歪了则下一个四星必为 UP。")
    lines.append("  抽卡顺序指「优先池」：优先池达标后才切到另一个池子，抽数用尽即结束。")
    lines.append("  模型只统计四星，不包含五星与四星的星级冲突（五星保底），如需五星请另行建模。")
    lines.append("=" * 78)
    return "\n".join(lines)


def result_to_dict(result: PlanResult, verify: Dict[str, object] | None = None) -> Dict[str, object]:
    plan = result.plan
    chosen = result.chosen
    other = result.other
    data: Dict[str, object] = {
        "input": {
            "order": plan.order,
            "pulls": plan.pulls,
            "character": {
                "pity": plan.char_pity,
                "guarantee": plan.char_guarantee,
                "targets": list(plan.char_targets),
            },
            "weapon": {
                "pity": plan.weapon_pity,
                "guarantee": plan.weapon_guarantee,
                "targets": list(plan.weapon_targets),
            },
        },
        "result": {
            # 注意：全部完成概率与顺序无关（两池独立，总消耗 = T角色 + T武器）
            "success": chosen.success,
            "success_chosen_order": chosen.success,
            "success_other_order": other.success,
            "expected_pulls_used": chosen.expected_pulls_used,
            "banner_complete_given_full_budget": {
                key: result.curves[key].complete[plan.pulls] for key in ("character", "weapon")
            },
            "banner_complete_by_order": {
                order: {
                    key: _pool_actual(outcome, key) for key in ("character", "weapon")
                }
                for order, outcome in result.outcomes.items()
            },
            "item_prob": chosen.item_prob,
            "item_prob_by_order": {
                order: outcome.item_prob for order, outcome in result.outcomes.items()
            },
            "item_expect": chosen.item_expect,
            # 成功率随抽数变化的函数：success_by_pulls[m] = 给 m 抽时的成功率
            "curve_max_pulls": result.curve.max_pulls,
            "success_by_pulls": result.curve.prob,
            "pulls_needed_for": {
                f"{level:.2f}": result.curve.needed_for(level)
                for level in CONFIDENCE_LEVELS
            },
        },
    }
    if result.distribution is not None:
        dist = result.distribution
        data["result"]["outcome_distribution"] = {
            "total_outcomes": dist.total_outcomes,
            "expected_gap": dist.expected_gap,
            "gap_probability": dist.gap_probs,
            "top_outcomes": [
                {
                    "rank": rank,
                    "probability": p,
                    "cumulative": sum(e[2] for e in dist.entries[:rank]),
                    "char_up": list(char_counts),
                    "weapon_up": list(weapon_counts),
                    "gap": _gap_of(dist.plan, char_counts, weapon_counts),
                }
                for rank, (char_counts, weapon_counts, p) in enumerate(dist.entries, start=1)
            ],
        }
    if verify is not None:
        data["monte_carlo"] = verify
    return data


# ============================================================================
# 八、命令行 / 交互式界面
# ============================================================================


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f"（默认 {default}）" if default is not None else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            raw = ""
        if not raw and default is not None:
            return default
        if raw:
            return raw
        print("  请输入内容。")


def _ask_int(prompt: str, default: int, lo: int = 0, hi: int = 10 ** 9) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            val = int(raw)
        except ValueError:
            print("  请输入整数。")
            continue
        if not lo <= val <= hi:
            print(f"  请输入 {lo} ~ {hi} 之间的整数。")
            continue
        return val


def _ask_yes_no(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    while True:
        raw = _ask(prompt + " [y/n]", d).lower()
        if raw in ("y", "yes", "是", "有", "1"):
            return True
        if raw in ("n", "no", "否", "没有", "0"):
            return False
        print("  请输入 y 或 n。")


def interactive_plan() -> Plan:
    print("=" * 60)
    print(" 原神四星抽卡概率计算器 · 交互式输入")
    print("=" * 60)
    order_raw = _ask("抽卡顺序 [1=先角色池后武器池, 2=先武器池后角色池]", "1")
    order = "weapon-first" if order_raw.strip() in ("2", "w", "武器", "weapon") else "character-first"
    pulls = _ask_int("当前可用抽数", 100, 0, 100000)
    char_pity = _ask_int(f"角色池已垫抽数（0~{CHARACTER.n_pity - 1}）", 0, 0, CHARACTER.n_pity - 1)
    weapon_pity = _ask_int(f"武器池已垫抽数（0~{WEAPON.n_pity - 1}）", 0, 0, WEAPON.n_pity - 1)
    char_guarantee = _ask_yes_no("角色池是否处于四星大保底", False)
    weapon_guarantee = _ask_yes_no("武器池是否处于四星大保底", False)

    print(f"\n-- 抽卡计划：想要几个 UP 角色（{CHARACTER.n_up} 个，0 表示不需要）--")
    char_targets = tuple(
        _ask_int(f"  {CHARACTER.up_names[i]} 数量", 0, 0, 999) for i in range(CHARACTER.n_up)
    )
    print(f"-- 抽卡计划：想要几个 UP 武器（{WEAPON.n_up} 个，0 表示不需要）--")
    weapon_targets = tuple(
        _ask_int(f"  {WEAPON.up_names[i]} 数量", 0, 0, 999) for i in range(WEAPON.n_up)
    )
    print()
    return Plan(
        order=order,
        pulls=pulls,
        char_pity=char_pity,
        char_guarantee=char_guarantee,
        weapon_pity=weapon_pity,
        weapon_guarantee=weapon_guarantee,
        char_targets=char_targets,      # type: ignore[arg-type]
        weapon_targets=weapon_targets,  # type: ignore[arg-type]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genshin_4star_calculator.py",
        description="原神四星抽卡概率计算器（马尔可夫链精确解）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python genshin_4star_calculator.py --order weapon --pulls 200 \\\n"
            "      --char-pity 3 --weapon-pity 5 --weapon-guarantee \\\n"
            "      --char-targets 2 1 0 --weapon-targets 1 0 1\n"
        ),
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="交互式输入参数")
    parser.add_argument(
        "--order",
        choices=["character-first", "weapon-first", "character", "weapon", "c", "w"],
        default="character-first",
        help="抽卡顺序：先抽哪个池子（c/character=先角色, w/weapon=先武器，默认 c）",
    )
    parser.add_argument("-p", "--pulls", type=int, help="当前可用抽数")
    parser.add_argument("--char-pity", type=int, default=0, help="角色池已垫抽数（0~9）")
    parser.add_argument("--weapon-pity", type=int, default=0, help="武器池已垫抽数（0~9）")
    parser.add_argument("--char-guarantee", action="store_true", help="角色池处于四星大保底")
    parser.add_argument("--no-char-guarantee", dest="char_guarantee", action="store_false",
                        help="角色池不是大保底（默认）")
    parser.add_argument("--weapon-guarantee", action="store_true", help="武器池处于四星大保底")
    parser.add_argument("--no-weapon-guarantee", dest="weapon_guarantee", action="store_false",
                        help="武器池不是大保底（默认）")
    parser.set_defaults(char_guarantee=False, weapon_guarantee=False)
    parser.add_argument("--char-targets", "-ct", type=int, nargs=CHARACTER.n_up,
                        metavar=tuple(f"C{i + 1}" for i in range(CHARACTER.n_up)),
                        default=(0,) * CHARACTER.n_up,
                        help=f"UP角色1~{CHARACTER.n_up} 各想要几个")
    parser.add_argument("--weapon-targets", "-wt", type=int, nargs=WEAPON.n_up,
                        metavar=tuple(f"W{i + 1}" for i in range(WEAPON.n_up)),
                        default=(0,) * WEAPON.n_up,
                        help=f"UP武器1~{WEAPON.n_up} 各想要几个")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument(
        "--curve", type=int, nargs="?", const=-1, default=None, metavar="MAX",
        help="打印「成功率随抽数变化」表，最多算到 MAX 抽（省略 MAX 时自动取 "
             "max(2×当前抽数, 当前抽数+60)）",
    )
    parser.add_argument("--csv", metavar="PATH", help="把成功率曲线导出为 CSV")
    parser.add_argument("--svg", metavar="PATH", help="把成功率曲线导出为 SVG 折线图")
    parser.add_argument(
        "--dist", type=int, nargs="?", const=12, default=None, metavar="K",
        help="打印「花光抽数后的结果分布」表（按概率从高到低显示前 K 种，默认 12）",
    )
    parser.add_argument("--dist-csv", metavar="PATH", help="把完整结果分布导出为 CSV")
    parser.add_argument("--dist-svg", metavar="PATH", help="把结果分布导出为 SVG 柱状图")
    parser.add_argument("--no-dist", action="store_true", help="不计算/不打印结果分布")
    parser.add_argument("--no-curve", action="store_true",
                        help="不把曲线算到当前抽数之外（更快）；若同时给了 --curve MAX，以 MAX 为准")
    parser.add_argument("--verify", type=int, nargs="?", const=20000, default=None,
                        metavar="TRIALS", help="额外跑 N 次蒙特卡洛模拟做对照（默认 20000 次）")
    parser.add_argument("--seed", type=int, default=None, help="蒙特卡洛随机种子")
    return parser


def default_curve_horizon(pulls: int) -> int:
    """默认把曲线算到多远：当前抽数 + max(60, 当前抽数/2)。"""
    return int(pulls) + max(60, int(pulls) // 2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interactive or (args.pulls is None and sys.stdin is not None and sys.stdin.isatty()):
        plan = interactive_plan()
    elif args.pulls is None:
        parser.error("请用 --pulls 指定抽数，或加 -i 进入交互式输入")
        return 2
    else:
        order = {
            "c": "character-first",
            "character": "character-first",
            "w": "weapon-first",
            "weapon": "weapon-first",
        }.get(args.order, args.order)
        plan = Plan(
            order=order,
            pulls=args.pulls,
            char_pity=args.char_pity,
            char_guarantee=args.char_guarantee,
            weapon_pity=args.weapon_pity,
            weapon_guarantee=args.weapon_guarantee,
            char_targets=tuple(args.char_targets),      # type: ignore[arg-type]
            weapon_targets=tuple(args.weapon_targets),  # type: ignore[arg-type]
        )

    total_states = 1
    for key in ("character", "weapon"):
        size = 1
        for t in plan.target_of(key):
            size *= t + 1
        total_states += size

    # 成功率曲线算到多远：显式 --curve MAX 最优先；--no-curve 只算到当前抽数；否则自动
    if args.curve is not None and args.curve >= 0:
        horizon = max(plan.pulls, args.curve)
    elif args.no_curve:
        horizon = plan.pulls
    else:
        horizon = default_curve_horizon(plan.pulls)

    if total_states * max(horizon, 1) > 40_000_000:
        print(f"提示：目标数量或抽数较大（状态格子约 {total_states} 个、曲线算到 {horizon} 抽），"
              f"计算可能需要一点时间……", file=sys.stderr)

    want_dist = not args.no_dist
    dist_top_k = args.dist if (args.dist is not None and args.dist >= 0) else 12
    if args.no_dist and (args.dist is not None or args.dist_csv or args.dist_svg):
        want_dist = True          # 显式要求导出/打印时，--no-dist 让位

    result = solve(plan, max_pulls=horizon, want_distribution=want_dist)

    verify = None
    if args.verify is not None:
        verify = monte_carlo(plan, args.verify, args.seed)

    if args.json:
        print(json.dumps(result_to_dict(result, verify), ensure_ascii=False, indent=2))
        return 0

    show_dist_table = (args.dist is not None) or not args.no_dist
    print(format_report(result, show_curve=args.curve is not None,
                        dist_top_k=dist_top_k if show_dist_table else 0))

    written: List[str] = []
    if args.csv:
        written.append(write_csv_curve(result.curve, args.csv, plan.pulls))
    if args.svg:
        written.append(write_svg_curve(
            result.curve, args.svg, plan.pulls,
            title=f"达成抽卡计划的成功率（当前 {plan.pulls} 抽: "
                  f"{result.success_at_current_pulls * 100:.2f}%）",
        ))
    if args.dist_csv:
        assert result.distribution is not None
        written.append(write_csv_outcome(result.distribution, args.dist_csv))
    if args.dist_svg:
        assert result.distribution is not None
        written.append(write_svg_outcome(
            result.distribution, args.dist_svg, top_k=dist_top_k,
            title=f"花光 {plan.pulls} 抽后的结果分布（按概率从高到低，前 {dist_top_k} 种）",
        ))
    if written:
        print()
        print("【已导出的文件】")
        for path in written:
            print(f"  {path}")

    if verify is not None:
        chosen = result.chosen
        print()
        print(f"【蒙特卡洛对照】模拟 {verify['trials']} 次（当前顺序 {chosen.label}）")
        print(f"  {'指标':<22}{'马尔可夫链':>12}{'模拟':>12}{'差值':>10}")
        rows = [
            ("全部完成概率", chosen.success, float(verify["success"])),
            ("平均消耗抽数", chosen.expected_pulls_used, float(verify["expected_pulls_used"])),
        ]
        for key in ("character", "weapon"):
            banner = BANNERS[key]
            for i in range(banner.n_up):
                if plan.target_of(key)[i] == 0:
                    continue
                rows.append((
                    f"{banner.up_names[i]}达成率",
                    chosen.item_prob[key][i],
                    float(verify["item_prob"][key][i]),  # type: ignore[index]
                ))
        for name, exact_v, sim_v in rows:
            print(f"  {name:<22}{exact_v:>12.4f}{sim_v:>12.4f}{sim_v - exact_v:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
