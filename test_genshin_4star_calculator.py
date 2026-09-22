#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""genshin_4star_calculator.py 的测试：解析值校验 + 蒙特卡洛对照 + 界面冒烟测试。

运行方式::

    python test_genshin_4star_calculator.py      # 直接跑
    python -m pytest test_genshin_4star_calculator.py -q
"""

from __future__ import annotations

import builtins
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import genshin_4star_calculator as G

CHAR_MISS = 1.0 - 0.051          # 角色池单抽不出四星的概率
WEAPON_MISS = 1.0 - 0.060        # 武器池单抽不出四星的概率
# 从「刚出过四星」开始，一路都没出四星、直到最后一档的概率
# 角色池：第 1~8 抽 5.1%、第 9 抽 56.1% -> 9 抽都没出 = 0.949^8 * 0.439
CHAR_9_MISS = CHAR_MISS ** 8 * (1.0 - 0.561)
# 武器池：第 1~7 抽 6.0%、第 8 抽 66.0%   -> 8 抽都没出 = 0.94^7 * 0.34
WEAPON_8_MISS = WEAPON_MISS ** 7 * (1.0 - 0.660)

CHAR_N = G.CHARACTER.n_up        # 3 个 UP 角色
WEAPON_N = G.WEAPON.n_up         # 5 个 UP 武器
CHAR_PITY = G.CHARACTER.n_pity   # 10 档（第 10 抽必出）
WEAPON_PITY = G.WEAPON.n_pity    # 9 档（第 9 抽必出）


def wt(*values) -> tuple:
    """武器池目标（5 个），写测试时只写关心的前几个，后面补 0。"""
    assert len(values) <= WEAPON_N
    return tuple(values) + (0,) * (WEAPON_N - len(values))


def ct(*values) -> tuple:
    """角色池目标（3 个）。"""
    assert len(values) <= CHAR_N
    return tuple(values) + (0,) * (CHAR_N - len(values))


def all_targets(banner, value: int) -> tuple:
    """该池每个 UP 都要 value 个（用于只关心保底机制、不关心具体物品的测试）。"""
    return (value,) * banner.n_up


def stepped_chain(banner, targets, pulls, pity=0, guarantee=False):
    chain = G.FourStarChain(banner, targets, pity, guarantee)
    for _ in range(pulls):
        chain.step()
    return chain


class TestProbabilityTable(unittest.TestCase):
    """概率表本身。"""

    def test_table_contents(self):
        self.assertEqual(G.CHARACTER_TABLE, (0.051,) * 8 + (0.561, 1.071))
        self.assertEqual(G.WEAPON_TABLE, (0.060,) * 7 + (0.660, 1.260))
        # >100% 在运算中必须被夹到 100%
        self.assertEqual(G.CHARACTER.probs[-1], 1.0)
        self.assertEqual(G.WEAPON.probs[-1], 1.0)
        # 两个池子的保底档数不同：角色 10 抽必出、武器 9 抽必出
        self.assertEqual(G.CHARACTER.n_pity, CHAR_PITY)
        self.assertEqual(G.WEAPON.n_pity, WEAPON_PITY)
        self.assertEqual((CHAR_PITY, WEAPON_PITY), (10, 9))
        self.assertEqual(len(G.CHARACTER.probs), CHAR_PITY)
        self.assertEqual(G.MAX_PITY, CHAR_PITY)          # 最长的那个
        # 武器池第 8 抽是软保底（66%），第 9 抽必出
        self.assertAlmostEqual(G.WEAPON.probs[6], 0.06, places=12)
        self.assertAlmostEqual(G.WEAPON.probs[7], 0.66, places=12)
        self.assertEqual(G.WEAPON.probs[8], 1.0)

    def test_up_item_counts(self):
        # 角色池 3 个 UP 角色，武器池 5 个 UP 武器
        self.assertEqual(G.CHARACTER.n_up, G.CHARACTER_UP_ITEMS)
        self.assertEqual(G.WEAPON.n_up, G.WEAPON_UP_ITEMS)
        self.assertEqual(G.CHARACTER_UP_ITEMS, 3)
        self.assertEqual(G.WEAPON_UP_ITEMS, 5)
        self.assertEqual(len(G.WEAPON.up_names), 5)
        with self.assertRaises(ValueError):
            G.FourStarChain(G.WEAPON, (1, 1, 1), 0, False)      # 武器池要 5 个目标
        with self.assertRaises(ValueError):
            G.FourStarChain(G.CHARACTER, (1, 1, 1, 0, 0), 0, False)

    def test_up_rate_per_banner(self):
        # 「不歪」概率：角色池 50%、武器池 75%
        self.assertAlmostEqual(G.CHARACTER.up_rate, 0.50, places=12)
        self.assertAlmostEqual(G.WEAPON.up_rate, 0.75, places=12)
        self.assertAlmostEqual(G.CHARACTER.lose_rate, 0.50, places=12)
        self.assertAlmostEqual(G.WEAPON.lose_rate, 0.25, places=12)
        # 非大保底时「出四星且是 UP」的总概率
        self.assertAlmostEqual(G.WEAPON.probs[0] * G.WEAPON.up_rate, 0.06 * 0.75, places=12)
        # 未处于大保底时，歪的概率不因池子而混用
        for banner in (G.CHARACTER, G.WEAPON):
            chain = G.FourStarChain(banner, all_targets(banner, 1), 0, False)
            self.assertAlmostEqual(chain.up_rate, banner.up_rate, places=12)
            # 大保底状态的质量 = 出四星的概率 × 歪的概率
            chain.step()
            marg = chain.pity_guarantee_marginal()
            self.assertAlmostEqual(marg[0][1], banner.probs[0] * banner.lose_rate, places=12)

    def test_mean_interval(self):
        # 与「按定义逐项求和」对比
        def brute(probs):
            total = 0.0
            miss = 1.0
            for n in range(1, 200):
                q = probs[min(n - 1, len(probs) - 1)]
                total += n * miss * q
                miss *= 1.0 - q
                if miss < 1e-15:
                    break
            return total

        for banner in (G.CHARACTER, G.WEAPON):
            self.assertAlmostEqual(G.mean_interval(banner.probs),
                                   brute(banner.probs), places=10)
        self.assertAlmostEqual(G.mean_interval(G.CHARACTER.probs), 7.655, places=3)
        # 武器池第 8 抽就有软保底、第 9 抽必出，所以平均间隔更短
        self.assertAlmostEqual(G.mean_interval(G.WEAPON.probs), 6.728, places=3)
        self.assertLess(G.mean_interval(G.WEAPON.probs),
                        G.mean_interval(G.CHARACTER.probs))

    def test_mean_interval_matches_simulation(self):
        rng = random.Random(20240501)
        for banner, expected in ((G.CHARACTER, 7.655), (G.WEAPON, 6.728)):
            trials, total = 40000, 0
            for _ in range(trials):
                pulls = 0
                while True:
                    q = banner.probs[min(pulls, banner.n_pity - 1)]
                    pulls += 1
                    if rng.random() < q:
                        break
                total += pulls
            self.assertAlmostEqual(total / trials, expected, delta=0.08)


class TestPityMechanics(unittest.TestCase):
    """四星保底计数（垫抽）的解析校验。"""

    def test_no_four_star_probability(self):
        # 一路都没出四星：角色池到第 8 抽 = 0.949^8，武器池到第 7 抽 = 0.94^7
        char = stepped_chain(G.CHARACTER, all_targets(G.CHARACTER, 0), 8).pity_guarantee_marginal()
        self.assertAlmostEqual(char[8][0] + char[8][1], CHAR_MISS ** 8, places=12)
        weapon = stepped_chain(G.WEAPON, all_targets(G.WEAPON, 0), 7).pity_guarantee_marginal()
        self.assertAlmostEqual(weapon[7][0] + weapon[7][1], WEAPON_MISS ** 7, places=12)

    def test_soft_pity_pull(self):
        # 角色池第 9 抽 56.1%，武器池第 8 抽 66%
        char = stepped_chain(G.CHARACTER, all_targets(G.CHARACTER, 0), 9).pity_guarantee_marginal()
        self.assertAlmostEqual(char[9][0] + char[9][1], CHAR_9_MISS, places=12)
        weapon = stepped_chain(G.WEAPON, all_targets(G.WEAPON, 0), 8).pity_guarantee_marginal()
        self.assertAlmostEqual(weapon[8][0] + weapon[8][1], WEAPON_8_MISS, places=12)

    def test_guaranteed_pull(self):
        # 第 1 抽出四星、之后一路不出 -> 恰好停在最后一档
        for banner, last_miss in ((G.CHARACTER, CHAR_9_MISS), (G.WEAPON, WEAPON_8_MISS)):
            n = banner.n_pity
            marg = stepped_chain(banner, all_targets(banner, 0), n).pity_guarantee_marginal()
            self.assertAlmostEqual(marg[n - 1][0] + marg[n - 1][1],
                                   banner.probs[0] * last_miss, places=12)
            # 任何连续 n 抽里至少有一个四星：只可能停在 0..n-1，且概率和为 1
            self.assertAlmostEqual(sum(p + g for p, g in marg), 1.0, places=12)
            self.assertEqual(len(marg), n)

    def test_pity_is_capped_and_validated(self):
        # 角色池 10 档、武器池 9 档：垫到最后一档之后就不该再垫
        with self.assertRaises(ValueError):
            G.FourStarChain(G.CHARACTER, ct(1, 1, 1), CHAR_PITY, False)
        with self.assertRaises(ValueError):
            G.FourStarChain(G.WEAPON, wt(1), WEAPON_PITY, False)
        with self.assertRaises(ValueError):
            G.FourStarChain(G.CHARACTER, ct(-1), 0, False)
        # 边界值本身是合法的
        for banner, targets in ((G.CHARACTER, ct(1)), (G.WEAPON, wt(1))):
            G.FourStarChain(banner, targets, banner.n_pity - 1, False)


class TestSinglePullAnalytics(unittest.TestCase):
    """单抽 / 两抽的解析值。"""

    def test_first_pull_up_probability(self):
        # 5.1% 出四星，其中 50% 不歪，角色 UP 三选一
        chain = stepped_chain(G.CHARACTER, ct(1), 1)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 0.051 * 0.5 / 3, places=12)

    def test_first_pull_with_guarantee(self):
        chain = stepped_chain(G.CHARACTER, ct(1), 1, guarantee=True)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 0.051 / 3, places=12)

    def test_last_pull_of_pity(self):
        # 垫了 9 抽时下一抽必出四星
        chain = stepped_chain(G.CHARACTER, ct(1), 1, pity=9)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 0.5 / 3, places=12)
        chain = stepped_chain(G.CHARACTER, ct(1), 1, pity=9, guarantee=True)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 1.0 / 3, places=12)

    def test_weapon_first_pull(self):
        # 武器池：6% 出四星，其中 75% 不歪，武器 UP 五选一（每个 75%/5 = 15%）
        chain = stepped_chain(G.WEAPON, wt(0, 1), 1)
        self.assertAlmostEqual(chain.item_reached_probs()[1], 0.06 * 0.75 / 5, places=12)
        self.assertAlmostEqual(chain.item_reached_probs()[1], 0.009, places=12)

    def test_weapon_guarantee_and_last_pity(self):
        # 垫满 8 抽时必出四星：无大保底命中某把 UP 武器 75%/5，有大保底 1/5
        chain = stepped_chain(G.WEAPON, wt(1), 1, pity=WEAPON_PITY - 1)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 0.75 / 5, places=12)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 0.15, places=12)
        chain = stepped_chain(G.WEAPON, wt(1), 1, pity=WEAPON_PITY - 1, guarantee=True)
        self.assertAlmostEqual(chain.item_reached_probs()[0], 1.0 / 5, places=12)

    def test_losing_5050_then_guarantee(self):
        # 垫满 9 抽(第 1 抽必出四星), 目标只要 UP角色1:
        #   a) 第 1 抽直接命中 UP角色1               : 0.5 * 1/3
        #   b) 第 1 抽歪了 -> 第 2 抽四星必为 UP     : 0.5 * (1/3) * 0.051
        #   c) 第 1 抽中了 UP角色2/3 -> 第 1 抽保底清零,
        #      第 2 抽仍可能再出四星并命中 UP角色1   : 0.5 * (2/3) * 0.051 * (0.5/3)
        expected = 0.5 / 3 + 0.5 * 0.051 / 3 + 0.5 * (2 / 3) * 0.051 * (0.5 / 3)
        chain = stepped_chain(G.CHARACTER, ct(1), 2, pity=9)
        self.assertAlmostEqual(chain.item_reached_probs()[0], expected, places=12)
        self.assertAlmostEqual(expected, 0.178, places=12)

    def test_losing_75_then_guarantee_weapon(self):
        # 武器池同样的事，只是「不歪」概率是 75%、5 选 1
        p, n = G.WEAPON.up_rate, WEAPON_N
        #   a) 第 1 抽直接命中目标武器                 : p * 1/n
        #   b) 第 1 抽歪了 -> 第 2 抽四星必为 UP       : (1-p) * 0.06 * (1/n)
        #   c) 第 1 抽中了别的 UP 武器 -> 保底清零,
        #      第 2 抽仍可能再出四星并命中目标武器     : p*(n-1)/n * 0.06 * (p/n)
        expected = (p / n
                    + (1 - p) * 0.06 * (1 / n)
                    + p * (n - 1) / n * 0.06 * (p / n))
        chain = stepped_chain(G.WEAPON, wt(1), 2, pity=WEAPON_PITY - 1)
        self.assertAlmostEqual(chain.item_reached_probs()[0], expected, places=12)
        self.assertAlmostEqual(p, 0.75, places=12)

    def test_all_up_items_are_symmetric(self):
        for banner, n_up in ((G.CHARACTER, CHAR_N), (G.WEAPON, WEAPON_N)):
            chain = stepped_chain(banner, all_targets(banner, 1), 7)
            probs = chain.item_reached_probs()
            self.assertEqual(len(probs), n_up)
            for p in probs[1:]:
                self.assertAlmostEqual(probs[0], p, places=12)
            # 非 UP 的四星不算任何一个 UP，所以概率之和 <= 1
            self.assertLessEqual(sum(probs), 1.0)

    def test_zero_target_is_always_met(self):
        chain = stepped_chain(G.CHARACTER, all_targets(G.CHARACTER, 0), 3)
        self.assertEqual(chain.item_reached_probs(), [1.0] * CHAR_N)
        self.assertEqual(chain.item_expectations(), [0.0] * CHAR_N)
        self.assertAlmostEqual(chain.complete_prob(), 1.0, places=12)
        chain = stepped_chain(G.WEAPON, all_targets(G.WEAPON, 0), 3)
        self.assertEqual(chain.item_reached_probs(), [1.0] * WEAPON_N)

    def test_expectation_never_exceeds_target(self):
        for banner, targets in ((G.CHARACTER, ct(2, 1, 3)), (G.WEAPON, wt(2, 1, 0, 3, 1))):
            chain = stepped_chain(banner, targets, 120)
            for value, target in zip(chain.item_expectations(), targets):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, target + 1e-12)

    def test_guarantee_helps(self):
        for banner, targets in ((G.CHARACTER, ct(2, 2, 2)), (G.WEAPON, wt(2, 2, 2))):
            without = G.analyze_banner(banner, targets, 0, False, 60)
            with_g = G.analyze_banner(banner, targets, 0, True, 60)
            self.assertGreater(with_g.final, without.final)
            # 垫到最后一档（下一抽必出四星）比没垫更容易完成
            earlier = G.analyze_banner(banner, targets, banner.n_pity - 1, False, 60)
            later = G.analyze_banner(banner, targets, 0, False, 60)
            self.assertGreater(earlier.final, later.final)


class TestCurveStructure(unittest.TestCase):
    """曲线/分布的结构性质。"""

    def test_analyze_matches_step_by_step(self):
        curve = G.analyze_banner(G.WEAPON, wt(2, 1, 1), 3, True, 25)
        for k in (0, 1, 5, 12, 25):
            ref = stepped_chain(G.WEAPON, wt(2, 1, 1), k, 3, True)
            self.assertAlmostEqual(curve.complete[k], ref.complete_prob(), places=12)
            self.assertAlmostEqual(curve.item_prob[k][0], ref.item_reached_probs()[0],
                                   places=12)

    def test_no_state_leaks_between_runs(self):
        a = G.analyze_banner(G.CHARACTER, ct(2, 1), 4, False, 30)
        G.analyze_banner(G.WEAPON, all_targets(G.WEAPON, 3), WEAPON_PITY - 1, True, 40)   # 中间跑别的
        b = G.analyze_banner(G.CHARACTER, ct(2, 1), 4, False, 30)
        self.assertEqual(a.complete, b.complete)

    def test_monotone_and_bounded(self):
        for banner, targets in ((G.CHARACTER, ct(3, 2, 1)), (G.WEAPON, wt(2, 2, 1, 1))):
            curve = G.analyze_banner(banner, targets, 5, False, 80)
            self.assertAlmostEqual(curve.complete[0], 0.0, places=12)
            for k in range(1, len(curve.complete)):
                self.assertGreaterEqual(curve.complete[k] + 1e-15, curve.complete[k - 1])
                self.assertLessEqual(curve.complete[k], 1.0)
                for i in range(banner.n_up):
                    self.assertGreaterEqual(curve.item_prob[k][i],
                                            curve.item_prob[k - 1][i] - 1e-15)
                    self.assertLessEqual(curve.item_prob[k][i], 1.0)

    def test_exact_completion_sums_to_final(self):
        for curve in (G.analyze_banner(G.CHARACTER, ct(2, 2, 2), 0, False, 70),
                      G.analyze_banner(G.WEAPON, wt(1, 1, 1, 1, 1), WEAPON_PITY - 1, True, 70)):
            exact = curve.exact_completion()
            self.assertTrue(all(v >= -1e-15 for v in exact))
            self.assertAlmostEqual(sum(exact), curve.final, places=12)

    def test_expected_pulls_used(self):
        curve = G.analyze_banner(G.CHARACTER, ct(1), 0, False, 50)
        # 手算: E[min(T, 50)] = Σ_{j<50} P(T > j)
        manual = sum(1.0 - v for v in curve.complete[:-1])
        self.assertAlmostEqual(curve.expected_pulls_used(), manual, places=12)
        self.assertLessEqual(curve.expected_pulls_used(), 50.0)
        # 目标为 0 时不消耗抽数（允许浮点残差）
        self.assertAlmostEqual(
            G.analyze_banner(G.CHARACTER, all_targets(G.CHARACTER, 0), 0, False, 50).expected_pulls_used(),
            0.0, places=9)

    def test_budget_zero(self):
        curve = G.analyze_banner(G.CHARACTER, ct(1), 0, False, 0)
        self.assertEqual(curve.complete, [0.0])
        self.assertEqual(curve.item_prob, [[0.0, 1.0, 1.0]])   # 目标 0 视为已达成
        self.assertEqual(curve.expected_pulls_used(), 0.0)
        # 武器池同理，只是有 5 个 UP
        curve = G.analyze_banner(G.WEAPON, wt(1), 0, False, 0)
        self.assertEqual(curve.item_prob, [[0.0, 1.0, 1.0, 1.0, 1.0]])


class TestOrderCombination(unittest.TestCase):
    """两个池子按顺序组合。"""

    @staticmethod
    def make_plan(order="weapon-first", pulls=60):
        return G.Plan(
            order=order, pulls=pulls,
            char_pity=2, char_guarantee=False,
            weapon_pity=5, weapon_guarantee=True,
            char_targets=(2, 1, 0), weapon_targets=wt(1, 1, 0),
        )

    def test_success_equals_total_convolution(self):
        plan = self.make_plan()
        res = G.solve(plan)
        c, w, n = res.curves["character"], res.curves["weapon"], plan.pulls
        ec, ew = c.exact_completion(), w.exact_completion()
        # 直接枚举「角色池用时 + 武器池用时 <= N」，不区分顺序
        total = 0.0
        for k in range(n + 1):
            if ec[k] == 0.0:
                continue
            for j in range(n + 1 - k):
                total += ec[k] * ew[j]
        self.assertAlmostEqual(res.chosen.success, total, places=12)
        self.assertAlmostEqual(res.other.success, total, places=12)

    def test_total_is_order_independent(self):
        plan = self.make_plan(pulls=25)
        res = G.solve(plan)
        self.assertAlmostEqual(res.outcomes["weapon-first"].success,
                               res.outcomes["character-first"].success, places=12)
        self.assertAlmostEqual(res.outcomes["weapon-first"].expected_pulls_used,
                               res.outcomes["character-first"].expected_pulls_used, places=10)

    def test_first_pool_always_gets_full_budget(self):
        plan = self.make_plan(pulls=25)
        res = G.solve(plan)
        self.assertAlmostEqual(res.outcomes["weapon-first"].first_complete,
                               res.curves["weapon"].complete[25], places=12)
        self.assertAlmostEqual(res.outcomes["character-first"].first_complete,
                               res.curves["character"].complete[25], places=12)

    def test_second_pool_is_never_better_off(self):
        # 先抽的池子至少和后抽的一样容易完成
        plan = self.make_plan(pulls=25)
        res = G.solve(plan)
        for key in ("character", "weapon"):
            first = G._pool_actual(res.outcomes[f"{key}-first"], key)
            second = G._pool_actual(res.outcomes["weapon-first" if key == "character"
                                                 else "character-first"], key)
            self.assertGreaterEqual(first + 1e-12, second)

    def test_empty_plan_never_fails(self):
        plan = G.Plan("weapon-first", 40, 0, False, 0, False, ct(0, 0, 0), wt(0, 0, 0))
        res = G.solve(plan)
        self.assertAlmostEqual(res.chosen.success, 1.0, places=12)
        self.assertAlmostEqual(res.chosen.expected_pulls_used, 0.0, places=9)

    def test_only_one_pool_needed(self):
        # 武器池目标为 0，成功率应该等于角色池单独完成率
        plan = G.Plan("weapon-first", 80, 3, False, 0, False, ct(2, 1, 0), wt(0, 0, 0))
        res = G.solve(plan)
        self.assertAlmostEqual(res.chosen.success, res.curves["character"].final, places=12)
        # 先角色池：角色池拿满预算，结果相同
        plan2 = G.Plan("character-first", 80, 3, False, 0, False, ct(2, 1, 0), wt(0, 0, 0))
        self.assertAlmostEqual(G.solve(plan2).chosen.success, res.chosen.success, places=12)

    def test_item_prob_of_priority_pool(self):
        # 预算不足时，先抽的池子的物品达成率应当更高
        plan = self.make_plan(pulls=25)
        res = G.solve(plan)
        self.assertGreater(res.outcomes["weapon-first"].item_prob["weapon"][0],
                           res.outcomes["character-first"].item_prob["weapon"][0])

    def test_more_pulls_never_hurt(self):
        prev = -1.0
        for pulls in (0, 10, 20, 40, 80, 160):
            plan = self.make_plan(pulls=pulls)
            val = G.solve(plan).chosen.success
            self.assertGreaterEqual(val + 1e-12, prev)
            prev = val
        self.assertAlmostEqual(prev, G.solve(self.make_plan(pulls=160)).chosen.success)


class TestSuccessCurve(unittest.TestCase):
    """成功率随抽数变化的函数。"""

    @staticmethod
    def make_plan(order="weapon-first", pulls=60):
        return G.Plan(
            order=order, pulls=pulls,
            char_pity=2, char_guarantee=False,
            weapon_pity=5, weapon_guarantee=True,
            char_targets=(2, 1, 0), weapon_targets=wt(1, 1, 0),
        )

    def test_curve_matches_independent_convolution(self):
        plan = self.make_plan(pulls=60)
        res = G.solve(plan, max_pulls=140)
        curve = res.curve
        self.assertEqual(curve.max_pulls, 140)
        self.assertEqual(len(curve.prob), 141)
        c, w = res.curves["character"], res.curves["weapon"]
        ec = c.exact_completion()
        for m in (0, 1, 7, 30, 60, 99, 140):
            manual = 0.0
            for k in range(m + 1):                 # 角色池用时 k + 武器池用时 <= m-k
                manual += ec[k] * w.complete[m - k]
            self.assertAlmostEqual(curve.prob[m], manual, places=12)

    def test_curve_value_at_current_pulls_equals_reported_success(self):
        for order in ("weapon-first", "character-first"):
            plan = self.make_plan(order=order, pulls=60)
            res = G.solve(plan, max_pulls=100)
            self.assertAlmostEqual(res.curve.at(60), res.chosen.success, places=12)
            self.assertAlmostEqual(res.success_at_current_pulls, res.chosen.success, places=12)

    def test_curve_is_monotone_and_bounded(self):
        res = G.solve(self.make_plan(pulls=40), max_pulls=120)
        prob = res.curve.prob
        self.assertGreaterEqual(prob[0], 0.0)
        self.assertLessEqual(prob[-1], 1.0)
        for m in range(1, len(prob)):
            self.assertGreaterEqual(prob[m] + 1e-12, prob[m - 1])

    def test_horizon_does_not_change_the_answer_at_the_budget(self):
        plan = self.make_plan(pulls=45)
        short = G.solve(plan)
        long = G.solve(plan, max_pulls=200)
        self.assertAlmostEqual(short.chosen.success, long.chosen.success, places=12)
        self.assertAlmostEqual(short.chosen.expected_pulls_used,
                               long.chosen.expected_pulls_used, places=10)
        for key in ("character", "weapon"):
            self.assertAlmostEqual(short.curves[key].complete[45],
                                   long.curves[key].complete[45], places=12)
            for i in range(3):
                self.assertAlmostEqual(short.chosen.item_prob[key][i],
                                       long.chosen.item_prob[key][i], places=12)
        # 短曲线是长曲线的前缀
        for m in range(46):
            self.assertAlmostEqual(short.curve.prob[m], long.curve.prob[m], places=12)

    def test_at_and_needed_for(self):
        res = G.solve(self.make_plan(pulls=50), max_pulls=150)
        curve = res.curve
        with self.assertRaises(ValueError):
            curve.at(-1)
        with self.assertRaises(ValueError):
            curve.at(151)
        self.assertEqual(curve.at(0), curve.prob[0])
        need = curve.needed_for(0.5)
        self.assertIsNotNone(need)
        self.assertGreaterEqual(curve.prob[need], 0.5)
        self.assertLess(curve.prob[need - 1], 0.5)
        self.assertIsNone(curve.needed_for(1.0))        # 不可能 100%
        self.assertIsNone(curve.needed_for(0.999999))

    def test_gains_and_pairs(self):
        res = G.solve(self.make_plan(pulls=50), max_pulls=70)
        curve = res.curve
        gains = curve.gains(50, steps=(10, 20, 30, 50))
        self.assertEqual([s for s, _ in gains], [10, 20])   # 50+30 / 50+50 超范围被跳过
        for step, p in gains:
            self.assertAlmostEqual(p, curve.prob[50 + step], places=12)
        pairs = curve.as_pairs()
        self.assertEqual(pairs[0], (0, curve.prob[0]))
        self.assertEqual(len(pairs), 71)

    def test_confidence_levels(self):
        # 把握档位：10% / 25% / 30% / 50% / 75% / 80% / 90% / 95% / 99%
        self.assertEqual(G.CONFIDENCE_LEVELS,
                         (0.10, 0.25, 0.30, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99))
        res = G.solve(self.make_plan(pulls=50), max_pulls=300)
        curve = res.curve
        text = G.format_confidence(curve, 50)
        for level in G.CONFIDENCE_LEVELS:                  # 每一档都要出现在表里
            self.assertIn(f"{level:.0%}", text)
        # needed_for 返回「第一次达到该把握」的抽数，且抽数随把握单调不减
        needs = []
        for level in G.CONFIDENCE_LEVELS:
            need = curve.needed_for(level)
            needs.append(need)
            if need is not None:
                self.assertGreaterEqual(curve.prob[need], level - 1e-12)
                if need:
                    self.assertLess(curve.prob[need - 1], level)
        reachable = [n for n in needs if n is not None]
        self.assertEqual(reachable, sorted(reachable))
        # 低把握需要的抽数最少：10% ≤ 25% ≤ 30% ≤ 50% ≤ …
        for low, high in ((0.10, 0.25), (0.25, 0.30), (0.30, 0.50),
                          (0.50, 0.75), (0.75, 0.80), (0.80, 0.90)):
            a, b = curve.needed_for(low), curve.needed_for(high)
            if a is not None and b is not None:
                self.assertLessEqual(a, b)
        # 10% 把握需要的抽数应该明显少于 50%
        self.assertLess(curve.needed_for(0.10), curve.needed_for(0.50))

    def test_trivial_plans(self):
        # 什么都不需要 -> 永远 100%
        plan = G.Plan("weapon-first", 30, 0, False, 0, False, ct(0, 0, 0), wt(0, 0, 0))
        curve = G.solve(plan, max_pulls=40).curve
        self.assertTrue(all(abs(p - 1.0) < 1e-12 for p in curve.prob))
        self.assertEqual(curve.needed_for(0.99), 0)
        # 有目标但 0 抽 -> 0%；曲线本身仍可算到更远的抽数
        plan = G.Plan("weapon-first", 0, 0, False, 0, False, ct(1, 0, 0), wt(0, 0, 0))
        curve = G.solve(plan, max_pulls=20).curve
        self.assertEqual(curve.prob[0], 0.0)
        self.assertEqual(curve.at(0), 0.0)
        self.assertGreater(curve.at(20), 0.0)
        # 曲线在 N=20 处的值 == 直接以 20 抽为预算求解的成功率
        plan20 = G.Plan("weapon-first", 20, 0, False, 0, False, ct(1, 0, 0), wt(0, 0, 0))
        self.assertAlmostEqual(curve.at(20), G.solve(plan20).chosen.success, places=12)


class TestCurveExport(unittest.TestCase):
    """曲线导出：文本表 / CSV / SVG。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.plan_args = ["--pulls", "60", "--char-targets", "2", "1", "0",
                          "--weapon-targets", "1", "0", "1", "0", "0", "--curve", "120"]

    def tearDown(self):
        self._tmp.cleanup()

    def test_table_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(self.plan_args)
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("成功率随抽数变化", out)
        self.assertIn("<-- 当前抽数", out)
        self.assertIn("达成把握与所需抽数", out)
        self.assertIn("追加投入的效果", out)
        for level in ("10%", "25%", "30%", "50%", "75%", "80%", "90%", "95%", "99%"):
            self.assertIn(level, out)

    def test_no_curve_is_faster_and_shorter(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            G.main(["--pulls", "60", "-ct", "1", "0", "0", "--no-curve"])
        out = buf.getvalue()
        self.assertNotIn("成功率随抽数变化", out)
        self.assertIn("（曲线算到 60 抽）", out)

    def test_csv_export(self):
        path = os.path.join(self.tmp, "curve.csv")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(self.plan_args + ["--csv", path])
        self.assertEqual(code, 0)
        with open(path, "r", encoding="utf-8-sig") as fh:
            rows = [line.strip().split(",") for line in fh if line.strip()]
        self.assertEqual(rows[0], ["pulls", "success_probability", "is_current"])
        self.assertEqual(len(rows), 122)                    # 0..120 + 表头
        self.assertEqual(rows[1][0], "0")
        self.assertEqual(rows[-1][0], "120")
        values = [float(r[1]) for r in rows[1:]]
        self.assertEqual(values, sorted(values))
        self.assertEqual(sum(int(r[2]) for r in rows[1:]), 1)  # 只有一个当前抽数

    def test_svg_export(self):
        path = os.path.join(self.tmp, "curve.svg")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(self.plan_args + ["--svg", path])
        self.assertEqual(code, 0)
        with open(path, "r", encoding="utf-8") as fh:
            svg = fh.read()
        self.assertTrue(svg.startswith("<?xml"))
        self.assertIn("<polyline", svg)
        self.assertIn("当前 60 抽", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("已导出的文件", buf.getvalue())
        # XML 必须是良构的，折线点数 = 曲线长度
        import xml.etree.ElementTree as ET
        root = ET.fromstring(svg)
        polyline = root.find("{http://www.w3.org/2000/svg}polyline")
        self.assertIsNotNone(polyline)
        self.assertEqual(len(polyline.get("points").split()), 121)

    def test_json_contains_curve(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(["--pulls", "50", "-ct", "1", "0", "0", "--json",
                           "--curve", "80", "--seed", "1"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["result"]["curve_max_pulls"], 80)
        self.assertEqual(len(data["result"]["success_by_pulls"]), 81)
        self.assertAlmostEqual(data["result"]["success_by_pulls"][50],
                               data["result"]["success"], places=12)
        needed = data["result"]["pulls_needed_for"]
        for level in G.CONFIDENCE_LEVELS:                  # 10/25/30/50/75/80/90/95/99 都要有
            self.assertIn(f"{level:.2f}", needed)
        self.assertIn("0.50", needed)

    def test_distribution_table_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(["--pulls", "60", "-ct", "2", "1", "0",
                           "-wt", "1", "1", "0", "0", "0", "--dist", "5", "--no-curve"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("结果分布", out)
        self.assertIn("缺口分布", out)
        self.assertIn("全部达成", out)
        self.assertIn("平均还差", out)

    def test_no_dist_flag(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            G.main(["--pulls", "60", "-ct", "1", "0", "0", "--no-curve", "--no-dist"])
        self.assertNotIn("结果分布", buf.getvalue())

    def test_distribution_exports(self):
        csv_path = os.path.join(self.tmp, "outcome.csv")
        svg_path = os.path.join(self.tmp, "outcome.svg")
        args = ["--pulls", "70", "-ct", "2", "1", "0", "-wt", "1", "0", "1", "0", "0",
                "--no-curve", "--dist", "6", "--dist-csv", csv_path, "--dist-svg", svg_path]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(args)
        self.assertEqual(code, 0)
        with open(csv_path, "r", encoding="utf-8-sig") as fh:
            rows = [line.strip().split(",") for line in fh if line.strip()]
        self.assertEqual(rows[0][:3], ["rank", "probability", "cumulative"])
        # 表头必须包含全部 UP 列：角色 3 个 + 武器 5 个
        self.assertEqual(rows[0][3:6], ["char_up1", "char_up2", "char_up3"])
        self.assertEqual(rows[0][6:11],
                         ["weapon_up1", "weapon_up2", "weapon_up3", "weapon_up4", "weapon_up5"])
        self.assertEqual(rows[0][-2:], ["gap", "all_completed"])
        gap_index = len(rows[0]) - 2
        for row in rows[1:]:
            self.assertEqual(len(row), len(rows[0]))                   # 每行列数一致
        probs = [float(r[1]) for r in rows[1:]]
        self.assertEqual(probs, sorted(probs, reverse=True))          # 最可能的在最前
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertEqual([r[0] for r in rows[1:]], [str(i) for i in range(1, len(rows))])
        # 名次/累计自洽，且「全部达成」标记与缺口一致（缺口 0 的结果只有一个）
        cumulative = 0.0
        for row in rows[1:]:
            cumulative += float(row[1])
            self.assertAlmostEqual(float(row[2]), cumulative, places=6)
            gap = int(row[gap_index])
            self.assertEqual(int(row[gap_index + 1]), 1 if gap == 0 else 0)
        self.assertEqual(sum(1 for r in rows[1:] if int(r[gap_index]) == 0), 1)
        # 目标总数 = 角色 2+1+0 + 武器 1+0+1+0+0 = 5，所以 缺口 = 5 − 实际获得总数
        for row in rows[1:]:
            got = sum(int(x) for x in row[3:gap_index])
            self.assertEqual(int(row[gap_index]), 5 - got)
        with open(svg_path, "r", encoding="utf-8") as fh:
            svg = fh.read()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(svg)
        ns = "{http://www.w3.org/2000/svg}"
        self.assertEqual(len(root.findall(ns + "rect")) - 1, 6)        # 6 根柱子（另加背景）
        self.assertIn("结果分布", svg)
        self.assertIn("已导出的文件", buf.getvalue())

    def test_json_contains_distribution(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(["--pulls", "60", "-ct", "1", "1", "0", "-wt", "1", "0", "0", "0", "0",
                           "--json", "--no-curve"])
        self.assertEqual(code, 0)
        result = json.loads(buf.getvalue())["result"]
        data = result["outcome_distribution"]
        self.assertGreater(data["total_outcomes"], 1)
        self.assertAlmostEqual(sum(data["gap_probability"]), 1.0, places=9)
        self.assertAlmostEqual(data["gap_probability"][0], result["success"], places=12)
        top = data["top_outcomes"]
        self.assertEqual(top[0]["rank"], 1)
        # 排在第一的确实是概率最高的那个结果（不一定就是「全部达成」）
        self.assertEqual(top[0]["probability"], max(o["probability"] for o in top))
        self.assertTrue(all(top[i]["probability"] >= top[i + 1]["probability"]
                            for i in range(len(top) - 1)))
        self.assertEqual(sum(1 for o in top if o["gap"] == 0), 1)



class TestOutcomeDistribution(unittest.TestCase):
    """花光全部抽数后的结果分布。"""

    @staticmethod
    def make_plan(order="weapon-first", pulls=60, char_targets=(2, 1, 0),
                  weapon_targets=wt(1, 1, 0)):
        return G.Plan(
            order=order, pulls=pulls,
            char_pity=2, char_guarantee=False,
            weapon_pity=5, weapon_guarantee=True,
            char_targets=char_targets, weapon_targets=weapon_targets,
        )

    def solve_dist(self, **kwargs):
        plan = self.make_plan(**kwargs)
        return plan, G.solve(plan, want_distribution=True).distribution

    def test_distribution_is_a_probability_distribution(self):
        _, dist = self.solve_dist()
        self.assertAlmostEqual(sum(e[2] for e in dist.entries), 1.0, places=10)
        self.assertAlmostEqual(sum(dist.gap_probs), 1.0, places=10)

    def test_sorted_most_likely_first(self):
        _, dist = self.solve_dist()
        probs = [e[2] for e in dist.entries]
        self.assertEqual(probs, sorted(probs, reverse=True))
        # 并列时「拿到的更多」的在前
        for a, b in zip(dist.entries, dist.entries[1:]):
            if abs(a[2] - b[2]) < 1e-15:
                gap_a = G._gap_of(dist.plan, a[0], a[1])
                gap_b = G._gap_of(dist.plan, b[0], b[1])
                self.assertLessEqual(gap_a, gap_b)

    def test_success_equals_headline_probability(self):
        plan, dist = self.solve_dist()
        result = G.solve(plan)
        self.assertAlmostEqual(dist.success, result.chosen.success, places=12)
        self.assertAlmostEqual(dist.gap_probs[0], result.chosen.success, places=12)
        # 缺口为 0 的结果只有一个（就是两个池子都达标）
        zero_gap = [e for e in dist.entries
                    if G._gap_of(plan, e[0], e[1]) == 0]
        self.assertEqual(len(zero_gap), 1)

    def test_marginals_match_the_per_item_results(self):
        plan, dist = self.solve_dist()
        result = G.solve(plan)
        for key in ("character", "weapon"):
            for i in range(3):
                self.assertAlmostEqual(dist.item_marginal(key, i),
                                       result.chosen.item_prob[key][i], places=10)

    def test_expected_counts_match_the_per_item_expectations(self):
        plan, dist = self.solve_dist()
        result = G.solve(plan)
        totals = {"character": [0.0] * 3, "weapon": [0.0] * 3}
        for char_counts, weapon_counts, p in dist.entries:
            for i in range(3):
                totals["character"][i] += char_counts[i] * p
                totals["weapon"][i] += weapon_counts[i] * p
        for key in ("character", "weapon"):
            for i in range(3):
                self.assertAlmostEqual(totals[key][i], result.chosen.item_expect[key][i],
                                       places=10)

    def test_expected_gap(self):
        plan, dist = self.solve_dist()
        result = G.solve(plan)
        missing = (sum(plan.char_targets) + sum(plan.weapon_targets)
                   - sum(result.chosen.item_expect["character"])
                   - sum(result.chosen.item_expect["weapon"]))
        self.assertAlmostEqual(dist.expected_gap, missing, places=10)
        self.assertAlmostEqual(sum(g * p for g, p in enumerate(dist.gap_probs)),
                               dist.expected_gap, places=12)

    def test_counts_never_exceed_targets(self):
        plan, dist = self.solve_dist()
        for char_counts, weapon_counts, _ in dist.entries:
            for i in range(3):
                self.assertLessEqual(char_counts[i], plan.char_targets[i])
                self.assertLessEqual(weapon_counts[i], plan.weapon_targets[i])

    def test_order_changes_the_distribution(self):
        # 成功率与顺序无关，但结果分布与顺序有关：优先池更容易拿满
        p1, d1 = self.solve_dist(order="weapon-first")
        p2, d2 = self.solve_dist(order="character-first")
        weapon_full_first = sum(p for c, w, p in d1.entries if w == p1.weapon_targets)
        weapon_full_second = sum(p for c, w, p in d2.entries if w == p2.weapon_targets)
        self.assertGreater(weapon_full_first, weapon_full_second)
        char_full_first = sum(p for c, w, p in d1.entries if c == p1.char_targets)
        char_full_second = sum(p for c, w, p in d2.entries if c == p2.char_targets)
        self.assertLess(char_full_first, char_full_second)
        # 但两者的「全部达成」概率必须相同
        self.assertAlmostEqual(d1.success, d2.success, places=12)

    def test_trivial_plans(self):
        # 什么都不需要：只有一个结果，概率 1
        plan = G.Plan("weapon-first", 30, 0, False, 0, False, ct(0, 0, 0), wt(0, 0, 0))
        dist = G.solve(plan, want_distribution=True).distribution
        self.assertEqual(dist.total_outcomes, 1)
        self.assertEqual(dist.entries[0][0], (0, 0, 0))
        self.assertEqual(dist.entries[0][1], (0, 0, 0, 0, 0))
        self.assertAlmostEqual(dist.entries[0][2], 1.0, places=12)
        # 0 抽但有目标：只有一个「什么都没拿到」的结果
        plan = G.Plan("weapon-first", 0, 0, False, 0, False, ct(2), wt(1))
        dist = G.solve(plan, want_distribution=True).distribution
        self.assertEqual(dist.total_outcomes, 1)
        self.assertEqual(dist.entries[0][0], (0, 0, 0))
        self.assertAlmostEqual(dist.entries[0][2], 1.0, places=12)
        self.assertEqual(dist.gap_probs[3], 1.0)

    def test_requires_distribution_flag(self):
        plan = self.make_plan()
        result = G.solve(plan)                 # 没开开关
        with self.assertRaises(ValueError):
            G.build_outcome_distribution(plan, result)

    def test_top_and_omitted(self):
        _, dist = self.solve_dist(pulls=120, char_targets=(2, 2, 0),
                                  weapon_targets=wt(1, 1, 0))
        self.assertEqual(len(dist.top(3)), 3)
        rest_n, rest_p = dist.omitted(3)
        self.assertEqual(rest_n, dist.total_outcomes - 3)
        self.assertAlmostEqual(rest_p + sum(e[2] for e in dist.top(3)), 1.0, places=10)
        self.assertEqual(dist.omitted(10 ** 6), (0, 0.0))

    def test_against_monte_carlo(self):
        plan = self.make_plan(order="character-first", pulls=45,
                              char_targets=(1, 1, 0), weapon_targets=wt(1, 0, 1))
        dist = G.solve(plan, want_distribution=True).distribution
        trials = 30000
        sim = G.monte_carlo(plan, trials, seed=31415)
        hist = sim["outcome_hist"]
        # 精确分布里概率最高的 5 种结果，逐一和模拟频率对照
        for char_counts, weapon_counts, p in dist.top(5):
            sim_p = hist.get((char_counts, weapon_counts), 0) / trials
            sigma = math.sqrt(max(p * (1 - p), 1e-9) / trials)
            self.assertAlmostEqual(sim_p, p, delta=max(4 * sigma, 0.006),
                                   msg=f"结果 {char_counts}/{weapon_counts}: "
                                       f"精确 {p:.4f} vs 模拟 {sim_p:.4f}")
        # 模拟里没出现过的结果，精确概率也应该很小
        for char_counts, weapon_counts, p in dist.entries:
            if (char_counts, weapon_counts) not in hist:
                self.assertLess(p, 0.01)


class TestMonteCarloAgreement(unittest.TestCase):
    """蒙特卡洛模拟与马尔可夫链精确解的对照。"""

    def test_against_simulation(self):
        plan = G.Plan(
            order="weapon-first", pulls=50,
            char_pity=4, char_guarantee=True,
            weapon_pity=8, weapon_guarantee=False,
            char_targets=(1, 1, 0), weapon_targets=wt(1, 0, 1),
        )
        exact = G.solve(plan).chosen
        trials = 20000
        sim = G.monte_carlo(plan, trials, seed=12345)

        sigma = math.sqrt(exact.success * (1 - exact.success) / trials)
        self.assertAlmostEqual(sim["success"], exact.success,
                               delta=max(4 * sigma, 0.005))
        self.assertAlmostEqual(sim["first_complete"], exact.first_complete,
                               delta=max(4 * sigma, 0.008))
        self.assertAlmostEqual(sim["second_complete"], exact.second_complete,
                               delta=max(4 * sigma, 0.008))
        self.assertAlmostEqual(sim["expected_pulls_used"], exact.expected_pulls_used,
                               delta=0.25)
        for key in ("character", "weapon"):
            for i in range(3):
                p = exact.item_prob[key][i]
                s = math.sqrt(max(p * (1 - p), 1e-6) / trials)
                self.assertAlmostEqual(sim["item_prob"][key][i], p,
                                       delta=max(4 * s, 0.006),
                                       msg=f"{key}[{i}] 精确 {p} vs 模拟 {sim['item_prob'][key][i]}")

    def test_simulation_of_other_order(self):
        plan = G.Plan(
            order="character-first", pulls=40,
            char_pity=0, char_guarantee=False,
            weapon_pity=WEAPON_PITY - 1, weapon_guarantee=True,
            char_targets=(1, 1, 1), weapon_targets=wt(1, 0, 0),
        )
        exact = G.solve(plan).chosen
        sim = G.monte_carlo(plan, 15000, seed=999)
        sigma = math.sqrt(exact.success * (1 - exact.success) / 15000)
        self.assertAlmostEqual(sim["success"], exact.success, delta=max(4 * sigma, 0.006))
        # 逐项对照（含第二池吃剩饭的情形）
        self.assertAlmostEqual(sim["item_prob"]["weapon"][0], exact.item_prob["weapon"][0],
                               delta=0.012)
        self.assertAlmostEqual(sim["expected_pulls_used"], exact.expected_pulls_used, delta=0.3)

    def test_guarantee_plus_full_pity_forces_up(self):
        # 垫到最后一档 + 大保底：第 1 抽必出四星且必为 UP，池内各 UP 等概率
        for banner, n_up in ((G.CHARACTER, CHAR_N), (G.WEAPON, WEAPON_N)):
            chain = stepped_chain(banner, all_targets(banner, 1), 1,
                                  pity=banner.n_pity - 1, guarantee=True)
            self.assertAlmostEqual(chain.item_reached_probs()[0], 1 / n_up, places=12)
            self.assertAlmostEqual(chain.complete_prob(), 0.0, places=12)
        # 顺手用模拟器验证同一件事（模拟器与马尔可夫链是两套独立实现）
        rng = random.Random(4)
        trials = 3000
        for banner, n_up in ((G.CHARACTER, CHAR_N), (G.WEAPON, WEAPON_N)):
            hits = [0] * n_up
            for _ in range(trials):
                state = [banner.n_pity - 1, 1] + [0] * n_up
                idx = G._simulate_pull(banner, rng, state)
                self.assertIsNotNone(idx)        # 必出四星
                self.assertEqual(state[0], 0)    # 保底计数清零
                self.assertEqual(state[1], 0)    # 大保底被消耗
                self.assertEqual(sum(state[2:]), 1)
                hits[idx] += 1
            for count in hits:
                self.assertAlmostEqual(count / trials, 1 / n_up, delta=0.03)
            self.assertEqual(sum(hits), trials)  # 每个 UP 都有机会被抽到


class TestInterface(unittest.TestCase):
    """命令行 / JSON / 交互式界面。"""

    def test_json_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(["--order", "w", "--pulls", "40",
                           "--char-pity", "3", "--weapon-guarantee",
                           "--char-targets", "1", "1", "0",
                           "--weapon-targets", "1", "0", "0", "0", "0", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["input"]["order"], "weapon-first")
        self.assertEqual(data["input"]["weapon"]["guarantee"], True)
        self.assertAlmostEqual(data["result"]["success"],
                               data["result"]["success_other_order"], places=12)
        self.assertIn("item_prob_by_order", data["result"])

    def test_text_report(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(["--order", "character", "--pulls", "90",
                           "--char-targets", "2", "1", "0",
                           "--weapon-targets", "1", "1", "1", "0", "0"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        for keyword in ("当前抽数 90 抽的成功率", "分池完成率", "各 UP 物品达成概率",
                        "达成把握与所需抽数", "先角色池，后武器池"):
            self.assertIn(keyword, out)

    def test_verify_flag(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = G.main(["--pulls", "20", "-ct", "1", "0", "0",
                           "--verify", "2000", "--seed", "7"])
        self.assertEqual(code, 0)
        self.assertIn("蒙特卡洛对照", buf.getvalue())

    def test_interactive_mode(self):
        # 顺序: 顺序=2(先武器) 抽数=30 角色垫=1 武器垫=2 角色大保底=n 武器大保底=y
        #       角色1=1 角色2=0 角色3=0 / 武器1=1 武器2=1 武器3=0 武器4=0 武器5=0
        answers = iter(["2", "30", "1", "2", "n", "y",
                        "1", "0", "0", "1", "1", "0", "0", "0"])
        original = builtins.input
        builtins.input = lambda prompt="": next(answers)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                plan = G.interactive_plan()
        finally:
            builtins.input = original
        self.assertEqual(plan.order, "weapon-first")
        self.assertEqual(plan.pulls, 30)
        self.assertEqual(plan.char_pity, 1)
        self.assertEqual(plan.weapon_pity, 2)
        self.assertFalse(plan.char_guarantee)
        self.assertTrue(plan.weapon_guarantee)
        self.assertEqual(plan.char_targets, ct(1))
        self.assertEqual(plan.weapon_targets, wt(1, 1))

    def test_missing_pulls_is_an_error(self):
        class _NotATty:
            def isatty(self) -> bool:
                return False

        original = sys.stdin
        sys.stdin = _NotATty()  # type: ignore[assignment]
        try:
            with self.assertRaises(SystemExit) as cm:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    G.main([])
        finally:
            sys.stdin = original
        self.assertEqual(cm.exception.code, 2)

    def test_report_alignment_helper(self):
        self.assertEqual(G._disp_width("ab中"), 4)
        self.assertEqual(G._pad("中", 4), "中  ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
