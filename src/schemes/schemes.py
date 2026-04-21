"""DACI controller + baseline schemes (updated per paper rewrite)."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.cluster import Cluster
from src.model_spec import ModelSpec
from src.predictor import Predictor, adaptive_horizon
from src.dp import solve_initial_dp, solve_runtime_dp, eval_surrogate
from src.cost_model import T_decode_window, compute_u_thermal, stage_mem_bytes


@dataclass
class SchemeState:
    name: str
    S: int = 0
    a: List[int] = field(default_factory=list)
    b: List[int] = field(default_factory=list)
    predictor: Optional[Predictor] = None
    TPOT_nominal_s: float = 0.05
    W_sec: float = 1.0
    last_committed_window: int = 0
    n_reconfigs: int = 0
    total_overhead_s: float = 0.0


def _bootstrap_TPOT(cluster: Cluster, ms: ModelSpec, S: int, a: List[int], b: List[int],
                    P: int, link_obs: dict) -> float:
    phi_unit = np.ones(cluster.N)
    return T_decode_window(cluster, ms, b, a, phi_unit, link_obs, P, 0)


class DACIScheme:
    name = "DACI"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.algo = cfg["algo"]
        self.lambda_slack = self.algo["switching"]["lambda_slack"]
        self.ablation = cfg.get("ablation", {}).get("mode", "none")

    def decide_initial(self, cluster: Cluster, ms: ModelSpec,
                       obs: dict, P: int, G_hat: int) -> SchemeState:
        W = self.algo["window"]["W_tokens"]
        S_min = self.algo["stage_range"]["S_min"]
        S_max = self.algo["stage_range"]["S_max"]

        phi_0 = self._phi_from_obs(obs, cluster)
        phi_avg = phi_0.copy()
        K_0 = self.algo["window"]["H_max"]

        best = solve_initial_dp(
            cluster, ms, S_min, S_max, ms.L, P, G_hat,
            phi_0, phi_avg, obs["link_obs"], obs["q_mem_obs"],
            obs["theta_obs"], K_0,
        )
        if best["S"] is None:
            raise RuntimeError("No feasible initial deployment found")

        TPOT_nom = _bootstrap_TPOT(cluster, ms, best["S"], best["a"], best["b"], P, obs["link_obs"])
        W_sec = W * TPOT_nom

        pred = Predictor.build(cluster, self.cfg,
                               obs["q_cmp_obs"], obs["q_mem_obs"], obs["theta_obs"])
        return SchemeState(
            name=self.name, S=best["S"], a=best["a"], b=best["b"],
            predictor=pred, TPOT_nominal_s=TPOT_nom, W_sec=W_sec,
        )

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])

        H_max = self.algo["window"]["H_max"]
        tau_cv = self.algo["window"]["tau_variance"]
        u_future = np.tile(u_curr.reshape(-1, 1), (1, H_max))
        fcst = st.predictor.forecast(cluster, u_future, H_max)

        # --- Ablation: no_predictor (persistence across horizon) ---
        if self.ablation == "no_predictor":
            phi0 = fcst["phi_hat"][:, 0:1]
            fcst["phi_hat"] = np.tile(phi0, (1, H_max))
            fcst["phi_var"] = np.zeros_like(fcst["phi_var"])
            qmem0 = fcst["q_mem_hat"][:, 0:1]
            fcst["q_mem_hat"] = np.tile(qmem0, (1, H_max))

        # --- Ablation: no_adaptive_H (fixed at H_max) ---
        if self.ablation == "no_adaptive_H":
            H_r = H_max
        else:
            H_r = adaptive_horizon(fcst["phi_hat"], fcst["phi_var"], tau_cv, H_max, st.a,
                                   min_horizon=self.algo["predictor"]["adaptive_horizon"]["min_horizon"])
        K_r = min(H_r, max(1, (G_rem + W - 1) // W))

        # --- Ablation: no_bottleneck (greedy, ignores H_s during search) ---
        if self.ablation == "no_bottleneck":
            from src.dp import solve_runtime_dp_greedy
            J_new, b_new = solve_runtime_dp_greedy(
                cluster, ms, st.S, st.a, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
                fcst["phi_hat"], fcst["phi_infinity"], fcst["q_mem_hat"], obs["link_obs"]
            )
        else:
            max_shift = self.algo.get("dp", {}).get("max_boundary_shift", None)
            J_new, b_new = solve_runtime_dp(
                cluster, ms, st.S, st.a, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
                fcst["phi_hat"], fcst["phi_infinity"], fcst["q_mem_hat"], obs["link_obs"],
                max_boundary_shift=max_shift,
            )

        J_incumbent = eval_surrogate(
            cluster, ms, st.S, st.a, st.b, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], obs["link_obs"]
        )

        # --- Ablation: no_lazy (always accept if candidate exists) ---
        if self.ablation == "no_lazy":
            accepted = b_new is not None and b_new != st.b
        else:
            accepted = b_new is not None and J_new < J_incumbent - self.lambda_slack

        b_out = b_new if accepted else st.b

        TPOT_nom = _bootstrap_TPOT(cluster, ms, st.S, st.a, b_out, P, obs["link_obs"])
        st.TPOT_nominal_s = TPOT_nom
        st.W_sec = W * TPOT_nom

        meta = {
            "H_r_star": H_r, "K_r": K_r,
            "J_new": J_new, "J_incumbent": J_incumbent,
            "phi_hat_curr": fcst["phi_hat"][:, 0].tolist(),
            "b_new_candidate": b_new,
            "u_thermal": u_curr.tolist(),
            "ablation": self.ablation,
        }
        return b_out, accepted, meta

    def _phi_from_obs(self, obs, cluster: Cluster) -> np.ndarray:
        """Instantaneous phi from current observations. phi_wk = 1 + rho * q_cmp/mu_n."""
        phi = np.ones(cluster.N)
        for n_idx, node in enumerate(cluster.nodes):
            phi_th = 1.0 + node.gamma * max(0.0, obs["theta_obs"][n_idx] - node.theta_th)
            phi_wk = 1.0 + node.rho * (obs["q_cmp_obs"][n_idx] / node.mu_flops)
            phi[n_idx] = phi_th * phi_wk
        return phi

    def _compute_u(self, cluster, ms, a, b, q_mem, P, t_r):
        """Thermal drive u per Eq.(12): memory-occupancy ratio."""
        return compute_u_thermal(cluster, ms, a, b, q_mem, P, t_r)


class SDAScheme(DACIScheme):
    name = "SDA"

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        meta = {"H_r_star": 0, "K_r": 0, "J_new": None, "J_incumbent": None,
                "phi_hat_curr": [], "b_new_candidate": None,
                "u_thermal": u_curr.tolist()}
        return st.b, False, meta


class RTScheme(DACIScheme):
    name = "RT"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.trigger_ratio = cfg["experiment"]["rt_baseline"]["trigger_ratio"]
        self.TPOT_baseline_nominal = None
        # AutoPipe-style trigger cool-down (default 50 windows; configurable)
        self.cool_down_windows = cfg["experiment"]["rt_baseline"].get("cool_down_windows", 50)
        self._last_trigger_r = -10**9

    def decide_initial(self, cluster, ms, obs, P, G_hat):
        st = super().decide_initial(cluster, ms, obs, P, G_hat)
        self.TPOT_baseline_nominal = st.TPOT_nominal_s
        self._last_trigger_r = -10**9
        return st

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        phi_now = self._phi_from_obs(obs, cluster)
        tpot_now = T_decode_window(cluster, ms, st.b, st.a, phi_now, obs["link_obs"], P, t_r)

        # Current window index (approximation: t_r / W)
        r_now = t_r // W
        on_cool_down = (r_now - self._last_trigger_r) < self.cool_down_windows

        triggered = (tpot_now > self.trigger_ratio * self.TPOT_baseline_nominal) and not on_cool_down
        if triggered:
            # Greedy placement swap: find most-degraded stage, swap to best cool alternative
            stage_phi = [phi_now[st.a[s - 1]] for s in range(1, st.S + 1)]
            hot_stage_idx = int(np.argmax(stage_phi))  # 0-indexed within stages
            hot_node = st.a[hot_stage_idx]

            in_stage = set(st.a)
            # Candidate pool: non-stage nodes with enough memory for the hot stage's blocks
            n_blocks_hot = st.b[hot_stage_idx + 1] - st.b[hot_stage_idx]
            need_mem = _stage_mem_need(ms, n_blocks_hot, P, t_r + G_rem)
            candidates = []
            for n in range(cluster.N):
                if n in in_stage:
                    continue
                node = cluster.nodes[n]
                m_eff = max(0.0, node.m_bytes - obs["q_mem_obs"][n])
                if need_mem <= m_eff:
                    candidates.append(n)
            if candidates:
                cool_node = min(candidates, key=lambda n: phi_now[n])
                # Propose a_new with swap
                a_new = list(st.a)
                a_new[hot_stage_idx] = cool_node
                # Re-optimize b given a_new via runtime DP (frozen placement = a_new)
                H_max = self.algo["window"]["H_max"]
                K_r = min(H_max, max(1, (G_rem + W - 1) // W))
                # Use current phi as horizon (reactive, no prediction)
                phi_hat_horizon = np.tile(phi_now.reshape(-1, 1), (1, H_max))
                phi_inf = phi_now.copy()
                q_mem_hat = np.tile(obs["q_mem_obs"].reshape(-1, 1), (1, H_max))
                max_shift = self.algo.get("dp", {}).get("max_boundary_shift", None)
                J_new, b_new = solve_runtime_dp(
                    cluster, ms, st.S, a_new, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
                    phi_hat_horizon, phi_inf, q_mem_hat, obs["link_obs"],
                    max_boundary_shift=max_shift,
                )
                # Reactive: accept if J_new improves (no lazy slack; trigger has already fired)
                if b_new is not None and J_new < tpot_now * G_rem:
                    old_a = st.a
                    st.a = a_new
                    self._last_trigger_r = r_now
                    meta = {"H_r_star": 0, "K_r": K_r, "J_new": J_new,
                            "J_incumbent": tpot_now * G_rem, "phi_hat_curr": phi_now.tolist(),
                            "b_new_candidate": b_new, "placement_changed": True,
                            "a_prev_for_omega": old_a, "u_thermal": u_curr.tolist()}
                    return b_new, True, meta

        meta = {"H_r_star": 0, "K_r": 0, "J_new": None, "J_incumbent": None,
                "phi_hat_curr": phi_now.tolist(), "b_new_candidate": None,
                "u_thermal": u_curr.tolist()}
        return st.b, False, meta


class FMScheme(DACIScheme):
    name = "FM"

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        H_max = self.algo["window"]["H_max"]
        tau_cv = self.algo["window"]["tau_variance"]
        u_future = np.tile(u_curr.reshape(-1, 1), (1, H_max))
        fcst = st.predictor.forecast(cluster, u_future, H_max)
        H_r = adaptive_horizon(fcst["phi_hat"], fcst["phi_var"], tau_cv, H_max, st.a,
                               min_horizon=self.algo["predictor"]["adaptive_horizon"]["min_horizon"])
        K_r = min(H_r, max(1, (G_rem + W - 1) // W))

        # Greedy 1-swap placement: hot stage -> coolest feasible non-stage node
        phi_hat_curr = fcst["phi_hat"][:, 0]
        stage_phi = [phi_hat_curr[st.a[s - 1]] for s in range(1, st.S + 1)]
        hot_stage_idx = int(np.argmax(stage_phi))
        n_blocks_hot = st.b[hot_stage_idx + 1] - st.b[hot_stage_idx]
        need_mem = _stage_mem_need(ms, n_blocks_hot, P, t_r + G_rem)
        in_stage = set(st.a)
        candidates = []
        for n in range(cluster.N):
            if n in in_stage:
                continue
            node = cluster.nodes[n]
            # Use forecast horizon max for memory feasibility (consistent w/ DP Eq.26)
            q_mem_max_h = float(np.max(fcst["q_mem_hat"][n, :K_r]))
            m_eff = max(0.0, node.m_bytes - q_mem_max_h)
            if need_mem <= m_eff:
                candidates.append(n)

        a_new = list(st.a)
        swapped = False
        if candidates:
            cool_node = min(candidates, key=lambda n: phi_hat_curr[n])
            if phi_hat_curr[cool_node] < stage_phi[hot_stage_idx]:
                a_new[hot_stage_idx] = cool_node
                swapped = True

        # Re-optimize b under a_new (or a unchanged if no swap)
        max_shift = self.algo.get("dp", {}).get("max_boundary_shift", None)
        J_new, b_new = solve_runtime_dp(
            cluster, ms, st.S, a_new, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], fcst["q_mem_hat"], obs["link_obs"],
            max_boundary_shift=max_shift if not swapped else None,
        )
        J_inc = eval_surrogate(
            cluster, ms, st.S, st.a, st.b, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            fcst["phi_hat"], fcst["phi_infinity"], obs["link_obs"]
        )
        accepted = b_new is not None and J_new < J_inc - self.lambda_slack
        if accepted:
            old_a = st.a
            if swapped:
                st.a = a_new
            b_out = b_new
        else:
            b_out = st.b
            old_a = st.a

        TPOT_nom = _bootstrap_TPOT(cluster, ms, st.S, st.a, b_out, P, obs["link_obs"])
        st.TPOT_nominal_s = TPOT_nom
        st.W_sec = W * TPOT_nom

        meta = {"H_r_star": H_r, "K_r": K_r, "J_new": J_new, "J_incumbent": J_inc,
                "phi_hat_curr": phi_hat_curr.tolist(),
                "b_new_candidate": b_new, "placement_changed": swapped and accepted,
                "a_prev_for_omega": old_a,
                "u_thermal": u_curr.tolist()}
        return b_out, accepted, meta


def _stage_mem_need(ms, n_blocks, P, t_end):
    """Conservative memory need for a stage with n_blocks up to decode step t_end."""
    from src.cost_model import stage_mem_bytes
    return stage_mem_bytes(ms, n_blocks, P, t_end)


class OracleScheme(DACIScheme):
    name = "OR"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._gt_phi_future = None
        self._gt_q_mem_future = None

    def set_ground_truth(self, phi_future: np.ndarray, q_mem_future: np.ndarray):
        self._gt_phi_future = phi_future
        self._gt_q_mem_future = q_mem_future

    def decide_runtime(self, cluster, ms, obs, st, P, t_r, G_rem, W):
        u_curr = self._compute_u(cluster, ms, st.a, st.b, obs["q_mem_obs"], P, t_r)
        for n in range(cluster.N):
            st.predictor.kalman_update(n, obs["theta_obs"][n], u_curr[n])
            st.predictor.ar_update(n, obs["q_cmp_obs"][n], obs["q_mem_obs"][n],
                                   self.algo["predictor"]["ar1"]["rls_window_size"])
        H_max = self.algo["window"]["H_max"]
        if self._gt_phi_future is None:
            phi_hat = np.ones((cluster.N, H_max))
            q_mem_hat = np.zeros((cluster.N, H_max))
        else:
            phi_hat = self._gt_phi_future
            q_mem_hat = self._gt_q_mem_future
        phi_inf = phi_hat[:, -1]
        K_r = min(H_max, max(1, (G_rem + W - 1) // W))
        J_new, b_new = solve_runtime_dp(
            cluster, ms, st.S, st.a, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            phi_hat, phi_inf, q_mem_hat, obs["link_obs"]
        )
        J_inc = eval_surrogate(
            cluster, ms, st.S, st.a, st.b, st.b, st.a, ms.L, P, t_r, G_rem, W, K_r,
            phi_hat, phi_inf, obs["link_obs"]
        )
        accepted = b_new is not None and J_new < J_inc
        b_out = b_new if accepted else st.b
        meta = {"H_r_star": H_max, "K_r": K_r, "J_new": J_new, "J_incumbent": J_inc,
                "phi_hat_curr": phi_hat[:, 0].tolist(), "b_new_candidate": b_new,
                "u_thermal": u_curr.tolist()}
        return b_out, accepted, meta


SCHEMES = {"DACI": DACIScheme, "SDA": SDAScheme, "RT": RTScheme,
           "FM": FMScheme, "OR": OracleScheme}


def build_scheme(name: str, cfg: dict):
    if name not in SCHEMES:
        raise KeyError(f"Unknown scheme: {name}")
    return SCHEMES[name](cfg)