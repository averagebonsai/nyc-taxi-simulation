from collections.abc import Mapping

import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ============================================================
# PARAMETERS — CHANGE THESE
# ============================================================

A = 1.0
ABS_EPSILON = 0.6
C = 0.75

# Desired outside-option share when both firms charge S = 1
#P_OUTSIDE = 0.3452

# Search range for surge multiplier
S_MIN = 0.5
S_MAX = 5.0


SWEEP_PARAMETER = "P_OUTSIDE"
#ABS_EPSILON = [0.2, 0.4, 0.6, 1.0, 1.5, 2.0] 
P_OUTSIDE = [0.10, 0.30, 0.3452, 0.50, 0.70, 0.87]

PARAMETER_NAMES = ("A", "ABS_EPSILON", "C", "P_OUTSIDE", "S_MIN", "S_MAX")


# ============================================================
# CALIBRATE OUTSIDE OPTION
# ============================================================

def derive_mu(p_outside: float, abs_epsilon: float) -> float:
    """Return the logit scale implied by the outside share and elasticity."""

    return (1 + p_outside) / (2 * abs_epsilon)

def calibrate_outside_utility(p_outside, A=1.0, MU=1.0):
    """
    Calibrates V0 such that:

        P(outside) = p_outside

    when:
        S1 = S2 = 1

    The optional A and MU arguments ensure the target remains valid when
    either parameter is swept.  With their defaults, this preserves the
    original calibration where A = S = 1.
    """

    ride_utility = (A - 1.0) / MU
    return ride_utility + np.log(2 * p_outside / (1 - p_outside))


# ============================================================
# DEMAND
# ============================================================

def market_shares(S1, S2, A, MU, V0):
    """
    Multinomial-logit market shares for:
        Firm 1
        Firm 2
        Outside option
    """

    V1 = (A - S1) / MU
    V2 = (A - S2) / MU

    exp1 = np.exp(V1)
    exp2 = np.exp(V2)
    exp0 = np.exp(V0)

    denominator = exp1 + exp2 + exp0

    q1 = exp1 / denominator
    q2 = exp2 / denominator
    q0 = exp0 / denominator

    return q1, q2, q0


# ============================================================
# PROFIT
# ============================================================

def profit(S_i, q_i, C):
    """
    Normalized profit:

        pi_i = (S_i - C) * q_i
    """

    return (S_i - C) * q_i


# ============================================================
# NASH EQUILIBRIUM
# ============================================================

def nash_foc(S, A, MU, C, V0):
    """
    Symmetric Nash FOC:

        S - C = MU / (1 - q_i)
    """

    q, _, _ = market_shares(S, S, A, MU, V0)

    return (S - C) - MU / (1 - q)


def solve_nash(A, MU, C, V0, s_min=S_MIN, s_max=S_MAX):

    return brentq(
        nash_foc,
        s_min,
        s_max,
        args=(A, MU, C, V0)
    )


# ============================================================
# MONOPOLY / JOINT-PROFIT PRICE
# ============================================================

def joint_profit(S, A, MU, C, V0):
    """
    Both firms charge the same S.
    Calculates total industry profit.
    """

    q1, q2, _ = market_shares(S, S, A, MU, V0)

    pi1 = profit(S, q1, C)
    pi2 = profit(S, q2, C)

    return pi1 + pi2


def solve_monopoly(A, MU, C, V0, s_min=S_MIN, s_max=S_MAX):

    result = minimize_scalar(
        lambda S: -joint_profit(S, A, MU, C, V0),
        bounds=(s_min, s_max),
        method="bounded"
    )

    return result.x


# ============================================================
# PARAMETER SWEEPS AND RUN
# ============================================================

def _as_finite_float(name, value):
    """Convert one scalar setting to a finite float with a clear error."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number; got {value!r}.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")
    return result


def parameter_sets(
    parameters: Mapping[str, float | list[float]],
    sweep_parameter: str | None = None,
) -> list[dict[str, float]]:
    """Return one validated parameter dictionary for each requested sweep value.

    Exactly one named parameter may be a list, and only when it is selected by
    ``sweep_parameter``.  This makes an accidental list in a configuration
    fail early instead of silently producing an unintended Cartesian product.
    """

    unexpected = set(parameters) - set(PARAMETER_NAMES)
    missing = set(PARAMETER_NAMES) - set(parameters)
    if unexpected or missing:
        details = []
        if unexpected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        raise ValueError("Parameter names must match calibration inputs (" + "; ".join(details) + ").")

    list_parameters = [name for name, value in parameters.items() if isinstance(value, list)]
    if sweep_parameter is None:
        if list_parameters:
            raise ValueError(
                "A list was supplied for "
                + ", ".join(list_parameters)
                + "; set SWEEP_PARAMETER to that parameter name."
            )
        values_to_run = [None]
    else:
        if sweep_parameter not in PARAMETER_NAMES:
            raise ValueError(
                f"SWEEP_PARAMETER must be one of {PARAMETER_NAMES}; got {sweep_parameter!r}."
            )
        if list_parameters != [sweep_parameter]:
            raise ValueError(
                "Exactly the parameter named by SWEEP_PARAMETER must be a list; "
                f"found lists for {list_parameters or 'none'}."
            )
        values_to_run = parameters[sweep_parameter]
        if not values_to_run:
            raise ValueError(f"The {sweep_parameter} sweep list cannot be empty.")

    cases = []
    for sweep_value in values_to_run:
        case = {
            name: _as_finite_float(
                name, sweep_value if name == sweep_parameter else value
            )
            for name, value in parameters.items()
        }
        if case["ABS_EPSILON"] <= 0:
            raise ValueError("ABS_EPSILON must be greater than zero.")
        if not 0 < case["P_OUTSIDE"] < 1:
            raise ValueError("P_OUTSIDE must be strictly between zero and one.")
        if case["S_MIN"] >= case["S_MAX"]:
            raise ValueError("S_MIN must be less than S_MAX.")
        cases.append(case)
    return cases


def run_calibration(parameters: Mapping[str, float]) -> dict[str, object]:
    """Solve the calibration and pricing benchmarks for one scalar case."""

    A_value = parameters["A"]
    abs_epsilon = parameters["ABS_EPSILON"]
    c_value = parameters["C"]
    p_outside = parameters["P_OUTSIDE"]
    s_min = parameters["S_MIN"]
    s_max = parameters["S_MAX"]
    mu_value = derive_mu(p_outside, abs_epsilon)

    V0 = calibrate_outside_utility(p_outside, A_value, mu_value)
    q1_base, q2_base, q0_base = market_shares(1.0, 1.0, A_value, mu_value, V0)
    S_nash = solve_nash(A_value, mu_value, c_value, V0, s_min, s_max)
    S_monopoly = solve_monopoly(A_value, mu_value, c_value, V0, s_min, s_max)
    q1_nash, q2_nash, q0_nash = market_shares(S_nash, S_nash, A_value, mu_value, V0)
    q1_monopoly, q2_monopoly, q0_monopoly = market_shares(
        S_monopoly, S_monopoly, A_value, mu_value, V0
    )

    return {
        "parameters": {**parameters, "MU": mu_value},
        "V0": V0,
        "baseline_shares": (q1_base, q2_base, q0_base),
        "S_nash": S_nash,
        "nash_shares": (q1_nash, q2_nash, q0_nash),
        "S_monopoly": S_monopoly,
        "monopoly_shares": (q1_monopoly, q2_monopoly, q0_monopoly),
    }


def print_result(result):
    """Print one calibration result in the script's original report format."""

    parameters = result["parameters"]
    A_value = parameters["A"]
    mu_value = parameters["MU"]
    c_value = parameters["C"]
    p_outside = parameters["P_OUTSIDE"]
    V0 = result["V0"]
    a0_value = V0 * mu_value
    q1_base, q2_base, q0_base = result["baseline_shares"]
    S_nash = result["S_nash"]
    q1_nash, q2_nash, q0_nash = result["nash_shares"]
    S_monopoly = result["S_monopoly"]
    q1_monopoly, q2_monopoly, q0_monopoly = result["monopoly_shares"]

    print("=" * 50)
    print("PARAMETERS")
    print("=" * 50)

    print(f"A:                 {A_value:.4f}")
    print(f"MU:                {mu_value:.4f}")
    print(f"C:                 {c_value:.4f}")
    print(f"Target P(outside): {p_outside:.4f}")
    print(f"Calibrated V0:     {V0:.4f}")
    print(f"Corresponding A0:  {a0_value:.4f}")

    print("\n" + "=" * 50)
    print("BASELINE (S = 1)")
    print("=" * 50)

    print(f"Firm 1 share:      {q1_base:.4f}")
    print(f"Firm 2 share:      {q2_base:.4f}")
    print(f"Outside share:     {q0_base:.4f}")

    print("\n" + "=" * 50)
    print("NASH EQUILIBRIUM")
    print("=" * 50)

    print(f"S_N:               {S_nash:.4f}")
    print(f"Firm 1 share:      {q1_nash:.4f}")
    print(f"Firm 2 share:      {q2_nash:.4f}")
    print(f"Outside share:     {q0_nash:.4f}")

    print("\n" + "=" * 50)
    print("MONOPOLY / JOINT PROFIT")
    print("=" * 50)

    print(f"S_M:               {S_monopoly:.4f}")
    print(f"Firm 1 share:      {q1_monopoly:.4f}")
    print(f"Firm 2 share:      {q2_monopoly:.4f}")
    print(f"Outside share:     {q0_monopoly:.4f}")

    print("\n" + "=" * 50)
    print("COLLUSIVE PRICE GAP")
    print("=" * 50)

    print(f"S_M - S_N:         {S_monopoly - S_nash:.4f}")
    print(f"S_M / S_N:         {S_monopoly / S_nash:.4f}")


if __name__ == "__main__":
    configured_parameters = {
        "A": A,
        "ABS_EPSILON": ABS_EPSILON,
        "C": C,
        "P_OUTSIDE": P_OUTSIDE,
        "S_MIN": S_MIN,
        "S_MAX": S_MAX,
    }
    cases = parameter_sets(configured_parameters, SWEEP_PARAMETER)
    for index, case in enumerate(cases):
        if index:
            print("\n")
        if SWEEP_PARAMETER is not None:
            print(f"SWEEP: {SWEEP_PARAMETER} = {case[SWEEP_PARAMETER]:.4f}")
        print_result(run_calibration(case))

"""
At parameters epsilon = 0.6, q_outside = 0.3452 (calibrated to 2-4am baseline): 
- Nash surge multiplier: 2.1546x
- Monopoly surge multiplier: 2.4528x 

The agent only chooses 1 action within this zone, we expect it to lie between 2.1546 and 2.4528. 

"""
