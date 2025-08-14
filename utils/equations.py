import numpy as np
import random

FUNC_TYPES = ['polynomial']

def _horner_eval(params, x):
    """Evaluate polynomial with ascending coeffs [a0, a1, ...] at x using Horner."""
    result = 0.0
    for coef in reversed(params):
        result = result * x + coef
    return result

def _count_excursions(
    params,
    x_min=-5.0,
    x_max=5.0,
    n_points=5001,
    thresh=5.0,
    hysteresis_inner=4.5,
    min_gap_width=0.2,
    min_excursion_width=0.05
):
    n = max(3, int(n_points))
    xs = [x_min + i * (x_max - x_min) / (n - 1) for i in range(n)]
    ys = [_horner_eval(params, x) for x in xs]
    abs_y = [abs(y) for y in ys]

    out = [ay > thresh for ay in abs_y]

    if hysteresis_inner < thresh:
        for i in range(n):
            if not out[i] and abs_y[i] > hysteresis_inner:
                out[i] = True

    dx = (x_max - x_min) / (n - 1)
    min_gap_samples = max(1, int(min_gap_width / dx))
    min_excursion_samples = max(1, int(min_excursion_width / dx))

    # Fill small gaps
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            gap_len = j - i
            if gap_len <= min_gap_samples:
                for k in range(i, j):
                    out[k] = True
            i = j
        else:
            i += 1

    # Count excursions
    excursions = 0
    i = 0
    while i < n:
        if out[i]:
            start = i
            while i < n and out[i]:
                i += 1
            end = i
            width = end - start
            if width >= min_excursion_samples:
                excursions += 1
        else:
            i += 1

    return excursions

def sample_polynomial_params(
    max_deg=6,
    sparsity_prob=0.25,
    drop_x6_prob=0.5,
    max_attempts=200,
    excursion_thresh=5.0,
    excursion_max_allowed=2,
):
    """Returns ascending coeffs [a0, a1, ..., a_max_deg]."""
    ranges = [
        (-1.0, 1.0),    # const (deg 0)
        (-2.0, 2.0),    # x (deg 1)
        (-1.0, 1.0),    # x^2 (deg 2)
        (-0.5, 0.5),    # x^3 (deg 3)
        (-0.15, 0.15),  # x^4 (deg 4)
        (-0.05, 0.05),  # x^5 (deg 5)
        (-0.01, 0.01),  # x^6 (deg 6)
    ]
    max_deg = max(0, min(6, int(max_deg)))

    attempts = 0
    last_params = None
    while attempts < max_attempts:
        attempts += 1

        use_factored = (random.random() < 0.5) and (max_deg >= 5)

        if not use_factored:
            params = []
            for lo, hi in ranges[: max_deg + 1]:
                if random.random() < sparsity_prob:
                    params.append(0.0)
                else:
                    params.append(random.uniform(lo, hi))
        else:
            if max_deg >= 6:
                deg = random.choice([5, 6])
            else:
                deg = 5

            roots = [random.uniform(-4.5, 4.5) for _ in range(deg)]
            a = 0.0
            while abs(a) < 1e-12:
                a = random.uniform(-0.03, 0.03)
            c = random.uniform(-3.0, 3.0)

            poly_high_to_low = [1.0]
            for r in roots:
                new = [0.0] * (len(poly_high_to_low) + 1)
                for i, coef in enumerate(poly_high_to_low):
                    new[i] += coef
                    new[i + 1] += -r * coef
                poly_high_to_low = new

            poly_high_to_low = [a * coeff for coeff in poly_high_to_low]
            poly_high_to_low[-1] += c

            params = [0.0] * (max_deg + 1)
            for i, coeff in enumerate(poly_high_to_low):
                exp = deg - i
                if exp <= max_deg:
                    params[exp] = coeff

        if max_deg >= 6 and random.random() < drop_x6_prob:
            params[6] = 0.0

        if all(abs(p) < 1e-12 for p in params):
            i = random.randint(0, max_deg)
            params[i] = random.uniform(*ranges[i])

        last_params = params

        excursions = _count_excursions(params, x_min=-5.0, x_max=5.0, n_points=1001, thresh=excursion_thresh)
        if excursions <= excursion_max_allowed:
            return params

    print(f"Warning: exceeded max_attempts ({max_attempts}) but returning last sample (excursions={_count_excursions(last_params)})")
    return last_params

def generate_equation(eq_type, params, x):
    if eq_type == 'polynomial':
        # Ensure params has 7 elements (pad with zeros if needed)
        padded_params = list(params) + [0.0] * (7 - len(params))
        
        return (padded_params[6]*x**6 + 
                padded_params[5]*x**5 + 
                padded_params[4]*x**4 + 
                padded_params[3]*x**3 + 
                padded_params[2]*x**2 + 
                padded_params[1]*x + 
                padded_params[0])
    else:
        return params[0] * np.exp(params[1] * x) + params[2]

def generate_equation_string(eq_type, params):
    """Generate human-readable equation string."""
    if eq_type == 'polynomial':
        padded_params = list(params) + [0.0] * (7 - len(params))
        return (f"{padded_params[6]:.3f}*x^6 + {padded_params[5]:.3f}*x^5 + "
                f"{padded_params[4]:.3f}*x^4 + {padded_params[3]:.3f}*x^3 + "
                f"{padded_params[2]:.3f}*x^2 + {padded_params[1]:.3f}*x + {padded_params[0]:.3f}")
    else:
        return f"{params[0]:.3f}*exp({params[1]:.3f}*x) + {params[2]:.3f}"