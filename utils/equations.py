import numpy as np
import random

FUNC_TYPES = ['polynomial']

def sample_polynomial_params(max_deg=6, sparsity_prob=0.25):
    # Coef ranges that shrink with degree (deg 0 .. 6)
    ranges = [
        (-1.0, 1.0),    # const
        (-2.0, 2.0),    # x
        (-1.0, 1.0),    # x^2
        (-0.5, 0.5),    # x^3
        (-0.15, 0.15),  # x^4
        (-0.05, 0.05),  # x^5
        (-0.01, 0.01),  # x^6
    ]
    params = []
    for lo, hi in ranges[: max_deg + 1]:
        if random.random() < sparsity_prob:
            params.append(0.0)   # randomly drop term -> more variety
        else:
            params.append(random.uniform(lo, hi))

    # Ensure at least one non-zero coefficient
    if all(abs(p) < 1e-9 for p in params):
        params[random.randint(0, max_deg)] = random.uniform(*ranges[0])
    return params  # len <= 7

def generate_equation(eq_type, params, x):
    if eq_type == 'polynomial':
        return params[0]*x**6 + params[1]*x**5 + params[2]*x**4 + params[3]*x**3 + params[4]*x**2 + params[5]*x + params[6]
    else:
        return params[0] * np.exp(params[1] * x) + params[2]

def generate_equation_string(eq_type, params):
    if eq_type == 'polynomial':
        return f"{params[0]:.3f}*x^6 + {params[1]:.3f}*x^5 + {params[2]:.3f}*x^4 + {params[3]:.3f}*x^3 + {params[4]:.3f}*x^2 + {params[5]:.3f}*x + {params[6]:.3f}"
    else:
        return f"{params[0]:.3f}*exp({params[1]:.3f}*x) + {params[2]:.3f}"