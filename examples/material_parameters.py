# %% [markdown]
# # Material parameters

# %%
from typing import Tuple, Union

# %%
import numpy as np

# %% [markdown]
# ## Temperature range

# %%
min_temperature = 293.00
max_temperature = 1300.00

# %% [markdown]
# ## Basic tensors

# %%
I2 = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
I4 = np.eye(6)
IxI = np.outer(I2, I2)
P1 = IxI / 3.0
P2 = I4 - P1

# %% [markdown]
# ## Temperature-dependent material parameters of copper phase (phase no. 0)

# %%
poisson_ratio_cu = lambda x: 3.40000e-01 * x**0
conductivity_cu = lambda x: 4.20749e05 * x**0 + -6.84915e01 * x**1
heat_capacity_cu = (
    lambda x: 2.94929e03 * x**0
    + 2.30217e00 * x**1
    + -2.95302e-03 * x**2
    + 1.47057e-06 * x**3
)
cte_cu = lambda x: 1.28170e-05 * x**0 + 8.23091e-09 * x**1
eps_th_cu = lambda x: 1.28170e-05 * (x - min_temperature) + 8.23091e-09 * 0.5 * (
    x * x - min_temperature**2
)
elastic_modulus_cu = (
    lambda x: 1.35742e08 * x**0 + 5.85757e03 * x**1 + -8.16134e01 * x**2
)
hardening_cu = lambda x: 20e06 * x**0

shear_modulus_cu = lambda x: elastic_modulus_cu(x) / (2.0 * (1.0 + poisson_ratio_cu(x)))
bulk_modulus_cu = lambda x: elastic_modulus_cu(x) / (
    3.0 * (1.0 - 2.0 * poisson_ratio_cu(x))
)
stiffness_cu = lambda x: bulk_modulus_cu(x) * IxI + 2.0 * shear_modulus_cu(x) * P2

# %% [markdown]
# ## Temperature-dependent material parameters of wsc phase (phase no. 1)

# %%
poisson_ratio_wsc = lambda x: 2.80000e-01 * x**0
conductivity_wsc = (
    lambda x: 2.19308e05 * x**0
    + -1.87425e02 * x**1
    + 1.05157e-01 * x**2
    + -2.01180e-05 * x**3
)
heat_capacity_wsc = (
    lambda x: 2.39247e03 * x**0
    + 6.62775e-01 * x**1
    + -2.80323e-04 * x**2
    + 6.39511e-08 * x**3
)
cte_wsc = lambda x: 5.07893e-06 * x**0 + 5.67524e-10 * x**1
eps_th_wsc = lambda x: 5.07893e-06 * (x - min_temperature) + 5.67524e-10 * 0.5 * (
    x * x - min_temperature**2
)
elastic_modulus_wsc = (
    lambda x: 4.13295e08 * x**0
    + -7.83159e03 * x**1
    + -3.65909e01 * x**2
    + 5.48782e-03 * x**3
)

shear_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (
    2.0 * (1.0 + poisson_ratio_wsc(x))
)
bulk_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (
    3.0 * (1.0 - 2.0 * poisson_ratio_wsc(x))
)
stiffness_wsc = lambda x: bulk_modulus_wsc(x) * IxI + 2.0 * shear_modulus_wsc(x) * P2


# %% [markdown]
# ## Yield stress function of the plastic phase of the material


# %%
def my_sig_y(
    theta: float, q: float, derivative=False
) -> Union[float, Tuple[float, float]]:
    """
    Defines the yield stress function of the plastic phase of the material

    :param theta: temperature
    :param q: hardening
    :param derivative: if the derivative should also be returned
    """
    if theta < 966.06:
        sig0 = (
            1.12133e02 * theta
            + 3.49810e04
            + 1.53393e05 * np.tanh((635.754 - theta) / 206.958)
        )
    else:
        sig0 = 2023.286
    H = 5e5  # equivalent to 500 MPa
    if derivative is False:
        return sig0 + H * q
    else:
        return sig0 + H * q, H
