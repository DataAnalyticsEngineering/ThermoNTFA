#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # Thermo-mechanical NTFA: Examples from paper
#
# Felix Fritzen <fritzen@simtech.uni-stuttgart.de>,
# Julius Herb <julius.herb@mib.uni-stuttgart.de>,
# Shadi Sharba <shadi.sharba@isc.fraunhofer.de>
#
# University of Stuttgart, Institute of Applied Mechanics, Chair for Data Analytics in Engineering
#
# > **Funding acknowledgment**
# > The IGF-Project no.: 21.079 N / DVS-No.: 06.3341 of the
# > “Forschungsvereinigung Schweißen und verwandte Verfahren e.V.” of the
# > German Welding Society (DVS), Aachener Str. 172, 40223 Düsseldorf, Germany,
# > was funded by the Federal Ministry for Economic Affairs and Climate Action (BMWK)
# > via the German Federation of Industrial Research Associations (AiF) in accordance
# > with the policy to support the Industrial Collective Research (IGF)
# > on the orders of the German Bundestag.
# >
# > Felix Fritzen is funded by the German Research Foundation (DFG) --
# > 390740016 (EXC-2075); 406068690 (FR2702/8-1); 517847245 (FR2702/10-1).
#
# ## Imports

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

data_path = "../data"

# %% [markdown]
# ## Twoscale loading plot:

# %%
dpi = 400
force = []
temp = []
for s in ("fine",):  # 'coarse', 'medium',
    file_name = os.path.join(data_path, "results", f"NTFA293K_{s}_temp_293-800.h5")
    with h5py.File(file_name, mode="r") as F:
        force.append(np.array(F["force28"]) / 1e6)
        temp.append(np.array(F["temperature28"]))

fig, ax = plt.subplots(1, 1, dpi=dpi)
labels = False
for f, T in zip((force), (temp)):
    t = np.linspace(0, 2, f.size)
    ax.plot(t, f, "-s", ms=4, label="tensile force", color="blue")
    if labels:
        ax.plot(1.0, f[np.where(t == 1.0)], "o", color="black", ms=6)
    else:
        ax2 = ax.twinx()
        nhalf = int((t.size + 1) / 2)
        # ax.plot(1.0, f[np.where(t==1.0)], 'o', color='black', ms=6, label='start unloading', zorder=4)
        ax.fill_between(
            t[:nhalf], f[:nhalf], np.zeros(nhalf), color="blue", alpha=0.2, zorder=2
        )
        ax.fill_between(
            t[(nhalf - 1) :],
            f[(nhalf - 1) :],
            np.zeros(f.size - nhalf + 1),
            color="red",
            alpha=0.2,
            zorder=2,
        )
        ax2.plot(t, T, "-s", ms=4, color="red", zorder=4, label="temperature")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force [kN]")
        ax2.set_ylabel("temperature [K]")
        ax.text(0.4, 2, "tensile loading", color="blue", bbox=dict(facecolor="white"))
        ax.text(1.1, 2, "force unloading", color="red", bbox=dict(facecolor="white"))
        ax.grid(zorder=1)
        ax.legend(loc=(0.65, 0.825))
        ax2.legend(loc=(0.65, 0.7))
    nhalf = int((t.size + 1) / 2)
    labels = True
fig.savefig(os.path.join(data_path, "figures", "twoscale_loading.pdf"))
plt.show()

# %% [markdown]
# ## Plot 2:

# %%
fig, ax = plt.subplots(1, 1, dpi=dpi)
ax.plot((1.0, 1.0), (-0.05, 1.05), "--", color="gray", lw=2, zorder=4)
ax.fill_between((0.0, 1.0), (0.0, 1.0), (0.0, 0.0), color="blue", alpha=0.2, zorder=2)
ax.fill_between((1.0, 2.0), (1.0, 0.0), (0.0, 0.0), color="red", alpha=0.2, zorder=2)
ax.plot((0.0, 1.0), (0.0, 1.0), color="blue", lw=2, zorder=1)
ax.plot((1.0, 2.0), (1.0, 0.0), "--", color="red", lw=2, zorder=1)
ax.text(0.4, 0.1, "tensile loading", color="blue", bbox=dict(facecolor="white"))
ax.text(1.1, 0.1, "force unloading", color="red", bbox=dict(facecolor="white"))
ax.set_xlabel("time [s]")
ax.set_ylabel(r"rel. stretch $u/u_\mathrm{max}$ and force $F/F_\mathrm{max}$ [-]")
ax.grid(zorder=3)
fig.savefig(os.path.join(data_path, "figures", "twoscale_loading.pdf"))
plt.show()

# %% [markdown]
# ## Plot:

# %%
file_name = os.path.join(data_path, "results", "test_coarse.h5")
with h5py.File(file_name, mode="r") as F:
    u = np.array(F["displacement1/X"])
    uy = u[1::2]
    D = np.array(F["DATA16/cData"]).reshape((-1, 12))
    eps = D[:, :6]
    sig = D[:, :6]
    f = np.array(F["force16"])[:17]
    print("uymax=", uy.max())
    print("s_yy mean=", sig[:, 2].mean())

fig, ax = plt.subplots(1, 1, dpi=dpi)
ax.plot(f)
fig.savefig(os.path.join(data_path, "figures", "force.pdf"))
plt.show()

# %% [markdown]
# ## Plot:

# %%
file_name = os.path.join(data_path, "results", "test_coarse.h5")
with h5py.File(file_name, mode="r") as F:
    d = []
    for i in range(16):
        D = np.array(F[f"DATA{i+1}/cData"])[:, 6:]
        d.append(D.mean(axis=0))
    d = np.array(d)
    plt.plot(d[:, 2])

    # f = np.array(F["force32"])
    # print(f)
    # plt.plot(f)
    plt.show()