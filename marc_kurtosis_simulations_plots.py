#%%

# plot Marc's simulation results in a ready format (WSe2 kurt paper)

import numpy as np
import matplotlib.pyplot as plt

# fontparameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 28

#save_fol = r'C:\Users\earro\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\fig2/'
#save_fol = r'D:\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\fig2/'

# load data
data = r'C:\Users\earro\Downloads\marc_sims/kurtosis_vs_time_diff_initial_kurtosis.dat'
#data = r'D:\DATA\Sims_marc/kurtosis_vs_time_diff_initial_kurtosis.dat'
data = np.loadtxt(data)

# extract columns
time_diff = data[:, 0]  # time array
pure_gaussian = data[:, 2]  # pure diffusion kurtosis
kurt_pos = data[:, 1]  # initial kurtosis = 1
kurt_neg = data[:, 3]  # initial kurtosis = -1

# create figure
plt.figure(figsize=(10, 5))
plt.plot(time_diff, pure_gaussian, label='Pure Gaussian', color='black', linewidth=2)
plt.plot(time_diff, kurt_pos, label='EK(0) > 0', color='blue', linewidth=2)
plt.plot(time_diff, kurt_neg, label='EK(0) < 0', color='red', linewidth=2)

plt.xlabel('Time (ns)')
plt.ylabel('Excess Kurtosis')
#plt.title('Kurtosis vs Time Difference')
plt.legend( frameon=False, fontsize=24)

plt.xlim(0, 20)
plt.ylim(-1.2, 1.)

plt.xticks(np.arange(0, 25, 10))
plt.yticks(np.arange(-1, 1.5, 1))

# right top ticks
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.tick_params(axis='both', which='major', length=5, width=2)
plt.tick_params(axis='both', which='minor', length=3, width=1)

#plt.savefig(save_fol + 'kurtosis_different_initial_kurtosis.svg', dpi=300, bbox_inches='tight')
plt.show()

# %%

# auger etc simulations

import numpy as np
import matplotlib.pyplot as plt

# fontparameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 28
# load data
data = r'C:\Users\earro\Downloads\marc_sims\Re_ Discussion on new publication/'
#data = r'D:\DATA\Sims_marc\model_comparison\Re_ Exciton diffusion_ WSe2_ evolution of kurtosis/'
Auger = np.loadtxt(data + 'Auger.dat')
Auger_kurt = np.loadtxt(data + 'Auger.kurtosis.dat')

# deep traps
deep_traps = np.loadtxt(data + 'deep_traps.dat')
deep_traps_kurt = np.loadtxt(data + 'deep_traps.kurtosis.dat')

# shallow traps
shallow_traps = np.loadtxt(data + 'shallow_traps.dat')
shallow_traps_kurt = np.loadtxt(data + 'shallow_traps.kurtosis.dat')

# pure diffusion
pure_diffusion = np.loadtxt(data + 'pure_diffusion.dat')
pure_diffusion_kurt = np.loadtxt(data + 'pure_diffusion.kurtosis.dat')

# Phonons: analytic solution, calculated separately

# create figure
plt.figure(figsize=(8, 5))
plt.plot(Auger[:, 0], Auger[:, 1], label='Auger', color='blue', linewidth=2)
plt.plot(deep_traps[:, 0], deep_traps[:, 1], label='Deep Traps', color='red', linewidth=2)
plt.plot(shallow_traps[:, 0], shallow_traps[:, 1], label='Traps', color='green', linewidth=2)
plt.plot(pure_diffusion[:, 0], pure_diffusion[:, 1], label='Pure Diffusion', color='black', linewidth=2)

plt.xlabel('Time (ns)')
plt.ylabel('Intensity (a.u.)')

plt.legend(frameon=False, fontsize=20)

# Kurtosis plot

plt.figure(figsize=(8, 5))
plt.plot(Auger[:, 0], Auger_kurt[:-1], label='Auger', color='red', linewidth=3)
#plt.plot(deep_traps[:, 0], deep_traps_kurt[:-1], label='Deep Traps', color='red', linewidth=2)
plt.plot(shallow_traps[:, 0], shallow_traps_kurt[:-1], label='Traps', color='blue', linewidth=3)
plt.plot(pure_diffusion[:, 0], pure_diffusion_kurt[:-1], label='Pure Diffusion', color='black', linewidth=3)

plt.xlabel('Time (ns)')
plt.ylabel('Excess Kurtosis')

plt.legend(frameon=False, fontsize=24)

plt.xlim(-0., 6)
plt.ylim(-1.2, 0.5)

plt.xticks(np.arange(0, 8, 2.))
plt.yticks(np.arange(-1, 0.1, 1))

plt.axhline(0, color='black', linestyle='--', linewidth=2)
#plt.axhline(-0.5, color='black', linestyle='--', linewidth=1)

# right top ticks
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.tick_params(axis='both', which='major', length=5, width=2)
plt.tick_params(axis='both', which='minor', length=3, width=1)

#save_fol = r'C:\Users\earro\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\fig2/'
#plt.savefig(save_fol + 'kurtosis_different_scenarios.svg', dpi=300, bbox_inches='tight')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
folder = r'C:\Users\earro\OneDrive - UAM\PhoND\Lab\Sims\variation\variation/'
#folder = r'C:\Users\PhoNDJaen\Downloads\variation\variation/'
# first plot the ones for different trap density
import glob

file_ref = folder + 'complete_2.220uJcm2.dat'

files = glob.glob(folder + 'c0*kurtosis.dat')
file_0 = files[2]
file_1 = files[1]
file_2 = files[4]

# plot them
time_arr = np.loadtxt(file_ref)[:, 0]
kurt_0 = np.loadtxt(file_0)[1:]
kurt_1 = np.loadtxt(file_1)[1:]
kurt_2 = np.loadtxt(file_2)[1:]

plt.figure(figsize=(12, 8))
#plt.title('Trap density', fontsize=32)
plt.plot(time_arr, kurt_1, label=r'0.125 $\mu m^{-1}$', color='blue', linewidth=3)
plt.plot(time_arr, kurt_0, label=r'0.25 $\mu m^{-1}$', color='black', linewidth=3, linestyle='--')
plt.plot(time_arr, kurt_2, label=r'1 $\mu m^{-1}$', color='red', linewidth=3)

plt.xlim(-0.01, 6)
plt.ylim(-0.6, 1.3)

plt.xticks(np.arange(0, 8, 2.))
plt.yticks(np.arange(-0.5, 1.3, 0.5))

plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel('Time (ns)', fontsize=32)
plt.ylabel('Excess Kurtosis', fontsize=32)

plt.legend(frameon=False, fontsize=32)

plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                labelsize=28)
plt.tick_params(axis='both', which='major', length=5, width=2)
plt.tick_params(axis='both', which='minor', length=3, width=1)


save_fol = r'C:\Users\earro\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\fig2/'
plt.savefig(save_fol + 'kurtosis_different_trap_density.pdf', dpi=300, bbox_inches='tight')

# %%

files = glob.glob(folder + 'mu2*kurtosis.dat')
file_0 = files[2]
file_1 = files[0]
file_2 = files[4]

# plot them
time_arr = np.loadtxt(file_ref)[:, 0]
kurt_0 = np.loadtxt(file_0)[1:]
kurt_1 = np.loadtxt(file_1)[1:]
kurt_2 = np.loadtxt(file_2)[1:]


plt.figure(figsize=(8, 8))
plt.title('Escape rate', fontsize=32)
plt.plot(time_arr, kurt_1, label=r'200 ns', color='blue', linewidth=3)
plt.plot(time_arr, kurt_0, label=r'5 ns', color='black', linewidth=3, linestyle='--')
plt.plot(time_arr, kurt_2, label=r'1.25 ns', color='red', linewidth=3)

plt.xlim(-0.01, 6)
plt.ylim(-0.6, 1.3)

plt.xticks(np.arange(0, 8, 2.))
plt.yticks(np.arange(-0.5, 1.6, 0.5))

plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel('Time (ns)')
plt.ylabel('Excess Kurtosis')

plt.legend(frameon=False, fontsize=24)

# %%

# trap decay rate

files = glob.glob(folder + 'nu2*kurtosis.dat')
file_0 = files[0]
file_1 = files[1]
file_2 = files[2]

# plot them
time_arr = np.loadtxt(file_ref)[:, 0]
kurt_0 = np.loadtxt(file_0)[1:]
kurt_1 = np.loadtxt(file_1)[1:]
kurt_2 = np.loadtxt(file_2)[1:]


plt.figure(figsize=(8, 8))
plt.title('Decay rate', fontsize=32)
plt.plot(time_arr, kurt_1, label=r'200 ns', color='blue', linewidth=3)
plt.plot(time_arr, kurt_0, label=r'10 ns', color='black', linewidth=3, linestyle='--')
plt.plot(time_arr, kurt_2, label=r'5 ns', color='red', linewidth=3)

plt.xlim(-0.01, 6)
plt.ylim(-0.6, 1.3)

plt.xticks(np.arange(0, 8, 2.))
plt.yticks(np.arange(-0.5, 1.6, 0.5))

plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel('Time (ns)')
plt.ylabel('Excess Kurtosis')

plt.legend(frameon=False, fontsize=24)

# %%

# trapping rate

files = glob.glob(folder + 'lambda2*kurtosis.dat')
file_0 = files[3]
file_1 = files[0]
file_2 = files[5]

# plot them
time_arr = np.loadtxt(file_ref)[:, 0]
kurt_0 = np.loadtxt(file_0)[1:]
kurt_1 = np.loadtxt(file_1)[1:]
kurt_2 = np.loadtxt(file_2)[1:]


plt.figure(figsize=(8, 8))
plt.title('Decay rate', fontsize=32)
plt.plot(time_arr, kurt_1, label=r' 0.0625 $ns^{-1}$', color='blue', linewidth=3)
plt.plot(time_arr, kurt_0, label=r'1.9 $ns^{-1}$', color='black', linewidth=3, linestyle='--')
plt.plot(time_arr, kurt_2, label=r'5.7 $ns^{-1}$', color='red', linewidth=3)

plt.xlim(-0.01, 6)
plt.ylim(-0.6, 1.3)

plt.xticks(np.arange(0, 8, 2.))
plt.yticks(np.arange(-0.5, 1.6, 0.5))

plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel('Time (ns)')
plt.ylabel('Excess Kurtosis')

plt.legend(frameon=False, fontsize=24)
# %%

# different Auger rates
files = glob.glob(folder + 'rho*kurtosis.dat')

file_0 = files[2]
file_1 = files[1]
file_2 = files[4]

# plot them
time_arr = np.loadtxt(file_ref)[:, 0]
kurt_0 = np.loadtxt(file_0)[1:]
kurt_1 = np.loadtxt(file_1)[1:]
kurt_2 = np.loadtxt(file_2)[1:]

plt.figure(figsize=(8, 8))
plt.title('Auger rate', fontsize=32)
plt.plot(time_arr, kurt_1, label=r'0.025 $\nu^{-1} \mu m^{-1}$', color='blue', linewidth=3)
plt.plot(time_arr, kurt_0, label=r'0.0125 $\nu^{-1} \mu m^{-1}$', color='black', linewidth=3, linestyle='--')
plt.plot(time_arr, kurt_2, label=r'0.1 $\nu^{-1} \mu m^{-1}$', color='red', linewidth=3)
plt.xlim(-0.01, 6)
plt.ylim(-0.6, 1.3)

plt.xticks(np.arange(0, 8, 2.))
plt.yticks(np.arange(-0.5, 1.6, 0.5))
plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel('Time (ns)')
plt.ylabel('Excess Kurtosis')
plt.legend(frameon=False, fontsize=24)
# %%


# lets plot a simple fake thingy


decay = 2*np.exp(-time_arr/2) -0.5
decay_fast = 2.3*np.exp(-time_arr/0.5) -0.5
decay_way_fast = 4.2*np.exp(-time_arr/0.1) -0.5

plt.plot(time_arr, decay, label='decay', color='black', linewidth=3)
plt.plot(time_arr, decay_fast, label='decay fast', color='red', linewidth=3)
plt.plot(time_arr, decay_way_fast, label='decay way fast', color='blue', linewidth=3)
plt.xlim(-0.01, 6)
plt.ylim(-0.61, 1.1)

plt.xticks(np.arange(0, 8, 2.), fontsize=22)
plt.yticks(np.arange(-0.5, 1.6, 0.5), fontsize=22)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Time (ns)', fontsize=22)
plt.ylabel('Excess Kurtosis', fontsize=22)


# %%

#file = r'D:\DATA\Sims_marc\traps/diffmap_shallow.dat'
file = r'C:\Users\earro\OneDrive - UAM\PhoND\Lab\Sims\profiles/diffmap_shallow.dat'
data = np.loadtxt(file)

X = np.linspace(-2, 2, len(data[0]))
t = np.linspace(0, 10, len(data))

plt.imshow(data, extent=[X[0], X[-1], t[-1], t[0]], aspect='auto', cmap='hot')

# plot 5 profiles in a single plot
cmap = plt.get_cmap('magma').reversed()
colors = cmap(np.linspace(0, 1, 5))
plt.figure(figsize=(8, 6))
plt.plot(X, data[1]/np.max(data[1]), label='t=0 ns', color=colors[1], linewidth=2)
plt.plot(X, data[10]/np.max(data[10]), label='t=1 ns', color=colors[2], linewidth=2)
plt.plot(X, data[50]/np.max(data[50]), label='t=5 ns', color=colors[3], linewidth=2)
plt.plot(X, data[100]/np.max(data[100]), label='t=10 ns', color=colors[4], linewidth=2)


plt.xlabel('Position (μm)', fontsize=32)
plt.ylabel('Normalized Intensity (a.u.)', fontsize=28)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=False, fontsize=14, loc='upper right')

plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=20,
                top=True, right=True, direction='in', length=5, width=2,
                )
plt.tick_params(axis='both', which='minor', labelsize=16,
                top=True, right=True, direction='in', length=3, width=1,
                )

plt.xlim(-2.1, 2.1)
plt.ylim(0, 1.1)
#save_fol = r'D:\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\slides\Drawings/'
#save_fol = r'C:\Users\earro\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\SI/' 
#plt.savefig(save_fol + 'diffmap_shallow_profiles.pdf', dpi=300, bbox_inches='tight')

#%%

# case for phonons (heat)

def phonon_ek(n00=1, n10=1.5):
    #n00 = 1
    #n10 = 1.5 # for heat
    nu0 = 1.0
    nu1 = 2.5 # for heat
    D0 = 0.1
    D1 = 0.005 # for heat

    # n0 == n0*exp(-nu0 *t)
    # n1 == n1*exp(-nu1 *t)
    # n = n0 + n1. Find kurtosis of the distribution

    time_array_sim = np.linspace(0, 10, 1000)  # time array
    Ekt = []
    for t in time_array_sim:
        n0 = n00 * np.exp(-nu0 * t)
        #sigma20 = 4*D0*t
        n1 = n10 * np.exp(-nu1 * t)
        #sigma21 = 4*D1*t
        n = n0 + n1

        #sigma2 = (n0* sigma20 + n1* sigma21)/n
        K = 2*n*(n0*D0**2 + n1*D1**2)/(n0*D0 + n1*D1)**2 
        EK = K - 2
        Ekt.append(EK)

    return np.array(Ekt), time_array_sim


plt.figure(figsize=(8, 5))
EK, time_array_sim = phonon_ek(n00=1, n10=1.)
plt.plot(time_array_sim, EK, label='Phonons', color='purple', linewidth=2)


#%%
file = r'C:\Users\earro\Downloads\marc_sims/excess_kurtosis_vs_time.dat'
data = np.loadtxt(file)

time = data[:, 0]
pure_diff = data[:, 1]
auger = data[:, 2]
traps = data[:, 3]
deep_traps = data[:, 4]

plt.figure(figsize=(5.5, 6))
plt.plot(time, pure_diff, label='Pure Diffusion', color='black', linewidth=2)
plt.plot(time, traps, label='Traps', color='blue', linewidth=2)
plt.plot(time, auger, label='Auger', color='red', linewidth=2)
#plt.plot(time, deep_traps, label='Deep Traps', color='green', linewidth=2)
plt.plot(time_array_sim, EK, label='Phonons', color='Green', linewidth=2)
plt.xlabel('Time (ns)', fontsize=32)
plt.ylabel('Excess Kurtosis', fontsize=32)
plt.legend(frameon=False, fontsize=28)

plt.xlim(-0.02, 6)
plt.ylim(-0.8, 1.8)

plt.xticks(np.arange(0, 8, 2.), )
plt.yticks(np.arange(-0.5, 1.6, 0.5), )
plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=32,
                top=True, right=True, direction='in', length=5, width=2,
                )
plt.tick_params(axis='both', which='minor', labelsize=32,
                top=True, right=True, direction='in', length=3, width=1,
                )
#save_fol = r'C:\Users\earro\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\fig2/'
#plt.savefig(save_fol + 'kurtosis_different_scenarios_EK1.svg', dpi=300, bbox_inches='tight')
# %%


file = file = r'C:\Users\earro\Downloads\marc_sims/excess_kurtosis_vs_time_Auger.dat'
data = np.loadtxt(file)

rhos = [0.05, 0.5, 5, ]  # Auger rates in um^2/ns
# rho to cm^2/s
rhos = [rho*1e-8*1e9 for rho in rhos]
colors = ['blue', 'black', 'red']
time = data[:, 0]
plt.figure(figsize=(8, 6))
for i, rho in enumerate(rhos):
    # format rho to no decimals if integer
    if rho.is_integer():
        rho = int(rho)
    plt.plot(time, data[:, i+1], label=f'{rho} $cm^2/s$', color=colors[i], linewidth=2)

plt.xlabel('Time (ns)', fontsize=28)
plt.ylabel('Excess Kurtosis', fontsize=28)
plt.legend(frameon=False, fontsize=24)

plt.xlim(-0.02, 6)
plt.ylim(-0.1, 1.8)

plt.xticks(np.arange(0, 8, 2.), fontsize=22)
plt.yticks(np.arange(-0., 1.6, 0.5), fontsize=22)
plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=20,
                top=True, right=True, direction='in', length=5, width=2,
                )
plt.tick_params(axis='both', which='minor', labelsize=16,
                top=True, right=True, direction='in', length=3, width=1,
                )

#save_fol = r'C:\Users\earro\OneDrive - UAM\PhoND\Paper_figures\Wse_traps\fig2/'
#plt.savefig(save_fol + 'kurtosis_different_Auger_rates.svg', dpi=300, bbox_inches='tight')
# %%
