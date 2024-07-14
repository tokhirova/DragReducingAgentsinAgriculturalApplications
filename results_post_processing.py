import numpy as np
import scipy
import matplotlib.pyplot as plt

drag_no_polymer = np.load("results/arrays/experiments/2000/drag_coeff.npy")
drag_polymer1 = np.load("results/arrays/experiments/2001/drag_coeff.npy")
drag_polymer2 = np.load("results/arrays/experiments/2002/drag_coeff.npy")
drag_polymer3 = np.load("results/arrays/experiments/1006/drag_coeff.npy")

time = np.load("results/arrays/experiments/2000/u_time.npy")

# from IPython import embed
# embed()

dp1 = np.divide(drag_polymer1-drag_no_polymer,drag_no_polymer) * 100
dp2 = np.divide(drag_polymer2-drag_no_polymer,drag_no_polymer) * 100
dp3= np.divide(drag_polymer3-drag_no_polymer,drag_no_polymer) * 100
d1 = scipy.integrate.simpson(dp1,time)
d2 = scipy.integrate.simpson(dp2,time)
d3 = scipy.integrate.simpson(dp3,time)
print(d1)
print(d2)
print(d3)

cond = np.load("results/arrays/experiments/4000/F_cond.npy")
cond_time = np.load("results/arrays/experiments/4000/u_time.npy")
dt = 400
t_max = 1.15
t_cond = np.linspace(0,t_max,num=460)

fig = plt.figure(figsize=(25, 8))
l1 = plt.plot(np.log(cond), label=r"FEniCSx", linewidth=2)
plt.title("Condition of FP non-linear solver")
plt.grid()
plt.legend()
plt.savefig("plots/experiments/4000/" + "condition_plot.png")

#plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(25, 8))
l1 = plt.plot(time,drag_no_polymer, linewidth=2)
l2 = plt.plot(time,drag_polymer3, linewidth=2)
plt.title("Drag comparison")
plt.xlabel("Time t")
plt.ylabel(r"$\text{Drag coefficient } \text{C}_{D}$")
plt.grid()
plt.legend(["no polymer", "30% polymer"], loc="lower right")
plt.savefig("plots/experiments/2003/" + "drag_comparison_with_no_polymer.png")
