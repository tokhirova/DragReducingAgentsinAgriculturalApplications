import numpy as np
import scipy
import matplotlib.pyplot as plt

drag_no_polymer = np.load("results/arrays/experiments/10026/drag_coeff.npy")
drag_polymer1 = np.load("results/arrays/experiments/2001/drag_coeff.npy")
drag_polymer2 = np.load("results/arrays/experiments/2002/drag_coeff.npy")
drag_polymer3 = np.load("results/arrays/experiments/10035/drag_coeff.npy")
drag_polymer5 = np.load("results/arrays/experiments/10029/drag_coeff.npy")
drag_polymer4 = np.load("results/arrays/experiments/10031/drag_coeff.npy")

time = np.load("results/arrays/experiments/2000/u_time.npy")

# from IPython import embed
# embed()

# dp1 = np.divide(drag_polymer1-drag_no_polymer,drag_no_polymer) * 100
# dp2 = np.divide(drag_polymer2-drag_no_polymer,drag_no_polymer) * 100
dp03= np.divide(drag_no_polymer-drag_polymer3,drag_no_polymer) * 100
dp04= np.divide(drag_no_polymer-drag_polymer4,drag_no_polymer) * 100
dp05= np.divide(drag_no_polymer-drag_polymer5,drag_no_polymer) * 100
# d1 = scipy.integrate.simpson(dp1,time)
# d2 = scipy.integrate.simpson(dp2,time)
d03 = scipy.integrate.simpson(dp03,time)
d04 = scipy.integrate.simpson(dp04,time)
d05 = scipy.integrate.simpson(dp05,time)
# print(d1)
# print(d2)
print(d03)
print(d04)
# print(d05)

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
l3 = plt.plot(time,drag_polymer4, linewidth=2)
# l4 = plt.plot(time,drag_polymer5, linewidth=2)
plt.title("Drag comparison")
plt.xlabel("Time t")
plt.ylabel(r"$\text{Drag coefficient } \text{C}_{D}$")
plt.grid()
plt.legend(["no polymer", "0.5% polymer", "0.1% polymer", "2% polymer"], loc="lower right")
plt.savefig("plots/experiments/10031/" + "drag_comparison_with_no_polymer.png")
