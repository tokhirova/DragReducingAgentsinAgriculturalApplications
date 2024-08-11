import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import scipy
import matplotlib.pyplot as plt

# this pipline is only used to visualize the various results we present
# ---------------------------------------------------------------------------------------------------------------------
i = 11011
j = 11012
k = 11013
l = 11014
m = 11017
n = 11005
o = 11009
drag_no_polymer = np.load(f"results/arrays/experiments/{i}/drag_coeff.npy")
drag_polymer1 = np.load(f"results/arrays/experiments/{j}/drag_coeff.npy") #5%
drag_polymer2 = np.load(f"results/arrays/experiments/{k}/drag_coeff.npy") #10%
drag_polymer3 = np.load(f"results/arrays/experiments/{l}/drag_coeff.npy") #20%
drag_polymer4 = np.load(f"results/arrays/experiments/{m}/drag_coeff.npy") #lower Re
# drag_polymer5 = np.load(f"results/arrays/experiments/{n}/drag_coeff.npy") #20%
# drag_polymer6 = np.load(f"results/arrays/experiments/{o}/drag_coeff.npy") #lower Re
# drag_polymer5 = np.load("results/arrays/experiments/11014/drag_coeff.npy")
# drag_polymer6 = np.load("results/arrays/experiments/10026/drag_coeff.npy")


time = np.load("results/arrays/experiments/11000/u_time.npy")

# dp1 = np.divide(drag_no_polymer-drag_polymer1,drag_no_polymer) * 100
# dp2 = np.divide(drag_no_polymer-drag_polymer2,drag_no_polymer) * 100
# dp03= np.divide(drag_no_polymer-drag_polymer3,drag_no_polymer) * 100
# dp04= np.divide(drag_no_polymer-drag_polymer4,drag_no_polymer) * 100
# # dp05= np.divide(drag_no_polymer-drag_polymer5,drag_no_polymer) * 100
# # dp06= np.divide(drag_no_polymer-drag_polymer6,drag_no_polymer) * 100
# d1 = scipy.integrate.simpson(dp1,time)
# d2 = scipy.integrate.simpson(dp2,time)
# d03 = scipy.integrate.simpson(dp03,time)
# d04 = scipy.integrate.simpson(dp04,time)
# # d05 = scipy.integrate.simpson(dp05,time)
# # d06 = scipy.integrate.simpson(dp06,time)
# print(d1)
# print(d2)
# print(d03)
# print(d04)
# # print(d05)
# # print(d06)
#
# fig = plt.figure(figsize=(25, 8))
# l1 = plt.plot(time,drag_no_polymer, linewidth=2)
# l4 = plt.plot(time,drag_polymer4, linewidth=2)
# # l4 = plt.plot(time,drag_polymer6, linewidth=2)
# # l4 = plt.plot(time,drag_polymer5, linewidth=2)
# plt.title("Drag comparison")
# plt.xlabel("Time t")
# plt.ylabel(r"$\text{Drag coefficient } \text{C}_{D}$")
# plt.grid()
# plt.legend(["no polymer", "lower reynolds number"], loc="lower right")
# plt.savefig("plots/experiments/11011/" + "Drag_comparison_in_turbulence_reduction.png")
#
# fig = plt.figure(figsize=(25, 8))
# l1 = plt.plot(time,drag_no_polymer, linewidth=2)
# l4 = plt.plot(time,drag_polymer1, linewidth=2)
# l4 = plt.plot(time,drag_polymer2, linewidth=2)
# l4 = plt.plot(time,drag_polymer3, linewidth=2)
# # l4 = plt.plot(time,drag_polymer4, linewidth=2)
# # l4 = plt.plot(time,drag_polymer6, linewidth=2)
# # l4 = plt.plot(time,drag_polymer5, linewidth=2)
# plt.title("Drag comparison")
# plt.xlabel("Time t")
# plt.ylabel(r"$\text{Drag coefficient } \text{C}_{D}$")
# plt.grid()
# plt.legend(["no polymer", "5% polymer", "10% polymer", "20% polymer"], loc="lower right")
# plt.savefig("plots/experiments/11011/" + "Drag_comparison_with_no_polymer.png")
# ---------------------------------------------------------------------------------------------------------------------
# cond = np.load("results/arrays/experiments/4000/F_cond.npy")
# cond_time = np.load("results/arrays/experiments/4000/u_time.npy")
# dt = 400
# t_max = 1.15
# t_cond = np.linspace(0,t_max,num=460)
#
# fig = plt.figure(figsize=(25, 8))
# l1 = plt.plot(np.log(cond), label=r"FEniCSx", linewidth=2)
# plt.title("Condition of FP non-linear solver")
# plt.grid()
# plt.legend()
# plt.savefig("plots/experiments/4000/" + "condition_plot.png")
# ---------------------------------------------------------------------------------------------------------------------
tau = np.load(f"results/fp/experiments/arrays/{7}/tau.npy")
s11 = np.sum(np.load(f"results/fp/experiments/arrays/{7}/sigma11.npy"), axis=1)[:800]
s12 = np.sum(np.load(f"results/fp/experiments/arrays/{7}/sigma12.npy"), axis=1)[:800]
s21 = np.sum(np.load(f"results/fp/experiments/arrays/{7}/sigma21.npy"), axis=1)[:800]
s22 = np.sum(np.load(f"results/fp/experiments/arrays/{7}/sigma22.npy"), axis=1)[:800]
s11_ = np.load(f"results/fp/experiments/arrays/{7}/sigma11.npy")[:800]
s12_ = np.load(f"results/fp/experiments/arrays/{7}/sigma12.npy")[:800]
s21_ = np.load(f"results/fp/experiments/arrays/{7}/sigma21.npy")[:800]
s22_ = np.load(f"results/fp/experiments/arrays/{7}/sigma22.npy")[:800]
tr = s11_ + s22_
tr_cond = tr < 30
true_count = np.sum(tr_cond)
false_count = max(len(tr_cond) - true_count,0)

# Data for the histogram
categories = ['True', 'False']
counts = [true_count, false_count]
plt.bar(categories, counts, color=['green', 'red'])
plt.title(r"$tr(\sigma)<b ?$")
plt.savefig(f"results/fp/experiments/plots/{7}/" + "sigma_hist_plot.png")
fig, axs = plt.subplots(2, 2)

# Iterate over the matrix elements and plot each element in its respective subplot

# axs[0, 0].plot(time,s11)
# axs[0, 0].set_title(r"$\sigma_{11}$")
# axs[0, 1].plot(time,s12)
# axs[0, 1].set_title(r"$\sigma_{12}$")
# axs[1, 0].plot(time,s21)
# axs[1, 0].set_title(r"$\sigma_{21}$")
# axs[1, 1].plot(time,s22)
# axs[1, 1].set_title(r"$\sigma_{22}$")
# plt.tight_layout()
# plt.savefig(f"results/fp/experiments/plots/{7}/" + "sigma_plot.png")
# ---------------------------------------------------------------------------------------------------------------------
