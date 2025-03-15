import matplotlib.pyplot as plt


# FIGURE 8
# Line Plot
y= [0.0197,0.0199,0.0200,0.0201,0.0202]
x = [10, 20, 30, 40,50]
plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# plt.title('Figure 8')
plt.plot(x, y, label='Line',color='red',marker='x')
#plt.title('Line Plot Example')
plt.xlabel('Assigned Diagonal Element in R Matrix')
plt.ylabel('RMSE Latitude')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
plt.savefig('Figures_for_the_paper/eskf_RMSE_Latitude_GPU.png',dpi=400)
# plt.legend()
plt.show()


#  FIGURE 9
# ESKF RMSE PLOTS
# Line Plot
y= [0.0069,0.0079,0.0122,0.0132,0.0133]
x = [10, 20, 30, 40,50]
plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# plt.title('Figure 9')
plt.plot(x,y, label='Line',color='red',marker='x')
#plt.title('Line Plot Example')
plt.xlabel('Assigned Diagonal Element in R Matrix')
plt.ylabel('RMSE Longitude')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
# plt.legend()
plt.savefig('Figures_for_the_paper/eskf_RMSE_Longitude_GPU.png', dpi=400)
plt.show()



# # FIGURE 10
# # ESKF Result
# # Bar Chart
# x = [1, 2, 3, 4,5,6,7,8,9,10]
# y = [0.3,0.32,0.311,0.312,0.312,0.313,0.311,0.30,0.311,0.322]
# plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# # plt.title('Figure 10')
# plt.bar(x, y, color='blue')
# plt.xlabel('Iteration No.')
# plt.ylabel('Execution Time Per Data Point (seconds) ESKF')
# plt.ylim([0, 0.5])
# plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
# plt.savefig('Figures_for_the_paper/eskf_time_cuda_GPU.png', dpi=400)
# plt.show()

# Figure 10 (updated)
# ESKF Result
# Line Plot
x = [1, 2, 3, 4,5,6,7,8,9,10]
# y1 = [0.3,0.32,0.311,0.312,0.312,0.313,0.311,0.30,0.311,0.322]
y1 = [0.217, 0.211, 0.211, 0.207, 0.212, 0.207, 0.212, 0.208, 0.210, 0.233]
y2 = [0.61, 0.612,0.614,0.613,0.614,0.911,1.11,1.05,0.95,0.7]
plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
plt.title("CPU and GPU Time Comparison for ESKF prediction")
plt.plot(x, y1,label="gpu time", color='blue',marker="s")
plt.plot(x, y2,label="cpu time", color='red',marker="*")
plt.xlabel('Iteration Number')
plt.ylabel('Execution Time per Data Point ESKF')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
plt.ylim([0, 1.25])
plt.legend()
plt.savefig('Figures_for_the_paper/ESKF_GPU_CPU_Comparison_for_Prediction.png', dpi=400)
plt.show()


# FIGURE 11
# PF Result
# Line Plot
N = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]  # order of matrix
cpu = [0.014, 0.058, 0.147, 0.337, 0.325, 0.49, 0.79, 1.11]
gpu = [0.105, 0.080, 0.08, 0.085, 0.094, 0.107, 0.206, 0.233]
plt.figure(figsize=(9, 9), dpi=100)  # 900x900 pixels
plt.plot(N, cpu, label="CPU", color='red', marker='*', markersize=10)
plt.plot(N, gpu, label="GPU", color='blue', marker='s', markersize=8)
plt.xlabel('Matrix Order')
plt.ylabel('Execution Time')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
plt.legend()
plt.savefig('Figures_for_the_paper/ESKF_GPU_CPU_Comparison_for_Matrix_size.png', dpi=400)
plt.show()
