# Commented out IPython magic to ensure Python compatibility.
from turtle import shape
import numpy as np
import scipy.io
import sys
from numba import jit   #numba is not supported in Python version 3.13, therefor it is not running on my laptop
import time
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy as cp
from numba import jit
import time
import cupy as cp
#import StringIO
# 
#sio = StringIO.StringIO()
# 
# 
# #Loading all the variables
# 
# # % This data (LLA.mat) is generated from 'INS_Implementation_4.m' file.
# # % In 'INS_Implementation_4.m' file INS data (obtained from X Plane)is used for generating
# # % Latitude, Long and Height. (Although Latitude, Longitude and Altitude (LLA)  can be directly obtained
# # % from X-Plane, since in real time implementation on Nano-Jetson with ROS Nodes in Python, only onboard
# # % INS sensor data will be available, that data has to be  processed and converted into LLA in the absence of GPS  )
# # % In XPlane huge amount of  flight data  is generated but we specifically transform the
# # % INS sensor data into Latitude, Longitude, Altitude (LLA) format.
# # %  INS data consist of linear acceleration i.e.(ax,ay and
# # % az) and angular velocity i.e.(wx,wy,wz)
LLA=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/LLA.mat')
# # print(type(LLA))
# """ for key, value in LLA.items():
#     print(key, ' : ', value) """
# 
# 
# #LLA is the name of the dictionary, it has 05 keys 1)h 2)lat 3)lon 4) t and 5) Vn
LLA_h=LLA["h"]
LLA_h=cp.asarray(LLA_h)
print(type(LLA_h))
# """ print("The type of LLA_h is :",type(LLA_h))
# print("The shape of LLA_h is :",cp.shape(LLA_h))
# print(LLA_h)
#  """
LLA_lat=LLA["lat"]
LLA_lon=LLA["lon"]
LLA_t=LLA["t"]
LLA_t = cp.asarray(LLA_t)
LLA_Vn=LLA["Vn"]
# 
# #print("The type of LLA_lat is :",type(LLA_lat))
# #print("The shape of LLA_lat is :",np.shape(LLA_lat))
# #print(LLA_t)
# 
# # % latitude and longituide reference generated from mission planner and used in Xplane
Xplane_Pos=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/Xplane_Pos.mat')
Xplane_Pos_Lat_ref=Xplane_Pos["Lat_ref"]
Xplane_Pos_Long_ref=Xplane_Pos["Long_ref"]
# 
# 
# """ % This data (AccForces.mat) is generated from Diff_variables.m ,
# % which is generating acceleration from velocity data. The velocity data is
# % coming from X-Plane.
# 
# % Queries to resolve
# % ---------------------
# % 1) Can't we generate the acceleration data directly from X-plane ? why we
# % are converting velocity into acceleration. In reality, from the on-board INS
# % Sensor we will get the linear acceleration values directly.
# % Response: Omar has informed Sofia that we cannot generate acceleration
# % data directly from X-Plane, hence we need to perform this conversion.
# % --------------------------------------
# % This data (AccForces.mat) will be later used in 'A' i.e. System Matrix """
# 
AccForces=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/AccForces.mat')
# # Variable is An [ 3 x 2998]
AccForces_An=AccForces["An"]
# """ print("The type is :", type(AccForces_An))
# print("The shape is :", cp.shape(AccForces_An)) """
# 
# """D:\PeerJ GPU Paper Python Code\Hardware-Implementation-TAN-EsKF\DEM_ROI_sub.mat
# % DEM Height values generated from actual DEM data by applying filter on DEM height values,
# % the resulting variable is DEM_Z2,. Original DEM has a size of [3600x
# % 3600] whereas the DEM_ROI_sub.mat has a size of [695x1957] (Variable name is DEM_Z2) """
# 
DEM_ROI_sub=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/DEM_ROI_sub.mat')
DEM_ROI_sub_DEM_Z2=DEM_ROI_sub["DEM_Z2"]
# 
# #  % Lat, Long locations of the corresponding height (dimension of Seleceted_DEM = dimension of DEM_Z2)
# # % The size is [695x1957], and it is a cell array
Selected_DEM=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/Selected_DEM.mat')
# """ print("The type is :", type(Selected_DEM_Matlab))
#     for key, value in Selected_DEM_Matlab.items():
#     print(key, ' : ', value) """
Selected_DEM_in_Python=Selected_DEM["Selected_DEM"]
# print("The shape of Selected_DEM_in_Python is :", np.shape(Selected_DEM_in_Python))
# 
# 
# 
# # % Load the DEM for MAD...
# # % DEM heights of the selected region
DEM_loaded = DEM_ROI_sub_DEM_Z2
# 
# """ % Declare the constants as defined in the  research paper Stancic 2010...
# % some values were not defined in the paper, they have been taken from
# % internet (which constants are taken from interent, Sofia will write here)
# 
# % format long
# 
# %A matrix has been developed from Stanic Paper """
Rm =   6335.439e3
g  =   9.8
Rp =   6399.594e3
ws =   7.3e-5
Re =   6371.0072e3
B = 0.1 #% beta
KK = 4
# 
# # % Window Size
W = 10
# 
# #% Counter Variable
c = 10
# 
# #%% Compute the derivatives...
VE = LLA_Vn[0,:]
VE = cp.asarray(VE)
VN = LLA_Vn[1,:]
VN = cp.asarray(VN)
VU = LLA_Vn[2,:]
VU = cp.asarray(VU)
VD = -VU
VD = cp.asarray(VD)

Lambda = LLA_lon
phi = LLA_lat

print(type(phi))
phi = cp.asarray(LLA_lat)
print(type(phi))
# #print("Printing phi:\n",phi)
# 
# #%% Lambda_dot
# 
Lambda_dot = VE/((Rp+LLA_h)*cp.cos(phi)) #%equation (2) of Staninc Paper
# #print("I am printing Lambda_dot:", Lambda_dot)
# 
# #%% Phi_dot
# 
phi_dot = VN/(Rm+LLA_h)
# #print("I am printing phi_dot",phi_dot)
h_dot = VU
# 
N = len(VU)
# #print("I am printing ",N)
# 
# """ %% Other initializations prior to
# % ErKF execution ...
#  """
delta_x = cp.zeros((15,1))
# 
# 
# """
# %---------------------------------------
# % Initialize the Kalman Filter variables
# %---------------------------------------
#  """
delta_x[[0,1,2],0] = [0.05, 0, 0]
# 
I = cp.eye(15)
# 
Q = (2e-3)*cp.eye(15)
# 
Q[0,0] = 1e-3
# 
Q[1,1] = 12
# # print("I am printing Q:",Q)
P = 0.5*cp.eye(15)
# 
# 
# #% Transformation Matrix ...
# 
a=cp.eye(3)
b=cp.zeros([3,12])
# 
Ck=cp.hstack((a,b))
# 
# """ % Measurement Noise Covariance
# 
# Rk = 0.07*eye(3); % dimension equal to #. of measurements
# 
# % For height measurement error covariance increased """
# 
Rk = cp.diag([50, 50, 200])
# 
# 
# 
# #% INS Position....(this is coming from LLA.mat file), these are the
# #% Latitude, Longitude values computed from INS data
# 
Lins = LLA_lat
# 
lins = LLA_lon
# 
hins = LLA_h #% Remember, hins has already been initialized i.e. the height from which the plane takes off is the initial value
# print('The shape of Lins is:',np.shape(Lins))
# #print("The values of Lins are:\n",Lins)
# 
# """ % From MAD Algorithm
# % ==================
# % ******************
# 
# %% Contains the Data from the X-Plane [63735 x 18]
# The mat file name is D:\PeerJ GPU Paper Python Code\Hardware-Implementation-TAN-EsKF\DataV4.mat, the variable name is also DataV4"""
DataV4=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/DataV4.mat')
DataV4_DataV4=DataV4["DataV4"]
# """ print("The type of DataV4_DataV4 is:",type(DataV4_DataV4))
# print("The shape of DataV4_DataV4 is:",np.shape(DataV4_DataV4))
# print("The DataV4 is:\n",DataV4_DataV4) """
# 
# 
# """ % 63735 are the total number of samples from
# % the X-Plane data. 18 are the different sensor data values """
# 
# 
Data =DataV4_DataV4[1:2001,:] # %This is the data that has been selected for prediction from the ErKF
#print(Data)
# #np.savetxt('abc.txt', Data)
# """ %The data is from the initial leg of the trajectory
# %We will call this data ROI """
# 
# 
# """ %---------------------------
# % Obtain the Radar Heights
# %--------------------------- """
# # The mat file name is GroundData.mat, the variable name is grounddata
GroundData=scipy.io.loadmat('/home/bilal/Hardware-Implementation-TAN-EsKF/GroundData.mat')
# #load GroundData.mat #%[63735 x 1] obtained from X-Plane
GroundData_grounddata=GroundData["grounddata"]
# 
grounddata = GroundData_grounddata[1:2001,:] #% getting the RADAR values of ROI
# 
Rad_height = grounddata
# #np.savetxt('Rad_height.txt', Rad_height)
# 
# 
# #%% Sample Number
# 
Ns = 201 # %This is the window size on which we will run the MAD Algorithm
# 
# #initialization
# # np.zeros((15,1))
L_Tercom = cp.zeros((1,Ns))
# 
l_Tercom = cp.zeros((1,Ns))
# 
h_Tercom = cp.zeros((1,Ns))
# 
# """ % ******************
# % ==================
# 
# % 2. Acceleration Forces """
# 
fE =  AccForces_An[0,:]
fE =  cp.asarray(fE)
fN =  AccForces_An[1,:]
fN = cp.asarray(fN)
fD =  -AccForces_An[2,:]
fD = cp.asarray(fD)
# print("The shape of fD is:",np.shape(fD))
# 
# 
# """
# np.savetxt('fE.txt', fE)
# np.savetxt('fD.txt', fD) """
# 
# #%% Error State Matrix...
# 
# #%This must be plotted (fourth plot)
Delta_x = cp.zeros((3,Ns)) #%[del_Lat, del_Long, del_h]
iteration = 1
# #% This is the counter variable
c = 0
d = cp.zeros((1,10))
# 
start_time=600
end_time=800
# 
LAT=cp.zeros((1,Ns))
LON=cp.zeros((1,Ns))
HGT=cp.zeros((1,Ns))

#phi =2
# 
# 
# 
# 
# 
# 
# #print("Alhumdulillah all is well")

# Commented out IPython magic to ensure Python compatibility.
#------------------------------------------------------------------------
#                  This code must be made efficient
# #----------------------------------------------------------------------


@jit(nopython=True)
def optimized_go_fast(hDB, hMeas, hMeas_bar, N, R, C):
    """Optimized MAD computation with preallocation and reduced operations."""
    MAD = np.zeros((R, C))
    for m in range(1, C):
        hDB_col = hDB[:, m-1]  # Extract column for reuse
        hDB_mean_col = np.convolve(hDB_col, np.ones(N) / N, mode='valid')  # Efficient mean calculation
        for n in range(1, R):
            index = n + np.arange(N)
            hDB_chunk = hDB_col[index - 1]  # Slice only once
            hDB_mean = hDB_mean_col[n - 1]  # Precomputed mean
            T = (hMeas - hMeas_bar) - (hDB_chunk - hDB_mean)
            MAD[n-1, m-1] = np.mean(np.abs(T))  # Efficient mean absolute deviation
    return MAD

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#start_time=time.time()
#start_execution_time=time.time()
R_test = [10,20,30,40,50]
#filename = "calculation_result.txt"
#with open(filename, "w") as file:
results= []


#for m in range(len(R)):
 #Rk = cp.diag([R[m],R[m],200])
for i in range(start_time, end_time+10, 10):
    print(i)
    """ %     while c~=11




#     %Why the iteration is starting from 600??
#     % Since the variations in the lat and long starts from 600 to 800,
#     % remember there are 2000 samples, and we are only selecting 200 on
#     % which the prediction will be performed

#     % 1 . Compute Complex Matrix Co-efficients
    """
    start_execution_time=time.time()
    A1 = -phi[0,i-1]/(Rm+LLA_h[0,i-1])  # phi is a numpy.ndarray of size (1,2000), phi[0,i-1] is the value of phi at [0,i-1] index location
    #print(type(A1))
    #A1 = cp.asarray(A1)
    A2 = 1/(Rm+LLA_h[0,i-1])
    #print(type(A2))
    #A2 = cp.asarray(A22)
    A3 = Lambda_dot[0,i-1]*LLA_t[i-1,0]*g*phi[0,i-1]
    #print(type(A3))
    #A3 = cp.asarray(A3)
    A4 = -(Lambda_dot[0,i-1])/(Rp+LLA_h[0,i-1])
    #print(type(A4))
    #A4 = cp.asarray(A4)
    A5 = (1/(Rp+LLA_h[0,i-1])*cp.cos(phi[0,i-1]))
    #print(type(A5))
    #A5 = cp.asarray(A5)
    #A6 = -VE(i)*cos(phi(i))*(2*ws+Lambda_dot(i)*( sec(phi(i))^2   ));
    A6 = -VE[i-1]*cp.cos(phi[0,i-1])*(2*ws+Lambda_dot[0,i-1]*(cp.square(1/cp.cos(phi[0,i-1]))))
    #print(type(A6))
    #A6 = cp.asarray(A6)
    A7 = ((VE[i-1]*Lambda_dot[0,i-1]*cp.sin(phi[0,i-1]))/(Rp+LLA_h[0,i-1])) - ((VD[i-1]*phi_dot[0,i-1]/(Rm+LLA_h[0,i-1])))
    #print(type(A7))
    #A7 = cp.asarray(A7)
    A8 = VD[i-1]/(Rm+LLA_h[0,i-1])
    #print(type(A8))
    #A8 = cp.asarray(A8)
    A9 = -2*(ws+Lambda_dot[0,i-1])*cp.sin(phi[0,i-1])
    #print(type(A9))
    #A9 = cp.asarray(A9)
    A10 = phi_dot[0,i-1]
    #print(type(A10))
    #A10 = cp.asarray(A10)
    A11 = 2*ws*((VN[i-1]*cp.cos(phi[0,i-1]))-(VD[i-1]*cp.sin(phi[0,i-1])))+Lambda_dot[0,i-1]*VN[i-1]*(1/cp.cos(phi[0,i-1]))
    #A11 = cp.asarray(A11)
    #print(type(A11))
    A12 = (-Lambda_dot[0,i-1]/(Rp+LLA_h[0,i-1]))*(VD[i-1]*cp.cos(phi[0,i-1])+VN[i-1]*cp.sin(phi[0,i-1]))
    #print(type(A12))
    #A12 = cp.asarray(A12)
    #I HAVE CHECKED ALL VALUES TILL A12, I must check the remaining values from A13 to A33
    A13 = (2*ws+Lambda_dot[0,i-1])*cp.sin(phi[0,i-1])
    #print(type(A13))
    #A13 = cp.asarray(A13)
    #print(A13)
    A14 = (1/(Rp+LLA_h[0,i-1]))*(VD[i-1]+VN[i-1]*LLA_t[i-1,0]*g*phi[0,i-1])
    #print(type(A14))
    #A14 = cp.asarray(14)
    #print(A14)
    A15 = ((2*ws)+Lambda_dot[0,i-1])*cp.cos(phi[0,i-1])
    #print(type(A15))
    #A15 = cp.asarray(A15)
    #print(A15)
    A16 = 2*ws*VE[i-1]*cp.sin(phi[0,i-1])
    #print(type(A16))
    #A16 = cp.asarray(A16)
    #print(A16)
    A17 = (VN[i-1]/(Rm+LLA_h[0,i-1])*phi_dot[0,i-1])+(VE[i-1]/(Rp+LLA_h[0,i-1])*Lambda_dot[0,i-1]*cp.cos(phi[0,i-1]))+(KK-2)*(g/Re)
    #print(type(A17))
    #A17 = cp.asarray(A17)
    #print(A17)
    A18 =  -2*phi[0,i-1]
    #print(type(A18))
    #A18 = cp.asarray(A18)
    A19 =  -2*(ws+Lambda_dot[0,i-1])*cp.cos(phi[0,i-1])
    #print(type(A19))
    #A19 = cp.asarray(A19)
    #print(A19)
    A20 = -ws*cp.sin(phi[0,i-1])
    #print(type(A20))
    #A20 = cp.asarray(A20)
    A21 = (-Lambda_dot[0,i-1])/(Rp+LLA_h[0,i-1])*cp.cos(phi[0,i-1])
    #print(type(A21))
    #A21 = cp.asarray(A21)
    #print(A21)
    A22 = 1/(Rp+LLA_h[0,i-1])
    #A22 = cp.asarray(A22)
    #print(type(A22))
    #A22 = cp.asarray(A22)
    #A22 = cp.asarray(A22)
    A23 = -(ws+Lambda_dot[0,i-1])*cp.sin(phi[0,i-1])
    #print(type(A23))
    #A23 = cp.asarray(A23)
    A24 = phi_dot[0,i-1]
    #print(type(A24))
    #A24 = cp.asarray(24)
    A25 = (phi_dot[0,i-1]/(Rm+LLA_h[0,i-1]))
    #print(type(A25))
    #A25 = cp.asarray(25)
    A26 = -1/(Rm+LLA_h[0,i-1])
    #print(type(A26))
    #A26 = cp.asarray(A26)
    A27 =  (ws+Lambda_dot[0,i-1])*cp.sin(phi[0,i-1])
    #print(type(A27))
    #A27 = cp.asarray(A27)
    A28 =  (ws+Lambda_dot[0,i-1])*cp.sin(phi[0,i-1])
    #print(type(A28))
   # A28 = cp.asarray(A28)
    A29 =  -(ws*cp.cos(phi[0,i-1])+Lambda_dot[0,i-1]*(1/cp.cos(phi[0,i-1])))
    #print(type(A29))
    #A29 = cp.asarray(A29)
    #print(A29)
    A30 =  (Lambda_dot[0,i-1]/(Rp+LLA_h[0,i-1]))*cp.sin(phi[0,i-1])
    #print(type(A30))
    #A30 = cp.asarray(A30)
    A31 =  -(LLA_t[i-1,0]*g*phi[0,i-1])/(Rp+LLA_h[0,i-1])
    #print(type(A31))
    #A31 = cp.asarray(A31)
    A32 =  -phi_dot[0,i-1]
    #print(type(A32))
    #A32 = cp.asarray(A32)
    A33 =  -(ws+Lambda_dot[0,i-1])*cp.cos(phi[0,i-1])
    #print(type(A33))
    #A33 = cp.asarray(A33)
    #print(A33)
    """
    """


    # 3. Complete Matrix 'A'
    #A=cp.zeros((15,15))
    #print("Shape of A is:",np.shape(A))
    #A[0][0]=0
    #A[0][1]=0
    #A=np.array([(1,2,3),(4,5,6)])
   
    

    z= A11.astype(int)
    g= A12.astype(int)
    print(z)
    print(type(z))
    #A11 = 2
    #A12 = 3
    #w = cp.array([[A11,A12],[3,A14]])
    #print(w)
    #print(type(B))
    B = cp.asarray(B)
    print(type(B))
    #print(type(3))
    A = cp.zeros((15,15))
    one=cp.array(1)
    mone = cp.array(-1)
    zero = cp.array(0)
    print(type(zero))
    A = cp.array([
    [zero, zero, A1, A2, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero],
    [zero, zero, A4, zero, A5, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero],
    [zero, zero, zero, zero, zero, zero, mone, zero, zero, zero, zero, zero, zero, zero, zero],
    [A6, zero, A7, A8, A9, A10, zero, -fD[i-1], fE[i-1], zero, zero, zero, one, zero, zero],
    [A11, zero, A12, A13, A14, A15, fD[i-1], zero, -fN[i-1], zero, zero, zero, zero, one, zero],
    [A16, zero, A17, A18, A19, zero, -fE[i-1], fN[i-1], zero, zero, zero, zero, zero, zero, one],
    [A20, zero, A21, zero, A22, zero,zero, A23, A24, mone, zero, zero, zero ,zero, zero],
    [zero, zero, A25, A26, zero, zero, A27, zero, A28, zero, mone, zero, zero, zero,zero],
    [A29, zero, A30, zero, A31, zero, A32, A33, zero, zero, zero, mone, zero, zero, zero],
    [zero, zero, zero, zero, zero, zero, zero, zero, zero, -B, zero, zero, zero, zero, zero],
    [zero, zero, zero, zero, zero, zero, zero, zero, zero,zero, -B, zero,zero , zero, zero],
    [zero,zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, -B, zero, zero, zero],
    [zero, zero, zero, zero, zero,zero, zero, zero, zero,zero, zero,zero, zero, zero, zero],
    [zero,zero, zero,zero, zero,zero,zero, zero,zero, zero, zero, zero, zero ,zero, zero],
    [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero]
    ])

    #A = cp.asarray(A)
    #print("tHE MATRIX A Is:/n",A)
    """ % Complex Coefficients programmed...A11, A12...
#     % Need to define the complete matrix A
#     % Next Insert fd, fn , fu...

#     % Change Variable According to Paper..
#     %According to the block diagram 'F' is the non-linear system matrix
#     % whereas A is the linearized system Matrix """
    F=A

    """ % Calculate the Model Discretization ....
#      % % Compute Delta 't':
    """
    if i == start_time: #LLA_t[i-1,0]
        dt = LLA_t[0,0]
    else:
        dt = LLA_t[i-1,0]-LLA_t[i-2,0]
    #print('The value of dt is:',dt)

    """   %--------------------------------------------------
#     % This is the Kalman Filter Prediction Stage
#     %
#     %--------------------------------------------------
    """

    PHI = I+np.multiply(F,dt)  #%Reference of this equation will be provided by Sofia
    PHI = cp.asarray(PHI)
    #print(PHI)
    """ % Why there are two  PHI_K_1 matrices? I think this condition is
#     % redundant PHI_k_1 is never updated in the code, it has the same value
#     % as PHI ......Sofia must check this condition
     """
    if i == start_time:
        PHI_K_1 = PHI
    else:
        PHI_K_1 = PHI_k_1
    """ % (PHI_K_1) is the A matrix, A matrix is
#     % the linearized matrix (details are availabe in the OneNOTE Handwritten NOTES)


#     % Predict the state vector...
     """
    delta_x = (PHI_K_1)*(delta_x)  #%delta_x is [15 x 1] matrix
     #% Predict the error state covariance...
    P = (PHI_K_1)*P*cp.transpose((PHI_K_1))+Q

    """ % Q can also be updated within the 'for' loop..
#     % Q has not been updated ,
#     % in some cases Q is also updated but we are holdin it constant in this code


#     % Store the previous PHI matrix... """
    PHI_k_1 = PHI #% previous values...

    """
#     %--------------------------------------------------------------------
#     %  We have to obtain the Latitude(ATERCOM) and Longitude(ATERCOM )from the MAD algorithm
#     %  by feeding in the terrain height i.e.(h_INS - h_RADAR )
#     %
#     %--------------------------------------------------------------------


#     % obtain the sensor measurements ...

#     % Read the height data from RADAR Altimeter
#     % =========================================

#     % Single value from the RADAR Altimeter ...
     """
    #print("The shape of Rad_height is",np.shape(Rad_height))
    h_RADAR = Rad_height[i-1,0]
    #print(h_RADAR)
    #% Strip-Length (This is Sofia's idea, to convert a single measurement into a window size of length 'SL' i.e. Strip Length)

    SL = 10

    #% Simulate window-size of 10 points
    #%In order to make the simulation realistic we are adding noise by
    #%generating random numbers

    n = 0.4*np.random.randn(SL,1)
    #print(n)
    """ % From a single RADAR Altimeter reading we are generating 10 arbitrary readings
#     % These readings are generated by extending one sensor measurement into a
#     % strip length of 10
#     % Generate Multiple Points from Radar Altimeter
     """

    #h_RADAR_SL = np.multiply(np.ones([SL,1]),h_RADAR)+n
    #h_RADAR_SL =  grounddata(i:i+[1:10]) # This is Sofia's concpet of getting 10 consecutive values from the grounddata, This code is wrong it is just taking 02 values
    #print('type of grounddata is',np.shape(grounddata))
    h_RADAR_SL =  grounddata[i-1:i+SL-1:1,0] #%This is MBK's correction, picking 10 consective values from the grouddata
    #print(h_RADAR_SL)
    #array2[0:3:2,0]
    """ %-------------------------------------------------------------
#     % We are using Bhukya paper to implement the MAD algorithm
#     % Compute statistics of the strip to compute MAD
#     %-------------------------------------------------------------

#     % I think the sign must be reversed i.e. hins(i)-h_RADAR_SL
#     % Sofia has to think and write her response here
    """

    #print('shape of hins',np.shape(hins))
    print(type(hins))
    print(type(h_RADAR_SL))
    hMeas =cp.asnumpy(hins[0,i-1])-h_RADAR_SL  #% The INS value is not being subtracted here...
    """    % h_RADAR_SL   [10 x 1]
#     % Subtracting a single value i.e. hins(i)
#     % from the  h_RADAR_SL """

    hMeas_bar = np.mean(hMeas)#  % we are calculating the  mean height of the terrain over which the plane is moving
    Temp1 = hMeas-hMeas_bar   #% Temp1 [10 x 1]= hMeas [10 x 1] - hMeas_bar [1 x 1]

    """ % In real practice we need to read the complete DEM
#     % to search in it for comparing the "strip-value" but this is
#     % computationally very expensive operation, therefore we need to cut
#     % down the DEM. We have already done this in the code (line #58)
    """
    sz = np.shape(DEM_loaded) #%the size of trimmed DEM
    #sz[0] --> number of rows in DEM_loaded
    #sz[1] --> number of columns in DEM_loaded


    #% Store the important parameters

    R= sz[0]
    C= sz[1]

    """ print("Rows :",R)
    print("Columns:",C)
    print(len(h_RADAR_SL))
    """
    k=np.arange(0,len(h_RADAR_SL))


    hDB = DEM_loaded

    N =len(h_RADAR_SL)
    #N[0] will contain the length of h_RADAR_SL
    print("N=",N)

    """ % Increase the size of hDB matrix (by adding N-1 number of rows)
    so that it does not produce error when the strip of length N is convolved vertically

#     % ---------------------------------------------------------------------
#     % Just to handle the strip of length 10 ;  for matching with the DEM values
#     % 100 is applied here to generate the maximum value closer to infinity
#     % We can change it with the other value to infinity
#     %----------------------------------------------------------------------
    """
    print('The shape of hDB is:',np.shape(hDB))
    #hDB[R+1:R+N-1,:]=np.amax(hDB)+100 #This is the Matlab Code for extneding the hDB matrix
    empty_array=np.empty((N-1,C),float)  #defining an empty matrix
    empty_array[:,:]=np.amax(hDB)+100    #initializing all entries of the empty matrix with max(hDB)+100
    hDB=np.append(hDB,empty_array,axis=0)

    """ % Temp1 is stored and saved.

#     % Work for hDB (Considering the heights from the database...)


#     % We need to search the strip heights in the DEM

#     % Initialize MAD... """
    MAD = np.zeros([R,C]) #% Size equal to selected DEM ROI
    #for i in range(start_time, end_time+10, 10):
    print("The shape of MAD is:",np.shape(MAD))
    #This is just for testing must remove it later
    #C=2
    #R=2

    # This loop takes 400 seconds to execute....
    # =========================================
    # =========================================
    #start_execution_time=time.time()
    #for m in range(1,C):  #% for each column
        #for n in range(1,R):
            #% Compute the hDB_mean for it
            #index = n+k #% Change the index
            #% Compute hDB for this chunk
            #hDB_mean = np.average(hDB[index-1,m-1])  #% Mean of Heights compared in DEM with the strip.
            #--------------------------------
            #% Computing  the Sum
            #--------------------------------
            #hDB_ind = hDB[index-1,m-1]
            #print("I am in the loop")
            #% Implement the MAD
            #% hMeas is always erroneous reading due to fusion of INS data
            #% with Radar based measurement
            #T = (hMeas-hMeas_bar)-(hDB_ind-hDB_mean)
            ##Tabs = np.absolute(T)
            #MAD[n-1,m-1]=(1/N)*np.sum(Tabs) #% In this MAD (n,m) variable mean absolute deviation of the heights are stored, we hvae
            #% find the corresponding value of
            #% the (LAT, LONG) using the minimum  heigt
    MAD=optimized_go_fast(hDB, hMeas, hMeas_bar, N, R, C)

    #% Compute the minimum value,
    #print("--- %s seconds ---" % (time.time() - start_execution_time))
    Height_min = np.amin(MAD)
    """ % Find the Index i.e. row and column number
#     % The 'find ' function determined the [r,c] index location where the 'Height_min'
#     % is found in the MAD MATRIX
     """
    #[r ,c] = find(MAD==Height_min)
    result = np.where(MAD == np.amin(MAD))
    r=result[0]
    c=result[1]
    print("The value of r is:",r[0])
    print("The value of c is:",c[0])

    #% Display the Latitude and Longitude

    Pos = Selected_DEM_in_Python[r[0],c[0]] #% This is the lookup table, output will be Latitude(TERCOM) and Longitude (TERCOM)
    print("I am printing Pos",Pos)
    #% Latitude(TERCOM)

    Lat = Pos[0,1]

    #% Longitude(TERCOM)

    Long = Pos[0,0]

    #% Store Ltercom, ltercom,htercom
    """ % Original L_Tercom is initialized to [1x201] entries. In the following
#     % code the value of i=600:10:800 L_Tercom(600) is assigned Lat value
#     % the size of L_Tercom matrix is automatically increased
     """
    print("Lat:",Lat)
    print("Long:",Long)
    print("The shape of L_Tercom is:", np.shape(L_Tercom))
    print("--- %s seconds ---" % (time.time() - start_execution_time))
    t = time.time()-start_execution_time
    results.append(t)
    #with open(filename, "w") as file:
    #file.write(f"The execution time per data point is {t}\n")

    #In Python assigning a big index value will not automatically
    # increase the size of matrix
    # e.g. in Matlab if a=[1,2,3] and we assign, a(8)=10 then the size of matrix
    # a is automatically increased the resulting a matrix will be
    # a=[1,2,3,0,0,0,0,10] , This is not the case in Python hence
    # the following Matlab code
    # L_Tercom[0,i-1] =  Lat
    # l_Tercom[0,i-1] =  Long
    # has to be modifed, i starts from 600 where as the size of L_Tercom is (1,201)
    L_Tercom[0,i-start_time] =  Lat
    l_Tercom[0,i-start_time] =  Long
    #%     htercom(i) =  Height_min;  %h_TERCOM is missing in the block diagram, MBK must insert it
    h_Tercom[0,i-start_time]= hDB[r[0],c[0]]

    """ % ================================================================
#     % KALMAN Filter Correction Stage
#     % ================================================================
    array1=np.array([[1,200],[3,4]])
#     %Measurement matrix """

    hins = hins.get()
    L_Tercom = L_Tercom.get()
    l_Tercom = cp.asnumpy(l_Tercom)
    h_Tercom = cp.asnumpy(h_Tercom)
    zk = [[Lins[0,i-1]-L_Tercom[0,i-start_time]],
          [lins[0,i-1]-l_Tercom[0,i-start_time]],
          [hins[0,i-1]-h_Tercom[0,i-start_time]]]
    print('zk')
    print(type(Lins))
    print(type(lins))
    print(type(hins))
    print(type(h_Tercom))
    print(type(l_Tercom))
    print(type(L_Tercom))

    #% Compute the Innovation Covariance Sk:
    #print('shape of Rk is:',np.shape(Rk))
    #print('shape of Ck is:',np.shape(Ck))
    #print('shape of P is:',np.shape(P))
    Sk = Rk+np.matmul(np.matmul(Ck,P),np.transpose(Ck))

    #% Kalman Gain

    K = np.matmul(np.matmul(P,np.transpose(Ck)),np.linalg.inv(Sk))

    #% Calculate the innovation
    print(type(delta_x))
    print(type(Ck))
    print(type(zk))
    #print(zk)

    

    #a = np.array(zk)
    Ik = np.array(zk) - np.matmul(Ck.get(),delta_x.get())
    #% Compute the corrected/updated error-state
    #% *****************************************
    print(type(K))
    delta_xhat = np.matmul(K.get(),Ik)  #% This is the output of EsKF

    delta_x = delta_xhat

    #% Compute the error-state covariance matrix

    P_hat = np.matmul((I-np.matmul(K,Ck)),P)

    #% ******************************************

    P = P_hat

    #% Construct the Transformation Matrix ...

    TF = Ck
    # % Extract the relevant position variables ...
    print('shape of Ck is:',np.shape(Ck))
    print('shape of delta_x is:',np.shape(delta_x))
    intermediate_variable = np.matmul(Ck.get(),delta_x)
    print(type(Delta_x))
    Delta_x[:,i-start_time]=cp.asarray(intermediate_variable[:,0])
    Delta_x = Delta_x.get()
    #array7[:,0]=array6[:,0]




    X_c = [[Lins[0,i-1]-Delta_x[0,i-start_time]], #% Latitude...
           [lins[0,i-1]-Delta_x[1,i-start_time]], #% Longitude...
           [hins[0,i-1]-Delta_x[2,i-start_time]]] #% Height...

    
    # converting list to array
    X_c = np.array(X_c)

    LAT[0,i-start_time] = X_c[0,0]
    LON[0,i-start_time] = X_c[1,0]
    HGT[0,i-start_time] = X_c[2,0]

    #disp(iteration);
    delta_x = cp.asarray(delta_x)
    hins = cp.asarray(hins)
    L_Tercom = cp.asarray(L_Tercom)
    Delta_x = cp.asarray(Delta_x)
    iteration = iteration+1
    c = c+1


#print("--- %s seconds ---" % (time.time() - start_execution_time))
print(iteration)
print(type(LAT))
print(type(LON))

from sklearn.metrics import mean_squared_error # Import mean_squared_error from sklearn.metrics
import math # Import math library
msex = mean_squared_error(L_Tercom.get(), LAT.get())
rmsex = (msex)**(1/2)
print("The calculated Root Mean Square Error (RMSE) is: " + str(rmsex))

msey = mean_squared_error(l_Tercom, LON.get())
rmsey = (msey)**(1/2)
print("The calculated Root Mean Square Error (RMSE) is: " + str(rmsey))

# For different values of Rk matrix in diagonal ,the rmse values obtained are:
rmsex_values = [0.0197,0.0199,0.0200,0.0201,0.0202]
rmsey_values = [0.0069,0.0079,0.0122,0.0132,0.0133]

print('*****************************************************************************************')
# Data Pre-processing
# Plotting Longitude 
# GPS Position
print(Xplane_Pos_Long_ref)
print(Xplane_Pos_Long_ref.shape)
selected = np.linspace(0, 1999, 21, dtype=int)
arr2 = Xplane_Pos_Long_ref[selected]
#print(arr2)
print('GPS Position')
arr2 = arr2.reshape(1,21)
print(arr2)
# Predicted Longitude
ind= np.nonzero(LON)
LON= LON[ind]
print('Predicted Longitude')
print(LON)

# LATITUDE
print(Xplane_Pos_Lat_ref)
print(Xplane_Pos_Lat_ref.shape)
selected = np.linspace(0, 1999, 21, dtype=int)
arr3 = Xplane_Pos_Lat_ref[selected]
#print(arr2)
print('GPS Position')
arr3 = arr3.reshape(1,21)
print(arr3)
# Predicted Longitude
ind= np.nonzero(LAT)
LAT= LAT[ind]
print('Predicted Latitude')
print(LAT)


#LON = list(LON)
#print(LON)
filename = "calculation_result.txt"
with open(filename, "w") as file:
#file.write(f"The execution time per data point is :\n")
 file.write(f"The execution time per data point is {results}\n")
 file.write(f"The value of R matrix diagonal element is {50}\n")
 file.write(f"The value of rmse latitude is {rmsex}\n")
 file.write(f"The value of rmse longitude is {rmsey}\n")
 file.write(f"Ground trurh GPS values longitude is {arr2}\n")
 file.write(f"ESKF Predicted values latitude are {LON}\n")
 file.write(f"Ground trurh GPS values latitude is {arr3}\n")
 file.write(f"ESKF Predicted values latitude are {LAT}\n")
 file.write(f"The values of covariance matrix tested are {R_test}\n")
 file.write(f"The corresponding rmse values of longitude {rmsex_values}\n")
 file.write(f"The corresponding rmse values of latitude  {rmsey_values}\n")
 
