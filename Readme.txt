1) The ESKF_CUPY_LOG_V2.py is the main file.
2) AFter executing this file the data will be saved in the file .Calculation_result.txt..................
3) plotting_results_offline.py file contains all the code to plot the figures included in the paper. THe data to be plotted 
is copied manually from the file.ESKF_CUPY_LOG_V2.py by generating results with different values of ESKF parameters.
This includes the sensitivity of Measurement covairiance matrix R and its effects on the RMSE values of latitude and longitude prediction.

4) The execution time per measurement update (i.e., for a single prediction is calculated within the file ESKF_CUPY_LOG.py and the result is plotted 
   for first ten iterations for both cpu as well as gpu based execution.
5) The laitude and longitude error curves have plotted by computing the error from gps based ground true values and the respective 
    predicted position estimates from the ESKF
6)  As the ESKF implementation mostly involves matrix multiplication, the cpu and gpu time comparison plot is genertated to show the difference
     in computation time with numpy blas library and cupy (cublas) library for different orders of matrix multiplication. ...................