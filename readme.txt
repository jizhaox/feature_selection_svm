This code provides a Matlab implementation of normalized margin SVM with additive kernels, which corresponds to feature selection part in reference [1]. Please cite reference [1] if you use this code.

Additive kernels include chi-squared kernel, histogram intersection kernel, Jensen-Shannon kernel, Hellinger's kernel (Bhattacharyya kernel), and linear kernel. If linear kernel is used in this code, the feature selection method becomes the method in reference [2]. The definition of usual additive kernels can be found in reference [3]. 

This code was written by Ji Zhao (Email: zhaoji84@gmail.com). The latest version of this code can be found in his homepage:
https://sites.google.com/site/drjizhao/

1. Reference
===============
[1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

[2] Minh Hoai Nguyen, Fernando De la Torre. Optimal Feature Selection for Support Vector Machines. Pattern Recognition, 43(3), 584 - 591, 2010.

[3] Andrea Vedaldi, Andrew Zisserman. Efficient Additive Kernels via Explicit Feature Maps. IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(3): 480 - 492, 2012.

2. Installation
===============
(1) install IPOPT.
Download and unzip pre-compiled mex files for IPOPT 3.11.8.
http://www.coin-or.org/download/binary/Ipopt/

(2) install libSVM
Download and install libSVM 3.20.
https://www.csie.ntu.edu.tw/~cjlin/libsvm/
Note: Copy compiled "svmtrain" file in libSVM to current Matlab path because Matlab has a built-in function with the same name.

(3) install CVX (Optional)
Download and install CVX 2.1
http://cvxr.com/cvx/download/

(4) install VLFeat toolbox (Optional)
We use function "vl_homkermap" in VLFeat for additive kernels' feature mapping.
Download and install VLFeat 0.9.20 binary package
http://www.vlfeat.org/download.html

Note: CVX and VLFeat toolbox are optional. These toolboxs are needed if approximate solution for fast initialization is enabled, i.e., para.initByKernelAppro is true in function "featureSelectionAddKernel".

3. Usage
===============
(1) Run demo1.m for an example. If you can obtain three figures as that in folder RESULTS, the installation is successful.

(2) This code is tested on 32-bit Windows 7, Matlab 8.3 (2014a), IPOPT 3.11.8 pre-compiled mex files, libSVM 3.20, CVX 2.1 and VLFeat 0.9.20 binary package. Mex files for libSVM is compiled by Visual Studio 2013.

(3) This version is 0.9.0. Released on 11/26/2015.













