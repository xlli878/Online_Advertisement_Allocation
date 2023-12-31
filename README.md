# Online_Advertisement_Allocation
In this package, we have all files and datasets running on Matlab with the solver package Gurobi.

File 1: ''AdFairnessTargets.m'', which is the main code of our simulation with comparing five policies under different cases to generate Figure 2 and 3 and Table 3 in the paper. It has four sections:\
	Section 1: Basic setting, which contains the whole inputs.\
	Section 2: Main Simulation, which contains the optimization and simulation of five policies over different cases and samples.\
	Section 3: Plot, which generates Figure 2 and 3 in our paper.\
	Section 4: Debt Regression, which generates Table 3 in our paper (see Appendix J. Mean-Reverting Behavior of the DWO Policy).

File 2: ''AdTimeComparison.m'', which is the code for comparison of average solving times to generate Table 1 in the paper.

Dataset 1: ''AdSetting.mat'', which is the basic setting part and can be inputs to ''AdFairnessTargets.m''.

Dataset 2: ''AdFigure.mat'', which is the results part to show Figure 2 and 3 in the paper.

We show how to use the files and datasets as follows:
1. To qualitatively replicate the similar results of the paper under the same basic setting (the same inputs), please load the dataset ''AdSetting.mat'' by using ''load('AdSetting.mat')'', skip section 1 ''Basic setting'' in ''AdFairnessTargets.m'', comment the codes under the comments: ''construct the requirements eta'' in line 201-212 in section 2 ''Main Simulation'', and then run the remaining sections. Figure 1-3 corespond to Figure 2 (a)-(c) in the paper and Figure 4-8 corespond to Figure 3 (a)-(e) in the paper respectively.
 
	In ''AdSetting.mat'', we list the details of variables as follows:\
	n: Number of ads\
	m: Number of customer types\
	K: Maximum cardinality\
	lambda: Regularization parameter of fairness metric\
	S: Number of samples in each case\
	tratio: Ration of T to n to construct average budget\
	T: Number of total periods\
	CV: The variation of CP parameter, of which the length is 4\
	LF: The variation of LF, of which the length is 5\
	r: Revenues of ads under different customer types, an n*m matrix\
	V: Utilities of ads under different customer types, an n*m matrix\
	eV: Parameters in MNL choice model, which is the exponent of V, an n*m matrix\
	etaTarget: Click requirements of targets under different cases with varying CV and LF, an n*m*4*5 array\
	pSummary: Distribution of customer arrivals with varying CV, an m*4 matrix

	For Table 3 in the paper, we list the variables of the debt regression results as follows:\
	ttestdyd: DWO policy\
	ttestdyl: Fluid policy\
	ttestdylr: Fluid-R policy\
	ttestdylir: Fluid-I-R policy\
	ttestdyler: Fluid-E-R policy

	For Figure 2 and 3 in the paper, we list the related variables in the next part.

3. To show exactly the same figures: Figure 2 and 3 in our paper, please run the section ''Plot'' in the file ''AdFairnessTargets.m'' after loading the dataset ''AdFigure.mat'' by using ''load('AdFigure.mat')'', a part of our original results. Figure 1-3 corespond to Figure 2 (a)-(c) in our paper and Figure 4-8 corespond to Figure 3 (a)-(e) in the paper respectively. 

	In ''AdFigure.mat'', we list the details of variables as follows:\
	CV: The variation of CP parameter, of which the length is 4\
	LF: The variation of LF, of which the length is 5\
	The next five variables (each is a 4\*5 matrix with varying CP and LF) are the ratio between the expected FV and theoritical upperbound under different policies:\
	tul: Fluid policy\
	tulr: Fluid-R policy\
	tulir: Fluid-I-R policy\
	tuler: Fluid-E-R policy\
	tud: DWO policy\
	The next five variables (each is a 4\*5 matrix with varying CP and LF) are the ratio between the standard deviation of FV and theoritical upperbound under different policies:\
	tsul: Fluid policy\
	tsulr:  Fluid-R policy\
	tsulir: Fluid-I-R policy\
	tsuler: Fluid-E-R policy\
	tsud: DWO policy\
	We have a 5\*5\*4 array ''targetLossAvg'' with varying LF, policies and CP respectively of the average proportion of unfilled click-through requirements:\
	targetLossAvg(:,1,:): Fluid policy\
	targetLossAvg(:,2,:): Fluid-R policy\
	targetLossAvg(:,3,:): Fluid-I-R policy\ 
	targetLossAvg(:,4,:): Fluid-E-R policy\
	targetLossAvg(:,5,:): DWO policy\
	We have a 3*T*5 arry ''npcsumQuantile'' with varying quantiles, periods and policies respectively of the quantiles of the click-through sample-paths of one specific ad:\
	npcsumQuantile(:,:,1): Fluid policy\ 
	npcsumQuantile(:,:,2): Fluid-R policy\
	npcsumQuantile(:,:,3): Fluid-I-R policy\ 
	npcsumQuantile(:,:,4): Fluid-E-R policy\
	npcsumQuantile(:,:,5): DWO policy\
	In the first dimension of npcsumQuantile(i,:,:), index i = 1: 0.1-quantile, 2: 0.5-quantile, 3: 0.9-quantile.   

4. To simulate under other cases, please change the basic setting and run ''AdFairnessTargets.m'' and do not comment codes in line 201-212. Please note:\
	A. The basic setting of cardinality constraint K is no more than 2. If set ''K = 1'', please comment the codes in line 73-86; if set ''K = 3'', please uncomment the codes in line 89-105 and line 108.\
	B. When the variable ''tratio'' which controls the budget is too low, feasible requirements ''eta'' could not be found in some case. One way to fix this issue is to change ''unifrnd(0,1,n,m)'' in line 211 to ''unifrnd(0,a,n,m)'', where ''a'' is a suitable constant smaller than 1.\
	C. In this simulation, we assume all bid price ''b_i = 1'' for different ads and set the budget by the codes in line 195-196, which could be changed to other settings.\
	D. In this simulation, when t = 1, we can set all debts as one in DWO policy to assign an offer-set in period 1, which is better than do-nothing and does not affect the main theoritical results.\
	E. In Section ''Debt regression'', line 693, the sample size is 30 million, which is too large for other cases with smaller scales and should be adjusted for specific cases.\
	F. To save the results, please uncomment the last-line code, change the filename, and then run this line.

5. To simulate the comparison of computational time, please change the basic setting at the very beginning and then run ''AdTimeComparison.m''. Please note:\
	A. In this file, we set the maximum of carinality to be 5 under DWO policy, but it is 4 under Fluid model to avoid the out of memory. If run Fluid model under K = 5, please uncomment the codes in line 207-233 and modify the codes.\
	B. The results are recorded in variables: ''timeAvgDWO'', average time of DWO policy to find the targets; ''timeAvgFluid'', average time of Fluid policy to find the Fluid policy. Both of them are with varying the cardinality.\
	C. To save the results, please uncomment the last-line code, change the filename, and then run this line.

The paper's link is https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3538755.
