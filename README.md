# SCHEDTUNE: A Heterogeneity-Aware GPU Scheduler for Deep Learning 

Note: SchedTune is not under active maintenance

# Citation:
H. Albahar, S. Dongare, Y. Du, N. Zhao, A. K. Paul and A. R. Butt, "SchedTune: A Heterogeneity-Aware GPU Scheduler for Deep Learning," 2022 22nd IEEE International Symposium on Cluster, Cloud and Internet Computing (CCGrid), Taormina, Italy, 2022, pp. 695-705, doi: 10.1109/CCGrid54584.2022.00079.


# Abstract: 
Modern cluster management systems, such as Kubernetes, support heterogeneous workloads and resources. However, existing resource schedulers in these systems do not differentiate between heterogeneous G PU resources-which are becoming a norm-and do not support GPU sharing-which is necessary to support emerging collocation of jobs and multi-tenant applications. Thus the systems suffer from low GPU resource utilization, higher queuing delays, and an increase in application makespan, i.e., the duration between the arrival of the first job and the completion of the last job of a workflow. This is especially a problem in supporting crucial deep learning (DL) applications. To this end, in this paper, we profile and analyze DL jobs on heterogeneous GPUs, investigate the interference caused by collocating jobs on GPUs, and use this information to predict the GPU memory demand and job completion times. We propose SCHEDTUNE, a machine-learning-based heterogeneity-aware scheduler that ensures higher GPU memory utilization and reduced out-of-memory (OOM) failures, while supporting improved makespan. Our evaluation shows that SCHEDTUNE GPU memory predictors and scheduler outperform the state-of-the-art predictors by achieving 81% higher GPU memory utilization, 100% detection and avoidance of OOM errors, and 17.5% reduction in makespan compared to the default Kubernetes scheduler.
keywords: {Deep learning;Processor scheduling;Memory management;Graphics processing units;Interference;Predictive models;Containers;Deep learning;Kubernetes;GPU sharing;Resource heterogeneity;Resource scheduling},

# URL: 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9826092&isnumber=9825913
