### Pytorch config
device_name = "mps"

### DATA CONFIG
gaussian_std = 25
patch_size = 64

### Optimization config
scs_info = {"max_iters": 400, "eps": 5e-3}
CLARABEL_info = {
    "max_iter": 100,
    "tol_gap_rel": 1e-4,
    "tol_feas": 1e-4,
    "verbose": False,
}

### Paths
data_dir = r"data/DIV2K/"
results_path = r"results/result.npz"
model_path = r"results/model.pth"
sample_path = r"data/sample/0784.png"
sample_dir = r"data/sample/"
plots_dir = r"results/plots"
scs_solver_08alpha_model = r"results/model_scs_solver_08Alpha.pth"
