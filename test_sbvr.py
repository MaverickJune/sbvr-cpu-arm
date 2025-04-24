import sys
import time
import torch
import sbvr
import copy
import os

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def print_tensor(tensor, name="Tensor"):
    print(b_str(name) + ": " 
          + g_str("shape: ") + str(tensor.shape))
    print(tensor)
    
def load_or_create_tensor(name, shape, device):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return torch.load(file_path).to(device)
    else:
        tensor = torch.randn(shape, device=device, dtype=torch.float16) * 0.3
        torch.save(tensor, file_path)
        return tensor

def load_or_create_sbvr(name, shape, device, num_sums, verbose_level=0):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return sbvr.load(file_path, device=device, verbose_level=verbose_level)
    else:
        tensor = load_or_create_tensor(name, shape, device)
        sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                                device=device, verbose_level=verbose_level)
        sbvr_tensor.save(file_path)
        return sbvr_tensor

def get_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    errors = tensor1 - tensor2
    mse = torch.mean(errors ** 2).item()
    max_error = torch.max(errors).item()
    min_error = torch.min(errors).item()
    std_dev = torch.std(errors).item()
    
    return errors, mse, max_error, min_error, std_dev
        
def print_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape: "
                         f"{tensor1.shape} vs {tensor2.shape}")
    print(g_str("Tensor 1: ") +
          y_str("Mean: ") + f"{torch.mean(tensor1):.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(tensor1.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{torch.max(tensor1):.4e}" + ", " +
          y_str("Min: ") + f"{torch.min(tensor1):.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{torch.std(tensor1):.4e}")
    print(g_str("Tensor 2: ") +
          y_str("Mean: ") + f"{torch.mean(tensor2):.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(tensor2.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{torch.max(tensor2):.4e}" + ", " +
          y_str("Min: ") + f"{torch.min(tensor2):.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{torch.std(tensor2):.4e}")
    errors, mse, max_error, min_error, std_dev = get_errors(tensor1, tensor2)
    print(r_str("Errors:   ") + 
          y_str("MSE:  ") + f"{mse:.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(errors.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{max_error:.4e}" + ", " +
          y_str("Min: ") + f"{min_error:.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{std_dev:.4e}")
    
def f64_matmul(mat_a, mat_b):
    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    return (mat_a.to(torch.float64) @ mat_b.to(torch.float64)).to(torch.float64)

def sbvr_randn_test(mat_len=512, sbvr_max_sums=6, device=torch.device("cpu")):
    mat_size = (mat_len, mat_len)

    mat_a = load_or_create_tensor("matrix_a", mat_size, device)
    mat_b = load_or_create_tensor("matrix_b", mat_size, device)

    mat_c_16 = f64_matmul(mat_a.to(torch.float16), mat_b.T.to(torch.float16))
    mat_c_bf16 = f64_matmul(mat_a.to(torch.bfloat16), 
                            mat_b.T.to(torch.bfloat16))
    mat_c_e4m3fn = f64_matmul(mat_a.to(torch.float8_e4m3fn), 
                              mat_b.T.to(torch.float8_e4m3fn))
    mat_c_e5m2 = f64_matmul(mat_a.to(torch.float8_e5m2), 
                            mat_b.T.to(torch.float8_e5m2))
    time_dict = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        time_start = time.time()
        mat_a_sbvr = load_or_create_sbvr("matrix_a", mat_size, device, i,
                                        verbose_level=2)
        mat_b_sbvr = load_or_create_sbvr("matrix_b", mat_size, device, i,
                                        verbose_level=2)
        # sbvr_matmul = sbvr.mm_T(mat_a_sbvr, mat_b_sbvr)
        sbvr_matmul = mat_a_sbvr.decode() @ mat_b_sbvr.decode().T 
        sbvr_dict[i] = sbvr_matmul
        time_dict[i] = time.time() - time_start

    print(y_str("Matrix Size: ") + str(mat_size))
    print(b_str("Case 1: Conversion to " + "bfloat16")) 
    print_errors(mat_c_16, mat_c_bf16)
    print(b_str("Case 2: Conversion to " + "float8_e4m3fn"))
    print_errors(mat_c_16, mat_c_e4m3fn)
    print(b_str("Case 3: Conversion to " + "float8_e5m2"))
    print_errors(mat_c_16, mat_c_e5m2)
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+4}: Conversion to " + f"SBVR {key} bits"))
        print_errors(mat_c_16, value)
        print(y_str("\tTime taken: ") + f"{time_dict[key]:.4f} seconds")
    
def sbvr_store_and_load_test(mat_len=512, num_sums=6, 
                             device=torch.device("cpu")):
    mat_size = (mat_len, mat_len)

    mat_a = load_or_create_tensor("matrix_a", mat_size, device)

    shape_str = "_".join(map(str, mat_a.shape))
    sbvr_matrix = sbvr.sbvr(mat_a, 
                            encoder_config={"num_sums": num_sums},
                            device=device)
    file_path = f"{out_dir}/sbvr_{num_sums}_matrix_a_[{shape_str}].pt"
    sbvr_matrix.save(file_path)
    load_sbvr_matrix = sbvr.load(file_path, device=device)
    target_matrix_decoded = load_sbvr_matrix.decode()
    
    print_errors(mat_a, target_matrix_decoded)
           
def sbvr_mat_mat_mult_test(mat_len=512, num_sums=6, 
                           device=torch.device("cpu"), do_print = False):
    # mat_a = torch.tensor([[1, 3.14, 0]], dtype=torch.float16, device=device)
    # mat_b = torch.tensor([[1, 0, 1],
    #                       [0, 1, 0],
    #                       [1, 0, 1],
    #                       [0, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 0, 0],], 
    #                     dtype=torch.float16, device=device)
    mat_a = load_or_create_tensor("matrix_a", (mat_len, mat_len), device)
    mat_b = load_or_create_tensor("matrix_b", (mat_len, mat_len), device)
    bias = torch.zeros((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    mat_mat_ab = mat_a @ mat_b.T + bias
    if do_print:
        print_tensor(mat_a, "mat_a")
        print_tensor(mat_b.T, "mat_b_T")
        print_tensor(mat_mat_ab, "mat_mat_ab")
        
    mat_a_sbvr = load_or_create_sbvr("matrix_a", (mat_len, mat_len), device,
                                    num_sums, verbose_level=0)
    mat_b_sbvr = load_or_create_sbvr("matrix_b", (mat_len, mat_len), device,
                                    num_sums, verbose_level=0)
    sbvr_decoded_mat_mat_ab = mat_a_sbvr.decode() @ mat_b_sbvr.decode().T + bias
    
    if do_print:
        print_tensor(mat_a_sbvr.bvr, "mat_a_sbvr.bvr")
        print_tensor(mat_a_sbvr.coeff_idx, "mat_a_sbvr.coeff_idx")
        print_tensor(mat_a_sbvr.coeff_cache, "mat_a_sbvr.coeff_cache")
        print_tensor(mat_a_sbvr.decode(), "mat_a_sbvr")
        print_tensor(mat_b_sbvr.bvr, "mat_b_sbvr.bvr")
        print_tensor(mat_b_sbvr.coeff_idx, "mat_b_sbvr.coeff_idx")
        print_tensor(mat_b_sbvr.coeff_cache, "mat_b_sbvr.coeff_cache")
        print_tensor(mat_b_sbvr.decode(), "mat_b_sbvr")
        print_tensor(mat_b_sbvr.decode().T, "mat_b_sbvr_T")
        print_tensor(sbvr_decoded_mat_mat_ab, "sbvr_decoded_mat_mat_ab")
    
    torch.cuda.nvtx.range_push("SBVR CUDA MatMul")
    sbvr_cuda_mat_mat_ab = sbvr.mm_T(mat_a_sbvr, mat_b_sbvr, bias)
    torch.cuda.nvtx.range_pop()
    if do_print:
        print_tensor(sbvr_cuda_mat_mat_ab, "sbvr_cuda_mat_mat_ab")
        
    print(b_str("Case 1: SBVR decoded Matmul vs SBVR CUDA Matmul"))
    print_errors(sbvr_decoded_mat_mat_ab, sbvr_cuda_mat_mat_ab)
    print(b_str("Case 2: Full precision Matmul vs SBVR decoded Matmul"))
    print_errors(mat_mat_ab, sbvr_decoded_mat_mat_ab)
    print(b_str("Case 3: Full precision Matmul vs SBVR CUDA Matmul"))
    print_errors(mat_mat_ab, sbvr_cuda_mat_mat_ab)
    
def sbvr_matmul_time_test(mat_len=512, sbvr_max_sums=6, 
                          device=torch.device("cpu"), num_runs=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_a_size = (1, mat_len)
    mat_b_size = (mat_len, mat_len)
    mat_a = load_or_create_tensor("matrix_a", mat_a_size, device)
    mat_b = load_or_create_tensor("matrix_b", mat_b_size, device)
    bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3

    for _ in range(5):
        f16_matmul = mat_a @ mat_b.T + bias
    time_start = time.time()
    for _ in range(num_runs):
        f16_matmul = mat_a @ mat_b.T + bias
    f16_time = (time.time() - time_start) / num_runs

    sbvr_time = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        mat_a_sbvr = load_or_create_sbvr("matrix_a", mat_a_size, device, i,
                                        verbose_level=1)
        mat_b_sbvr = load_or_create_sbvr("matrix_b", mat_b_size, device, i,
                                        verbose_level=1)
        for _ in range(5):
            f16_matmul = mat_a @ mat_b.T + bias
        time_start = time.time()
        for _ in range(num_runs):
            sbvr_matmul = sbvr.mm_T(mat_a_sbvr, mat_b_sbvr, bias)
        sbvr_time[i] = (time.time() - time_start) / num_runs
        sbvr_dict[i] = sbvr_matmul

    print(y_str("Matrix A Size: ") + str(mat_a_size) + ", " +
          y_str("Matrix B Size: ") + str(mat_b_size))
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+1}: f16 matmul vs SBVR {key} bits"))
        print_errors(f16_matmul, value)
        print(y_str("\tMatmul time taken: ") + f"{sbvr_time[key]:.8f} secs "
                     + y_str("vs ") + f"{f16_time:.8f} secs")
        print(y_str("\tSpeedup: ") + f"{f16_time/sbvr_time[key]:.4f}x")

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    mat_len = sys.argv[1]
    sbvr_max_sums = sys.argv[2]
    
    # sbvr_randn_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_store_and_load_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_mat_mat_mult_test(int(mat_len), int(sbvr_max_sums), device=device)
    sbvr_matmul_time_test(int(mat_len), int(sbvr_max_sums), device=device)
    # os.system(f"rm -rf {out_dir}")
