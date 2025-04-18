import sys
import time
import torch
import sbvr
import copy
import os

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
    
def get_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    errors = tensor1 - tensor2
    mse = torch.mean(errors ** 2).item()
    max_error = torch.max(errors).item()
    min_error = torch.min(errors).item()
    std_dev = torch.std(errors).item()
    
    return mse, max_error, min_error, std_dev
        
def print_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape: "
                         f"{tensor1.shape} vs {tensor2.shape}")
    
    mse, max_error, min_error, std_dev = get_errors(tensor1, tensor2)
    print(r_str("Errors: ") + 
          y_str("Mean: ") + f"{mse:.4e}" + ", " +
          y_str("Max: ") + f"{max_error:.4e}" + ", " +
          y_str("Min: ") + f"{min_error:.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{std_dev:.4e}")
    
def f64_matmul(mat_a, mat_b):
    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    return (mat_a.to(torch.float64) @ mat_b.to(torch.float64)).to(torch.float64)

def sbvr_randn_test(mat_len=512, sbvr_max_sums=6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_size = (mat_len, mat_len)

    mat_a = torch.randn(mat_size, dtype=torch.float64, device=device)*0.3
    mat_b = torch.randn(mat_size, dtype=torch.float64, device=device)*0.3
    # print_tensor(mat_b, "mat_b")

    mat_c_64 = f64_matmul(mat_a, mat_b)
    mat_c_32 = f64_matmul(mat_a.to(torch.float32), mat_b.to(torch.float32))
    mat_c_16 = f64_matmul(mat_a.to(torch.float16), mat_b.to(torch.float16))
    mat_c_bf16 = f64_matmul(mat_a.to(torch.bfloat16), mat_b.to(torch.bfloat16))
    mat_c_e4m3fn = f64_matmul(mat_a.to(torch.float8_e4m3fn), 
                              mat_b.to(torch.float8_e4m3fn))
    mat_c_e5m2 = f64_matmul(mat_a.to(torch.float8_e5m2), 
                            mat_b.to(torch.float8_e5m2))
    time_dict = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        time_start = time.time()
        mat_a_sbvr = sbvr.sbvr(mat_a, verbose_level=1, num_sums=i)
        mat_b_sbvr = sbvr.sbvr(mat_b, verbose_level=1, num_sums=i)
        sbvr_matmul = f64_matmul(mat_a_sbvr.decode(), 
                                 mat_b_sbvr.decode())
        sbvr_dict[i] = sbvr_matmul
        time_dict[i] = time.time() - time_start

    print(y_str("Matix Size: ") + str(mat_size))
    print(b_str("Case 1: Conversion to float32"))
    print_errors(mat_c_64, mat_c_32)
    print(b_str("Case 2: Conversion to float16"))
    print_errors(mat_c_64, mat_c_16)
    print(b_str("Case 3: Conversion to bfloat16")) 
    print_errors(mat_c_64, mat_c_bf16)
    print(b_str("Case 4: Conversion to float8_e4m3fn"))
    print_errors(mat_c_64, mat_c_e4m3fn)
    print(b_str("Case 5: Conversion to float8_e5m2"))
    print_errors(mat_c_64, mat_c_e5m2)
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+6}: Conversion to SBVR {key} bits"))
        print_errors(mat_c_64, value)
        print(y_str("\tTime taken: ") + f"{time_dict[key]:.4f} seconds")
        
def sbvr_mat_mat_mult_test(mat_len=512, sbvr_max_sums=6, do_print = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_a = torch.tensor([[1, 3.14, 0]], dtype=torch.float16, device=device)
    mat_b = torch.tensor([[1, 0, 1],
                          [0, 1, 0],
                          [1, 0, 1],], 
                        dtype=torch.float16, device=device)
    mat_a = torch.randn((mat_len, mat_len), 
                        dtype=torch.float64, device=device)*0.3
    mat_b = torch.randn((mat_len, mat_len), 
                        dtype=torch.float64, device=device)*0.3
    mat_mat_ab = mat_a @ mat_b.T
    if do_print:
        print_tensor(mat_a, "mat_a")
        print_tensor(mat_b.T, "mat_b_T")
        print_tensor(mat_mat_ab, "mat_mat_ab")
    
    sbvr_mat_a = sbvr.sbvr(mat_a, num_sums=sbvr_max_sums, 
                           verbose_level=1 if do_print else 0)
    sbvr_mat_b = sbvr.sbvr(mat_b, num_sums=sbvr_max_sums, 
                           verbose_level=1 if do_print else 0)
    sbvr_decoded_mat_mat_ab = sbvr_mat_a.decode() @ sbvr_mat_b.decode().T
    
    if do_print:
        print_tensor(sbvr_mat_a.bvr, "sbvr_mat_a.bvr")
        print_tensor(sbvr_mat_a.coeff_idx, "sbvr_mat_a.coeff_idx")
        print_tensor(sbvr_mat_a.coeff_cache, "sbvr_mat_a.coeff_cache")
        print_tensor(sbvr_mat_a.decode(), "sbvr_mat_a")
        print_tensor(sbvr_mat_b.bvr, "sbvr_mat_b.bvr")
        print_tensor(sbvr_mat_b.coeff_idx, "sbvr_mat_b.coeff_idx")
        print_tensor(sbvr_mat_b.coeff_cache, "sbvr_mat_b.coeff_cache")
        print_tensor(sbvr_mat_b.decode(), "sbvr_mat_b")
        print_tensor(sbvr_mat_b.decode().T, "sbvr_mat_b_T")
        print_tensor(sbvr_decoded_mat_mat_ab, "sbvr_decoded_mat_mat_ab")
    
    sbvr_cuda_mat_mat_ab = sbvr.mm_T(sbvr_mat_a, sbvr_mat_b)
    if do_print:
        print_tensor(sbvr_cuda_mat_mat_ab, "sbvr_cuda_mat_mat_ab")
        
    print(b_str("Case 1: SBVR decoded Matmul vs SBVR CUDA Matmul"))
    print_errors(sbvr_decoded_mat_mat_ab, sbvr_cuda_mat_mat_ab)
    print(b_str("Case 2: Full precision Matmul vs SBVR decoded Matmul"))
    print_errors(mat_mat_ab, sbvr_decoded_mat_mat_ab)
    print(b_str("Case 3: Full precision Matmul vs SBVR CUDA Matmul"))
    print_errors(mat_mat_ab, sbvr_cuda_mat_mat_ab)
    
def sbvr_store_and_load_test(mat_len=512, sbvr_max_sums=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_size = (mat_len, mat_len)

    target_matrix = torch.randn(mat_size, dtype=torch.float64, device=device)
    sbvr_matrix = sbvr.sbvr(target_matrix, num_sums=sbvr_max_sums)
    sbvr.save_sbvr(sbvr_matrix, f"sbvr_matrix_{mat_len}.pt")
    load_sbvr_matrix = sbvr.load_sbvr(f"sbvr_matrix_{mat_len}.pt", device=device)
    target_matrix_decoded = load_sbvr_matrix.decode()
    
    print_errors(target_matrix, target_matrix_decoded)
    
def sbvr_matmul_time_test(mat_len=512, sbvr_max_sums=6, num_runs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_size = (mat_len, mat_len)

    mat_a = torch.randn(mat_size, dtype=torch.float16, device=device)*0.3

    mat_a_cpy = copy.deepcopy(mat_a)
    time_start = time.time()
    for _ in range(num_runs):
        f16_matmul = mat_a @ mat_a_cpy.T
    f16_time = (time.time() - time_start) / num_runs

    sbvr_time = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        mat_a_sbvr_path = f"sbvr_{i}_matrix_a_{mat_len}.pt"
        if not os.path.exists(mat_a_sbvr_path):
            mat_a_sbvr = sbvr.sbvr(mat_a, num_sums=i)
            sbvr.save_sbvr(mat_a_sbvr, mat_a_sbvr_path)
        else:
            mat_a_sbvr = sbvr.load_sbvr(mat_a_sbvr_path, device=device)
        mat_a_sbvr_cpy = copy.deepcopy(mat_a_sbvr)
        time_start = time.time()
        for _ in range(num_runs):
            sbvr_matmul = sbvr.mm_T(mat_a_sbvr, mat_a_sbvr_cpy)
        sbvr_time[i] = (time.time() - time_start) / num_runs
        sbvr_dict[i] = sbvr_matmul

    print(y_str("Matix Size: ") + str(mat_size))
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+1}: SBVR {key} bits"))
        print_errors(f16_matmul, value)
        print(y_str("\tMatmul time taken: ") + f"{sbvr_time[key]:.4f} secs "
                     + y_str("vs ") + f"{f16_time:.4f} secs")
        print(y_str("\tSpeedup: ") + f"{f16_time/sbvr_time[key]:.4f}x")

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    mat_len = sys.argv[1]
    sbvr_max_sums = sys.argv[2]
    
    # sbvr_randn_test(int(mat_len), int(sbvr_max_sums))
    # sbvr_store_and_load_test(int(mat_len), int(sbvr_max_sums))
    # sbvr_mat_mat_mult_test(int(mat_len), int(sbvr_max_sums))
    sbvr_matmul_time_test(int(mat_len), int(sbvr_max_sums))
