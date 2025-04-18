import sys
import time
import torch
import sbvr

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
        mat_a_sbvr = sbvr.sbvr(mat_a, num_sums=i)
        mat_b_sbvr = sbvr.sbvr(mat_b, num_sums=i)
        sbvr.save_sbvr(mat_a_sbvr, f"mat_a_{mat_len}_sbvr_{i}.pt")
        sbvr.save_sbvr(mat_b_sbvr, f"mat_b_{mat_len}_sbvr_{i}.pt")
        mat_a_sbvr = sbvr.load_sbvr(f"mat_a_{mat_len}_sbvr_{i}.pt")
        mat_b_sbvr = sbvr.load_sbvr(f"mat_b_{mat_len}_sbvr_{i}.pt")
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
    mat_a = torch.tensor([[1, 0, 3.14, 0]], dtype=torch.float64, device=device)
    mat_b = torch.tensor([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 1, 0],
                          [0, 0, 0, 0],], 
                        dtype=torch.float64, device=device)
    mat_a = torch.randn((mat_len, mat_len), dtype=torch.float64, device=device)*0.3
    mat_b = torch.randn((mat_len, mat_len), 
                        dtype=torch.float64, device=device)*0.3
    mat_mat_ab = mat_a @ mat_b.T
    if do_print:
        print_tensor(mat_a, "mat_a")
        print_tensor(mat_b.T, "mat_b_T")
        print_tensor(mat_mat_ab, "mat_mat_ab")
    
    sbvr_mat_a = sbvr.sbvr(mat_a, num_sums=sbvr_max_sums, 
                           verbose_level=0, use_bias=True)
    sbvr_mat_b = sbvr.sbvr(mat_b, num_sums=sbvr_max_sums, 
                           verbose_level=0, use_bias=True)
    sbvr_decoded_mat_mat_ab = sbvr_mat_a.decode() @ sbvr_mat_b.decode().T
    
    if do_print:
        print_tensor(sbvr_mat_a.decode(), "sbvr_mat_a")
        print_tensor(sbvr_mat_b.decode().T, "sbvr_mat_b_T")
        print_tensor(sbvr_decoded_mat_mat_ab, "sbvr_decoded_mat_mat_ab")
    
    sbvr_cuda_mat_mat_ab = sbvr_mat_a.cuda_mat_mat_t_mul(sbvr_mat_b)
    if do_print:
        print_tensor(sbvr_cuda_mat_mat_ab, "sbvr_cuda_mat_mat_ab")
        
    print(b_str("Case 1: SBVR decoded vs SBVR CUDA"))
    print_errors(sbvr_decoded_mat_mat_ab, sbvr_cuda_mat_mat_ab)
    print(b_str("Case 2: SBVR decoded vs Full precision"))
    print_errors(sbvr_decoded_mat_mat_ab, mat_mat_ab)
    print(b_str("Case 3: Full precision vs SBVR CUDA"))
    print_errors(mat_mat_ab, sbvr_cuda_mat_mat_ab)
    
def sbvr_store_and_load_test(mat_len=512, sbvr_max_sums=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_size = (mat_len, mat_len)

    target_matrix = torch.randn(mat_size, dtype=torch.float64, device=device)*0.3
    sbvr_matrix = sbvr.sbvr(target_matrix, num_sums=sbvr_max_sums)
    sbvr.save_sbvr(sbvr_matrix, f"sbvr_matrix_{mat_len}.pt")
    load_sbvr_matrix = sbvr.load_sbvr(f"sbvr_matrix_{mat_len}.pt", device=device)
    target_matrix_decoded = load_sbvr_matrix.decode()
    
    print_errors(target_matrix, target_matrix_decoded)
    

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    mat_len = sys.argv[1]
    sbvr_max_sums = sys.argv[2]
    
    time_start = time.time()
    # sbvr_randn_test(int(mat_len), int(sbvr_max_sums))
    sbvr_store_and_load_test(int(mat_len), int(sbvr_max_sums))
    print (f"Total time taken: {time.time() - time_start:.4f} seconds")
    
    # sbvr_mat_mat_mult_test(int(mat_len), int(sbvr_max_sums))
