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

def load_or_create_sbvr(name, shape, device, num_sums, verbose_level=0, trans=False):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return sbvr.load(file_path, device=device, verbose_level=verbose_level)
    else:
        tensor = load_or_create_tensor(name, shape, device)
        sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                                device=device, verbose_level=verbose_level,
                                trans=trans)
        sbvr_tensor.save(file_path)
        return sbvr_tensor
    
def create_sbvr(tensor, name, shape, device, num_sums, verbose_level=0):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_{name}_[{shape_str}].pt"
    sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                            device=device, verbose_level=verbose_level)
    sbvr_tensor.save(file_path)
    return sbvr_tensor

def float_to_fp4_e3m0(x):
    x_clamped = torch.clamp(x, -16.0, 16.0)  # Representable range
    sign = (x_clamped < 0).to(torch.uint8)

    # Prevent log2(0) by flooring to a small positive number
    x_abs = x_clamped.abs()
    x_abs = torch.clamp(x_abs, min=1e-8)

    # Compute exponent (bias = 3), round to nearest integer
    exp_unbiased = torch.round(torch.log2(x_abs))
    exp_clamped = exp_unbiased.clamp(-3, 4)
    exp_q = (exp_clamped + 3).to(torch.uint8)  # bias = 3 → encoded in 3 bits

    # Encode as 4-bit value: [sign | exponent (3 bits)]
    fp4 = (sign << 3) | exp_q
    return fp4.to(torch.uint8)

def fp4_e3m0_to_float(fp4):
    sign = (fp4 >> 3) & 0b1
    exp_q = fp4 & 0b111  # 3-bit exponent
    exponent = exp_q.to(torch.int32) - 3  # bias = 3
    value = 2.0 ** exponent
    return torch.where(sign.bool(), -value, value)

def float_to_fp4_e2m1(x):
    x_clamped = torch.clamp(x, -6.0, 6.0)  # Only representable range
    sign = (x_clamped < 0).to(torch.uint8)
    x_abs = x_clamped.abs()

    # Prevent log2(0) → set small floor
    x_abs = torch.clamp(x_abs, min=1e-8)

    # Compute exponent (bias = 1)
    exp_unbiased = torch.floor(torch.log2(x_abs))
    exp_clamped = exp_unbiased.clamp(-1, 2)
    exp_q = (exp_clamped + 1).to(torch.uint8)

    # Reconstruct base value (without mantissa)
    base = 2.0 ** exp_clamped

    # Decide mantissa: if closer to base * 1.5 than base, set mantissa = 1
    mantissa = ((x_abs >= base * 1.25)).to(torch.uint8)

    # Combine to 4-bit format: [sign | exponent (2) | mantissa]
    fp4 = (sign << 3) | (exp_q << 1) | mantissa
    return fp4.to(torch.uint8)

def fp4_e2m1_to_float(fp4):
    sign = (fp4 >> 3) & 0b1
    exp_q = (fp4 >> 1) & 0b11
    mantissa = fp4 & 0b1

    exponent = exp_q.to(torch.int32) - 1  # bias = 1
    base = 2.0 ** exponent
    value = base * (1.0 + 0.5 * mantissa)
    return torch.where(sign.bool(), -value, value)

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

    mat_a_bf16 = mat_a.to(torch.bfloat16).to(torch.float32)
    mat_a_e4m3fn = mat_a.to(torch.float8_e4m3fn).to(torch.float32)
    mat_a_e4m3fnuz = mat_a.to(torch.float8_e4m3fnuz).to(torch.float32)
    mat_a_e5m2 = mat_a.to(torch.float8_e5m2).to(torch.float32)
    mat_a_e5m2fnuz = mat_a.to(torch.float8_e5m2fnuz).to(torch.float32)
    mat_a_e2m1 = fp4_e2m1_to_float(float_to_fp4_e2m1(mat_a))
    mat_a_e3m0 = fp4_e3m0_to_float(float_to_fp4_e3m0(mat_a))
    time_dict = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        time_start = time.time()
        mat_a_sbvr = load_or_create_sbvr("matrix_a", mat_a.shape, device, i,
                                        verbose_level=2)
        sbvr_dict[i] = mat_a_sbvr.decode()
        time_dict[i] = time.time() - time_start

    print(y_str("Matrix Size: ") + str(mat_size))
    print(b_str("Case 1: Conversion to " + "bfloat16")) 
    print_errors(mat_a, mat_a_bf16)
    print(b_str("Case 2: Conversion to " + "float8_e4m3fn"))
    print_errors(mat_a, mat_a_e4m3fn)
    print(b_str("Case 3: Conversion to " + "float8_e4m3fnuz"))
    print_errors(mat_a, mat_a_e4m3fnuz)
    print(b_str("Case 4: Conversion to " + "float8_e5m2"))
    print_errors(mat_a, mat_a_e5m2)
    print(b_str("Case 5: Conversion to " + "float8_e5m2fnuz"))
    print_errors(mat_a, mat_a_e5m2fnuz)
    print(b_str("Case 6: Conversion to " + "float4_e2m1"))
    print_errors(mat_a, mat_a_e2m1)
    print(b_str("Case 7: Conversion to " + "float4_e3m0"))
    print_errors(mat_a, mat_a_e3m0)
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+8}: Conversion to " + f"SBVR {key}"))
        print_errors(mat_a, value)
        print(y_str("\tTime taken: ") + f"{time_dict[key]:.4f} seconds")

def sbvr_randn_mult_test(mat_len=512, sbvr_max_sums=6, 
                         device=torch.device("cpu")):
    mat_size = (mat_len, mat_len)

    mat_a = load_or_create_tensor("matrix_a", mat_size, device)
    mat_b = load_or_create_tensor("matrix_b", mat_size, device)

    mat_c_16 = f64_matmul(mat_a.to(torch.float16), mat_b.T.to(torch.float16))
    mat_c_bf16 = f64_matmul(mat_a.to(torch.bfloat16), 
                            mat_b.T.to(torch.bfloat16))
    mat_c_e4m3fn = f64_matmul(mat_a.to(torch.float8_e4m3fn), 
                              mat_b.T.to(torch.float8_e4m3fn))
    mat_c_e4m3fnuz = f64_matmul(mat_a.to(torch.float8_e4m3fnuz), 
                              mat_b.T.to(torch.float8_e4m3fnuz))
    mat_c_e5m2 = f64_matmul(mat_a.to(torch.float8_e5m2), 
                            mat_b.T.to(torch.float8_e5m2))
    mat_c_e5m2fnuz = f64_matmul(mat_a.to(torch.float8_e5m2fnuz),
                            mat_b.T.to(torch.float8_e5m2fnuz))
    mat_c_e2m1 = f64_matmul(fp4_e2m1_to_float(float_to_fp4_e2m1(mat_a)),
                            fp4_e2m1_to_float(float_to_fp4_e2m1(mat_b.T)))
    mat_c_e3m0 = f64_matmul(fp4_e3m0_to_float(float_to_fp4_e3m0(mat_a)),
                            fp4_e3m0_to_float(float_to_fp4_e3m0(mat_b.T)))
    time_dict = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        time_start = time.time()
        mat_a_sbvr = load_or_create_sbvr("matrix_a", mat_a.shape, device, i,
                                        verbose_level=2)
        mat_b_sbvr = load_or_create_sbvr("matrix_b", mat_b.shape, device, i,
                                        verbose_level=2)
        sbvr_matmul = sbvr.mm_T(mat_a_sbvr, mat_b_sbvr, None)
        # sbvr_matmul = mat_a_sbvr.decode() @ mat_b_sbvr.decode().T 
        sbvr_dict[i] = sbvr_matmul
        time_dict[i] = time.time() - time_start

    print(y_str("Matrix Size: ") + str(mat_size))
    print(b_str("Case 1: MM_T after Conversion to " + "bfloat16")) 
    print_errors(mat_c_16, mat_c_bf16)
    print(b_str("Case 2: MM_T after Conversion to " + "float8_e4m3fn"))
    print_errors(mat_c_16, mat_c_e4m3fn)
    print(b_str("Case 3: MM_T after Conversion to " + "float8_e4m3fnuz"))
    print_errors(mat_c_16, mat_c_e4m3fnuz)
    print(b_str("Case 4: MM_T after Conversion to " + "float8_e5m2"))
    print_errors(mat_c_16, mat_c_e5m2)
    print(b_str("Case 5: MM_T after Conversion to " + "float8_e5m2fnuz"))
    print_errors(mat_c_16, mat_c_e5m2fnuz)
    print(b_str("Case 6: MM_T after Conversion to " + "float4_e2m1"))
    print_errors(mat_c_16, mat_c_e2m1)
    print(b_str("Case 7: MM_T after Conversion to " + "float4_e3m0"))
    print_errors(mat_c_16, mat_c_e3m0)
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+8}: MM_T after Conversion to " + f"SBVR {key}"))
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
    mat_a = load_or_create_tensor("matrix_a", (1, mat_len), device)
    mat_b = load_or_create_tensor("matrix_b", (mat_len, mat_len), device)
    bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    mat_mat_ab = mat_a @ mat_b.T + bias
    if do_print:
        print_tensor(mat_a, "mat_a")
        print_tensor(mat_b.T, "mat_b_T")
        print_tensor(mat_mat_ab, "mat_mat_ab")
        
    # mat_a_sbvr = create_sbvr(mat_a, "matrix_a", (mat_a.size(0), mat_a.size(1)), 
    #                          device, num_sums, verbose_level=0)
    # mat_b_sbvr = create_sbvr(mat_b, "matrix_b", (mat_b.size(0), mat_b.size(1)), 
    #                          device, num_sums, verbose_level=0)
    mat_a_sbvr = load_or_create_sbvr("matrix_a", mat_a.shape, device,
                                    num_sums, verbose_level=1)
    mat_b_sbvr = load_or_create_sbvr("matrix_b", mat_b.shape, device,
                                    num_sums, verbose_level=1)
    lhs_bvr = mat_a_sbvr.bvr
    lhs_coeff_idx = mat_a_sbvr.coeff_idx
    lhs_coeff_cache = mat_a_sbvr.coeff_cache
    rhs_bvr = mat_b_sbvr.bvr
    rhs_coeff_idx = mat_b_sbvr.coeff_idx
    rhs_coeff_cache = mat_b_sbvr.coeff_cache
    
    sbvr_decoded_mat_mat_ab = mat_a_sbvr.decode() @ mat_b_sbvr.decode().T + bias
    sbvr_cuda_mat_mat_ab = sbvr._sbvr_mm_T(
                                lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                                rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                bias)
    
    torch.cuda.profiler.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("PyTorch CUDA MatMul")
    sbvr_decoded_mat_mat_ab = mat_a_sbvr.decode() @ mat_b_sbvr.decode().T + bias
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("SBVR CUDA MatMul")
    sbvr_cuda_mat_mat_ab = sbvr._sbvr_mm_T(
                                lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                                rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                bias)
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.cudart().cudaProfilerStop()
    
    if do_print:
        print_tensor(mat_a_sbvr.bvr, "mat_a_sbvr.bvr")
        print_tensor(mat_a_sbvr.coeff_idx, "mat_a_sbvr.coeff_idx")
        print_tensor(mat_a_sbvr.coeff_cache, "mat_a_sbvr.coeff_cache")
        print_tensor(mat_a_sbvr.decode(), "mat_a_sbvr")
        print_tensor(mat_b_sbvr.bvr, "mat_b_sbvr.bvr")
        print_tensor(mat_b_sbvr.coeff_idx, "mat_b_sbvr.coeff_idx")
        print_tensor(mat_b_sbvr.coeff_cache, "mat_b_sbvr.coeff_cache")
        print_tensor(mat_b_sbvr.decode(), "mat_b_sbvr")
        print_tensor(sbvr_decoded_mat_mat_ab, "sbvr_decoded_mat_mat_ab")
        print_tensor(sbvr_cuda_mat_mat_ab, "sbvr_cuda_mat_mat_ab")
        
    print(b_str("Case 1: SBVR decoded Matmul vs SBVR CUDA Matmul"))
    print_errors(sbvr_decoded_mat_mat_ab, sbvr_cuda_mat_mat_ab)
    print(b_str("Case 2: Full precision Matmul vs SBVR decoded Matmul"))
    print_errors(mat_mat_ab, sbvr_decoded_mat_mat_ab)
    print(b_str("Case 3: Full precision Matmul vs SBVR CUDA Matmul"))
    print_errors(mat_mat_ab, sbvr_cuda_mat_mat_ab)
    
def sbvr_matmul_time_test(mat_len=512, sbvr_max_sums=6, 
                          device=torch.device("cpu"), num_runs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_a_size = (1, mat_len)
    mat_b_size = (mat_len, mat_len)
    mat_a = load_or_create_tensor("matrix_a", mat_a_size, device)
    mat_b = load_or_create_tensor("matrix_b", mat_b_size, device)
    bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    
    for i in range(10):
        f16_matmul = mat_a @ mat_b.T + bias
    torch.cuda.synchronize()
    time_start = time.perf_counter()
    for i in range(num_runs):
        f16_matmul = mat_a @ mat_b.T + bias
    torch.cuda.synchronize()
    f16_time = (time.perf_counter() - time_start) / num_runs

    sbvr_time = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        mat_a_sbvr = load_or_create_sbvr("matrix_a", mat_a.shape, device, i,
                                        verbose_level=1)
        mat_b_sbvr = load_or_create_sbvr("matrix_b", mat_b.shape, device, i,
                                        verbose_level=1)
        lhs_bvr = mat_a_sbvr.bvr
        lhs_coeff_idx = mat_a_sbvr.coeff_idx
        lhs_coeff_cache = mat_a_sbvr.coeff_cache
        rhs_bvr = mat_b_sbvr.bvr
        rhs_coeff_idx = mat_b_sbvr.coeff_idx
        rhs_coeff_cache = mat_b_sbvr.coeff_cache
        
        for _ in range(10):
            sbvr_matmul = sbvr._sbvr_mm_T(
                                    lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                                    rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                    bias)
        torch.cuda.synchronize()
        time_start = time.perf_counter()
        for _ in range(num_runs):
            sbvr_matmul = sbvr._sbvr_mm_T(
                                    lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                                    rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                    bias)
        torch.cuda.synchronize()
        sbvr_time[i] = (time.perf_counter() - time_start) / num_runs
        sbvr_dict[i] = sbvr_matmul

    print(y_str("Matrix A Size: ") + str(mat_a_size) + ", " +
          y_str("Matrix B Size: ") + str(mat_b_size))
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+1}: f16 matmul vs SBVR {key} bits"))
        if value is not None:
            print_errors(f16_matmul, value)
        print(y_str("\tMatmul time taken: ") 
              + f"{sbvr_time[key]*10e6:.4f} usecs"
              + y_str(" vs ") + f"{f16_time*10e6:.4f} usecs")
        print(y_str("\tSpeedup: ") + f"{f16_time/sbvr_time[key]:.4f}x")
        
def sbvr_rd_matmul_time_test(mat_len=512, sbvr_max_sums=6,
                             device=torch.device("cpu"), num_runs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_a_size = (1, mat_len)
    mat_b_size = (mat_len, mat_len)
    mat_a = load_or_create_tensor("matrix_a", mat_a_size, device)
    mat_b = load_or_create_tensor("matrix_b", mat_b_size, device)
    bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    
    for i in range(10):
        f16_matmul = mat_a @ mat_b.T + bias
    torch.cuda.synchronize()
    time_start = time.perf_counter()
    for i in range(num_runs):
        f16_matmul = mat_a @ mat_b.T + bias
    torch.cuda.synchronize()
    f16_time = (time.perf_counter() - time_start) / num_runs

    sbvr_time = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        mat_b_sbvr_trans = load_or_create_sbvr("matrix_b", mat_b.shape, device, i,
                                        verbose_level=1, trans=True)
        rhs_bvr = mat_b_sbvr_trans.bvr
        rhs_coeff_idx = mat_b_sbvr_trans.coeff_idx
        rhs_coeff_cache = mat_b_sbvr_trans.coeff_cache
        
        for _ in range(10):
            sbvr_matmul = sbvr.sbvr_cuda._sbvr_row_deq_mm_T(
                                    mat_a,
                                    rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                    bias)
        torch.cuda.synchronize()
        time_start = time.perf_counter()
        for _ in range(num_runs):
            sbvr_matmul = sbvr.sbvr_cuda._sbvr_row_deq_mm_T(
                                    mat_a,
                                    rhs_bvr, rhs_coeff_idx, rhs_coeff_cache,
                                    bias)
        torch.cuda.synchronize()
        sbvr_time[i] = (time.perf_counter() - time_start) / num_runs
        sbvr_dict[i] = sbvr_matmul
        
    print(y_str("Matrix A Size: ") + str(mat_a_size) + ", " +
        y_str("Matrix B Size: ") + str(mat_b_size))
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+1}: f16 matmul vs SBVR {key} bits"))
        if value is not None:
            print_errors(f16_matmul, value)
        print(y_str("\tMatmul time taken: ") 
            + f"{sbvr_time[key]*10e6:.4f} usecs"
            + y_str(" vs ") + f"{f16_time*10e6:.4f} usecs")
        print(y_str("\tSpeedup: ") + f"{f16_time/sbvr_time[key]:.4f}x")
        
def sbvr_online_test(mat_len=512, sbvr_max_sums=6, device=torch.device("cpu")):
    mat_size = (mat_len, mat_len)

    mat_a = load_or_create_tensor("matrix_a", mat_size, device)
    mat_b = load_or_create_tensor("matrix_b", mat_size, device)
    bias = torch.randn((mat_b.size(0),), dtype=torch.float16, device=device)*0.3
    
    f16_matmul = mat_a @ mat_b.T
    time_dict = {}
    sbvr_dict = {}
    for i in range (sbvr_max_sums, 1, -2):
        time_start = time.time()
        mat_b_sbvr = load_or_create_sbvr("matrix_b", mat_b.shape, device, i,
                                        verbose_level=2)
        mat_b_sbvr.profile_input(mat_a, encoder_config={"num_sums": i})
        sbvr_matmul = mat_b_sbvr.online_mm_T(mat_a, bias) # mat_a is RHS
        sbvr_dict[i] = sbvr_matmul
        time_dict[i] = time.time() - time_start

    print(y_str("Matrix Size: ") + str(mat_size))
    for i, (key, value) in enumerate(sbvr_dict.items()):
        print(b_str(f"Case {i+1}: Conversion to " + f"SBVR {key} bits"))
        print_errors(f16_matmul, value)
        print(y_str("\tTime taken: ") + f"{time_dict[key]:.4f} seconds")

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    mat_len = sys.argv[1]
    sbvr_max_sums = sys.argv[2]
    
    # sbvr_randn_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_randn_mult_test(int(mat_len), int(sbvr_max_sums), device=device)
    sbvr_rd_matmul_time_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_store_and_load_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_mat_mat_mult_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_matmul_time_test(int(mat_len), int(sbvr_max_sums), device=device)
    # sbvr_online_test(int(mat_len), int(sbvr_max_sums), device=device)
    # os.system(f"rm -rf {out_dir}")
