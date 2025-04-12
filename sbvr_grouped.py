import torch
import itertools
import math
import os
import sys
import datetime
import time

from sbvr_utils.utils_llama import get_llama, get_layer_ffn_weight
from sbvr_utils.log_config import get_logger, ExtLogger
logger = get_logger(__name__)

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
        
def print_errors(tensor1, tensor2, log_ext=False, **kwargs):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    mse, max_error, min_error, std_dev = get_errors(tensor1, tensor2)
    print(r_str("Errors: ") + 
          y_str("Mean: ") + f"{mse:.4e}" + ", " +
          y_str("Max: ") + f"{max_error:.4e}" + ", " +
          y_str("Min: ") + f"{min_error:.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{std_dev:.4e}")
    
    if log_ext:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(curr_dir, "logs", f"error.txt")
        parent_dir = os.path.dirname(log_path)
        os.makedirs(parent_dir, exist_ok=True)
        
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                pass
            
        log_text = (
            "Errors: "
            + "Mean: " + f"{mse:.4e}" + ", "
            + "Max: " + f"{max_error:.4e}" + ", "
            + "Min: " + f"{min_error:.4e}" + ", "
            + "Std. Dev.: " + f"{std_dev:.4e}" + "\n\n"
        )
        
        with open(log_path, "a") as log_file:
            log_file.write(f"Timestamp: {curr_datetime}\n")
            if kwargs.get("num_sums", None) is not None:
                log_file.write(f"Num Sums: {kwargs['num_sums']}\n")
            log_file.write(log_text)
    
def f64_matmul(mat_a, mat_b):
    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    return (mat_a.to(torch.float64) @ mat_b.to(torch.float64)).to(torch.float64)

class sbvr(): 
    def __init__(self, 
                 data: torch.Tensor = None, 
                 num_sums: int = 4,
                 coeff_group_size: int = 512,
                 r_search_num = 64,
                 s_search_num = 32,
                 b_search_num = 32,
                 coeff_dtype: torch.dtype = None,
                 bin_vec_dtype: torch.dtype = torch.int32,
                 compute_dtype: torch.dtype = torch.float16):
        if data is None:
            raise ValueError(r_str("Data cannot be None"))
            
        self.num_sums = num_sums
        self.coeff_group_size = coeff_group_size
        self.coeff_dtype = data.dtype if coeff_dtype is None else coeff_dtype
        self.bin_vec_dtype = bin_vec_dtype
        self.compute_dtype = compute_dtype
        
        self.r_search_num = r_search_num
        self.s_search_num = s_search_num
        self.b_search_num = b_search_num
        
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
        
        diff_mat_size = 3 * b_search_num * (2**self.num_sums) * \
            coeff_group_size * torch.tensor(0, dtype=self.compute_dtype, 
                                   device=data.device).element_size()
        total_mem = torch.cuda.get_device_properties(data.device).total_memory
        self.search_batch_size = int(total_mem * 0.90 / diff_mat_size)
            
        self.__encode_to_sbvr(data)
        
    @torch.no_grad()
    def __get_all_points(self, coeff_bias: torch.tensor, coeff: torch.tensor):
        num_coeff = coeff.shape[0]
        bin_combs = torch.tensor(
            list(itertools.product([0, 1], repeat=num_coeff)),
            dtype=coeff.dtype, device=coeff.device
        )
        return bin_combs @ coeff + coeff_bias
    
    @torch.no_grad()
    def __get_search_space(self, data):
        data_max = torch.max(data)
        data_avg = torch.mean(data)
        data_min = torch.min(data)
        data_97 = torch.quantile(data.to(torch.float32), 0.97)

        r_max = math.pi*2/3 + 0.1
        r_min = 1.0
        r_gran = (r_max - r_min) / self.r_search_num
        b_max = data_avg
        b_min = data_min - (abs(data_min)*0.1)
        b_gran = (b_max - b_min) / self.b_search_num
        s_max = (data_max - data_min) * 1.1
        s_min = (data_97 - data_avg) * 2.0
        s_gran = (s_max - s_min) / self.s_search_num
        
        print(r_str("\tnum_sums: ") + f"{self.num_sums}")
        print(r_str("\tR search range: ") + f"{(1.0 + r_gran):.4e} to " + 
              f"{r_max:.4e}, " +
              r_str("search granularity: ") + f"{r_gran:.4e}")
        print(r_str("\tBias search range: ") + f"{data_min:.4e} to " + 
              f"{data_avg:.4e}, " +
              r_str("search granularity: ") + f"{b_gran:.4e}")
        print(r_str("\tScale search range: ") + 
              f"{((data_97 - data_avg*2.0)):.4e} to " +
              f"{((data_max - data_min)*1.1):.4e}, " +
              r_str("search granularity: ") + f"{s_gran:.4e}")
        
        r_list = torch.arange(r_min + r_gran, r_max + r_gran, r_gran, 
                              device=data.device, dtype=data.dtype)
        s_list = torch.arange(s_min + s_gran, s_max + s_gran, s_gran, 
                              device=data.device, dtype=data.dtype)
        b_list = torch.arange(b_min + b_gran, b_max + b_gran, b_gran, 
                              device=data.device, dtype=data.dtype)
        exponents = torch.arange(self.num_sums, device=data.device)   

        search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
        search_space = \
            search_space / torch.sum(search_space, dim=1).unsqueeze(1)
        search_space = s_list.view(-1, 1, 1) * search_space.unsqueeze(0)
        search_space = search_space.view(-1, self.num_sums)

        return search_space, r_list, b_list, s_list
    
    @torch.no_grad()
    def __get_min_mse_coeff(self, biased_data, search_matrix):
        bin_combs = torch.tensor(
            list(itertools.product([0, 1], repeat=self.num_sums)),
            dtype=biased_data.dtype, device=biased_data.device
        ).T
        candidiate_matrix = search_matrix @ bin_combs
        
        data_size = biased_data.shape[1]
        bias_list_size = biased_data.shape[0]
        n_ss_row = candidiate_matrix.shape[0]
        n_ss_col = candidiate_matrix.shape[1]
        
        biased_data = biased_data.view(bias_list_size, 1, data_size, 1)
        candidiate_matrix = candidiate_matrix.view(1, n_ss_row, 1, n_ss_col)
        
        diff = (biased_data - candidiate_matrix)**2

        # (bias_list_size, n_ss_row, data_size)
        diff_selected, coeff_comb_indices = diff.min(dim=-1) 
        mse = diff_selected.mean(dim=-1)
        
        flat_min_idx = mse.view(-1).argmin()
        min_idx = torch.unravel_index(flat_min_idx, mse.shape)
        bias_idx, coeff_set_idx = min_idx
        coeff_comb_idx = coeff_comb_indices[bias_idx, coeff_set_idx]
        min_mse = mse[bias_idx, coeff_set_idx].item()

        return min_mse, bias_idx, coeff_set_idx, coeff_comb_idx
    
    @torch.no_grad()
    def __encode_data(self, data):
        
        search_space, r_list, b_list, s_list = self.__get_search_space(data)
        biased_data = data.unsqueeze(0) - b_list.view(-1, 1)
        len_search_space = search_space.shape[0]
        min_mse = float("inf")

        # Loop over the bias values
        for search_start in range(0, len_search_space, self.search_batch_size):
            torch.cuda.empty_cache()
            search_end = \
                min(search_start + self.search_batch_size, len_search_space)
            coeff_list = search_space[search_start:search_end]

            # Call a method to get the index and MSE among these coefficients
            mse, bias_idx, coeff_set_idx, coeff_comb_idx = \
                self.__get_min_mse_coeff(biased_data, coeff_list)
            
            search_space_idx = search_start + coeff_set_idx
            scale_idx = search_space_idx // len(r_list)
            r_idx = search_space_idx % len(r_list)
            
            if mse < min_mse:
                min_mse = mse
                best_bias = b_list[bias_idx]
                best_coeff = search_space[search_space_idx]
                best_coeff_idx = coeff_comb_idx
                best_r = r_list[r_idx]
                best_s = s_list[scale_idx]
                
        print (g_str("Best MSE: ") + f"{min_mse:.4e}" +
            ", " + g_str("Coeff: ") + str(best_coeff) +
            ", " + g_str("(r, b, s): ") + f"{best_r:.4e}, " +
            f"{best_bias.item():.4e}, {best_s:.4e}")
                
        return best_bias, best_coeff, best_coeff_idx
    
    @torch.no_grad()
    def __encode_to_sbvr(self, data):
        print(b_str("Encoding to SBVR..."))
        data_num = data.numel()
        num_coeff_groups = \
            (data_num + self.coeff_group_size - 1) // self.coeff_group_size
        self.coeff = torch.empty((num_coeff_groups, self.num_sums), 
                                    dtype=self.coeff_dtype, device=data.device)
        self.coeff_bias = torch.empty((num_coeff_groups), 
                                      dtype=self.coeff_dtype, 
                                      device=self.coeff.device)
        self.coeff_idx = torch.empty((data_num), dtype=int, 
                                     device=self.coeff.device)
        
        for i in range(num_coeff_groups):
            # logger.info(f"Encoding group {i + 1}/{num_coeff_groups}")
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, data_num)
            group_data = \
                data.flatten()[group_start:group_end].to(self.compute_dtype)
            coeff_bias, coeff, coeff_idx = self.__encode_data(group_data)
            self.coeff_bias[i] = coeff_bias.item()
            self.coeff[i] = coeff
            self.coeff_idx[group_start:group_end] = coeff_idx
        
    def get_decoded_tensor(self):
        decoded_tensor = torch.empty(self.original_data_shape,
                                      dtype=self.original_dtype,
                                      device=self.coeff.device)
        num_coeff_groups = self.coeff_bias.shape[0]
        for i in range(num_coeff_groups):
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, decoded_tensor.numel())
            group_coeff_bias = self.coeff_bias[i]
            group_coeff = self.coeff[i]
            group_coeff_idx = self.coeff_idx[group_start:group_end]
            group_all_points = \
                self.__get_all_points(group_coeff_bias, group_coeff)
            group_data = group_all_points[group_coeff_idx]
            decoded_tensor.flatten()[group_start:group_end] = group_data
        
        return decoded_tensor
    
    def get_sbvr_info(self):
        info_str = b_str("SBVR Info:") + \
        y_str("\n\tOriginal Data Type: ") + str(self.original_dtype) + \
        y_str("\n\tOriginal Data Shape: ") + str(self.original_data_shape) + \
        y_str("\n\tNumber of Summations: ") + str(self.num_sums) + \
        y_str("\n\tCoefficient Group Size: ") + str(self.coeff_group_size) + \
        y_str("\n\tCoefficient Data Type: ") + str(self.coeff_dtype) + \
        y_str("\n\tBinary Vector Data Type: ") + str(self.bin_vec_dtype) + \
        y_str("\n\tCoefficient Tensor Size: ") + str(self.coeff.shape)
        return info_str


def randn_test():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_size = (64, 64)

    mat_a = torch.randn(mat_size, dtype=torch.float64).to(device)
    mat_b = torch.randn(mat_size, dtype=torch.float64).to(device)
    # print_tensor(mat_a, "mat_a")
    # print_tensor(mat_b, "mat_b")

    mat_c_64 = f64_matmul(mat_a, mat_b)
    mat_c_32 = f64_matmul(mat_a.to(torch.float32), mat_b.to(torch.float32))
    mat_c_16 = f64_matmul(mat_a.to(torch.float16), mat_b.to(torch.float16))
    mat_c_bf16 = f64_matmul(mat_a.to(torch.bfloat16), mat_b.to(torch.bfloat16))
    mat_c_e4m3fn = f64_matmul(mat_a.to(torch.float8_e4m3fn), mat_b.to(torch.float8_e4m3fn))
    mat_c_e5m2 = f64_matmul(mat_a.to(torch.float8_e5m2), mat_b.to(torch.float8_e5m2))
    mat_c_sbvr_8 = f64_matmul(sbvr(mat_a, num_sums=8).get_decoded_tensor(), 
                            sbvr(mat_b, num_sums=8).get_decoded_tensor())
    mat_c_sbvr_6 = f64_matmul(sbvr(mat_a, num_sums=6).get_decoded_tensor(), 
                            sbvr(mat_b, num_sums=6).get_decoded_tensor())
    mat_c_sbvr_4 = f64_matmul(sbvr(mat_a, num_sums=4).get_decoded_tensor(), 
                            sbvr(mat_b, num_sums=4).get_decoded_tensor())
    mat_c_sbvr_2 = f64_matmul(sbvr(mat_a, num_sums=2).get_decoded_tensor(), 
                            sbvr(mat_b, num_sums=2).get_decoded_tensor())

    # print_tensor(mat_c_64, "mat_c")
    # print_tensor(mat_c_16, "mat_c_16")
    # print_tensor(mat_c_bf16, "mat_c_bf16")
    # print_tensor(mat_c_e4m3fn, "mat_c_e4m3fn")
    # print_tensor(mat_c_e5m2, "mat_c_e5m2")
    # print_tensor(mat_c_sbvr_8, "mat_c_sbvr_8")
    # print_tensor(mat_c_sbvr_6, "mat_c_sbvr_6")
    # print_tensor(mat_c_sbvr_4, "mat_c_sbvr_4")
    # print_tensor(mat_c_sbvr_2, "mat_c_sbvr_2")

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
    print(b_str("Case 6: Conversion to sbvr 8 bit"))
    print_errors(mat_c_64, mat_c_sbvr_8)
    print(b_str("Case 7: Conversion to sbvr 6 bit"))
    print_errors(mat_c_64, mat_c_sbvr_6)
    print(b_str("Case 8: Conversion to sbvr 4 bit"))
    print_errors(mat_c_64, mat_c_sbvr_4)
    print(b_str("Case 9: Conversion to sbvr 2 bit"))
    print_errors(mat_c_64, mat_c_sbvr_2)


def test_with_llama3_weight():
    MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
    TARGET_LAYER_IDX = 1
    model, _ = get_llama(MODEL_PATH)
    ffn_weight = get_layer_ffn_weight(model, TARGET_LAYER_IDX)
    logger.info(f"ffn_weight shape: {ffn_weight.shape}")
    logger.info(f"ffn_weight dtype: {ffn_weight.dtype}")
    logger.info(f"ffn_weight device: {ffn_weight.device}")
    
    ext_logger = ExtLogger("llama3_weight_test.txt")
    
    def random_fetcher_test(fetch_unit:int = 16, n_fetches:int = 60, weight:torch.Tensor = None):
        if weight is None:
            raise ValueError("weight cannot be None")
        if fetch_unit > weight.shape[0] or fetch_unit > weight.shape[1]:
            raise ValueError("fetch_unit must be smaller than weight shape")
        
        ext_logger.write(f"Fetch Unit: {fetch_unit}")
        ext_logger.write(f"Number of Fetches: {n_fetches}\n")
        
        fetch_r_indices = torch.randint(0, weight.shape[0] - fetch_unit + 1, (n_fetches,), device=weight.device)
        fetch_c_indices = torch.randint(0, weight.shape[1] - fetch_unit + 1, (n_fetches,), device=weight.device)
        
        num_sums_list = [10, 8, 6, 4, 2]
        for num_sums in num_sums_list:
            ext_logger.write(f"Num Sums: {num_sums}\n")
            mse_list = []
            for i in range(n_fetches):
                r_start = fetch_r_indices[i]
                r_end = min(r_start + fetch_unit, weight.shape[0])
                c_start = fetch_c_indices[i]
                c_end = min(c_start + fetch_unit, weight.shape[1])
                weight_fetch = weight[r_start:r_end, c_start:c_end]
                
                logger.info(f"shape of weight_fetch: {weight_fetch.shape}")
                
                restored_weight_sbvr = sbvr(weight_fetch, num_sums=num_sums).get_decoded_tensor()
                
                mse, max_error, min_error, std_dev = get_errors(weight_fetch, restored_weight_sbvr)
                mse_list.append(mse)
                log_text = (
                    f"Fetch {i + 1}/{n_fetches} (num_sums={num_sums}): "
                    f"MSE: {mse:.8e}, "
                    f"Max Error: {max_error:.4e}, "
                    f"Min Error: {min_error:.4e}, "
                    f"Std Dev: {std_dev:.4e}"
                )
                ext_logger.write(log_text + "\n")
            ext_logger.write(f"Average mse: {sum(mse_list)/len(mse_list):.8e}")
            ext_logger.write("\n")
            
    random_fetcher_test(fetch_unit=16, n_fetches=60, weight=ffn_weight)
    logger.info("Test completed. Check the log file for details.")
    
if __name__ == "__main__":
    randn_test()
    # test_with_llama3_weight()