import torch
import itertools
import math
import os
import sys
import datetime
import time
from tqdm import tqdm

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
        raise ValueError("Tensors must have the same shape")
    
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

class sbvr(): 
    def __init__(self, 
                 data: torch.Tensor = None, 
                 num_sums: int = 4,
                 coeff_group_size: int = 512,
                 r_search_num = 80,
                 s_search_num = 48,
                 b_search_num = 48,
                 min_search_cache_num = 8,
                 max_coeff_search_cache_num = 96,
                 max_bias_search_cache_num = 80,
                 max_mse_window_size: int = 20,
                 search_extend_ratio: float = 1.6,
                 coeff_dtype: torch.dtype = None,
                 bvr_dtype: torch.dtype = torch.int32,
                 compute_dtype: torch.dtype = torch.float16):
        if data is None:
            raise ValueError(r_str("Data cannot be None"))
            
        self.num_sums = num_sums
        self.coeff_group_size = coeff_group_size
        self.coeff_dtype = data.dtype if coeff_dtype is None else coeff_dtype
        self.bvr_dtype = bvr_dtype
        self.compute_dtype = compute_dtype
        
        self.r_search_num = r_search_num
        self.s_search_num = s_search_num
        self.b_search_num = b_search_num
        
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
        
        self.extend_ratio = search_extend_ratio
        elem_size = torch.tensor(0, dtype=self.compute_dtype, 
                                 device=data.device).element_size()
        diff_mat_size = 3 * b_search_num * self.extend_ratio * \
            (2**self.num_sums) * coeff_group_size * elem_size
            
        total_mem = torch.cuda.mem_get_info()[0]
        self.search_batch_size = int(total_mem * 0.9 / diff_mat_size)
        self.min_search_cache_num = min_search_cache_num
        self.max_coeff_search_cache_num = max_coeff_search_cache_num
        self.max_bias_search_cache_num = max_bias_search_cache_num
        self.max_mse_window_size = max_mse_window_size
        self.search_cache = {"coeff": [], "bias": [], "mse": []}
        self.cache_hits = 0
        self.runs = 0
        
        self.bin_combs = torch.tensor(
            list(itertools.product([0, 1], repeat=num_sums)),
            dtype=self.coeff_dtype, device=data.device
        )
        
        self.bvr = None
            
        coeff_idx = self.__encode_to_sbvr(data)
        self.__change_coeff_idx_to_bvr(coeff_idx)
        
    @torch.inference_mode()
    def __get_all_points(self, coeff_bias: torch.tensor, coeff: torch.tensor):
        num_coeff = coeff.shape[0]
        return self.bin_combs @ coeff + coeff_bias
    
    @torch.inference_mode()
    def __get_search_space_from_lists(self, r_list, b_list, s_list):
        exponents = torch.arange(self.num_sums, device=r_list.device) 
        search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
        search_space = \
            search_space / torch.sum(search_space, dim=1).unsqueeze(1)
        search_space = s_list.view(-1, 1, 1) * search_space.unsqueeze(0)
        search_space = search_space.view(-1, self.num_sums)

        return search_space, r_list, b_list, s_list
    
    @torch.inference_mode()
    def __get_search_space(self, data, extended=False, extend_ratio=1.4):
        data_max = torch.max(data)
        data_avg = torch.mean(data)
        data_min = torch.min(data)
        data_97 = torch.quantile(data.to(torch.float32), 0.97)

        if not extended:
            r_max = (math.pi*2/3 + 0.1)
            r_min = 1.0
            r_gran = (r_max - r_min) / self.r_search_num
            b_max = data_avg 
            b_min = data_min - abs(data_min) * 0.1
            b_gran = (b_max - b_min) / self.b_search_num
            s_max = (data_max - data_min) * 1.1
            s_min = (data_97 - data_avg) * 2.0
            s_gran = (s_max - s_min) / self.s_search_num
        else:
            print (r_str("\tUsing extended search space..."))
            r_max = (math.pi*2/3 + 0.1)
            r_min = 1.0
            r_gran = (r_max - r_min) / (self.r_search_num * extend_ratio)
            b_max = data_avg + abs(data_avg) * 0.2
            b_min = data_min - abs(data_min) * 0.2
            b_gran = (b_max - b_min) / (self.b_search_num * extend_ratio)
            s_max = (data_max - data_min) * 1.2
            data_92 = torch.quantile(data.to(torch.float32), 0.92)
            s_min = (data_92 - data_avg) * 2.0
            s_gran = (s_max - s_min) / (self.s_search_num * extend_ratio)
            
        print(b_str("\tNum_sums: ") + f"{self.num_sums}")
        print(y_str("\t\tR search range: ") + f"{r_min:.4e} to {r_max:.4e}, " +
              y_str("search granularity: ") + f"{r_gran:.4e}")
        print(y_str("\t\tBias search range: ") + 
              f"{b_min:.4e} to {data_avg:.4e}, " +
              y_str("search granularity: ") + f"{b_gran:.4e}")
        print(y_str("\t\tScale search range: ") + 
              f"{s_min:.4e} to {s_max:.4e}, " +
              y_str("search granularity: ") + f"{s_gran:.4e}")
        
        r_list = torch.arange(r_min + r_gran, r_max + r_gran, r_gran, 
                              device=data.device, dtype=data.dtype)
        s_list = torch.arange(s_min + s_gran, s_max + s_gran, s_gran, 
                              device=data.device, dtype=data.dtype)
        b_list = torch.arange(b_min + b_gran, b_max + b_gran, b_gran, 
                              device=data.device, dtype=data.dtype)
        return self.__get_search_space_from_lists(r_list, b_list, s_list)
    
    @torch.inference_mode()
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
    
    @torch.inference_mode()
    def __encode_data(self, data):
        
        min_mse = float("inf")
        self.runs += 1
        # Check cached search space
        if (len(self.search_cache["coeff"]) >= self.min_search_cache_num):
            search_space = torch.tensor(self.search_cache["coeff"],
                                        device=data.device,
                                        dtype=data.dtype)
            b_list = torch.tensor(self.search_cache["bias"],
                                 device=data.device,
                                 dtype=data.dtype)
            biased_data = data.unsqueeze(0) - b_list.view(-1, 1)
            len_search_space = search_space.shape[0]
            for search_start in \
                range(0, len_search_space, self.search_batch_size):
                torch.cuda.empty_cache()
                search_end = \
                    min(search_start + self.search_batch_size, len_search_space)
                coeff_list = search_space[search_start:search_end]

                # Call a method to get the index and MSE among these coefficients
                mse, bias_idx, coeff_set_idx, coeff_comb_idx = \
                    self.__get_min_mse_coeff(biased_data, coeff_list)
            
                search_space_idx = search_start + coeff_set_idx

                if mse < min_mse:
                    min_mse = mse
                    best_bias = b_list[bias_idx]
                    best_coeff = search_space[search_space_idx]
                    best_coeff_idx = coeff_comb_idx
                    best_r = -1.0
                    best_s = -1.0
            window_size = \
                min(len(self.search_cache["mse"]), self.max_mse_window_size)
            mse_window = self.search_cache["mse"][-window_size:]
            cutoff_mse = sum(mse_window) / len(mse_window)
            if min_mse < cutoff_mse:
                self.cache_hits += 1
                coeff_str = ['%.4f' % elem for elem in best_coeff.tolist()]
                # print (b_str("\nCache Hit ") +
                #     f"(Hitrate: {self.cache_hits/self.runs:.2f}) - " +
                #     y_str("Best MSE: ") + f"{min_mse:.4e}" +
                #     ", " + y_str("Cutoff MSE: ") + f"{cutoff_mse:.4e}" +
                #     ", " + y_str("Coeff: ") + str(coeff_str) +
                #     ", " + y_str("Bias: ") + f"{best_bias.item():.4e}")
                return best_bias, best_coeff, best_coeff_idx
            else:
                coeff_str = ['%.4f' % elem for elem in best_coeff.tolist()]
                print (r_str("\n\tCache Miss ") +
                    f"(Hitrate: {self.cache_hits/self.runs:.2f}) - " +
                    y_str("Cutoff MSE: ") + f"{cutoff_mse:.4e}" +
                    ", " + y_str("Best MSE: ") + f"{min_mse:.4e}" +
                    ", " + y_str("Bias: ") + f"{best_bias.item():.4e} " +
                    y_str("\n\t\tCoeff: ") + str(coeff_str))
        else:
            print(r_str("\n\tWarming up cache... "))

        search_space, r_list, b_list, s_list = \
            self.__get_search_space(data, (self.cache_hits / self.runs) > 0.5)
        biased_data = data.unsqueeze(0) - b_list.view(-1, 1)
        len_search_space = search_space.shape[0]

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
                
        # Cache the results
        if len(self.search_cache["coeff"]) < self.max_coeff_search_cache_num:
            cur_coeff = best_coeff.tolist()
            for coeff in self.search_cache["coeff"]:
                if all(abs(cur_coeff[i] - coeff[i]) < abs(cur_coeff[i] * 0.005) 
                       for i in range(min(3, len(cur_coeff)))):
                    break
            else:
                self.search_cache["coeff"].append(best_coeff.tolist())
        if len(self.search_cache["bias"]) < self.max_bias_search_cache_num:
            cur_bias = best_bias.item()
            for bias in self.search_cache["bias"]:
                if abs(cur_bias - bias) < abs(cur_bias * 0.01):
                    break
            else:
                self.search_cache["bias"].append(best_bias.item())
        self.search_cache["mse"].append(min_mse)
        coeff_str = ['%.4f' % elem for elem in best_coeff.tolist()]
        print (g_str("\tBest MSE: ") + f"{min_mse:.4e}" +
            ", " + y_str("(r, b, s): ") + f"{best_r:.4e}, " +
            f"{best_bias.item():.4e}, {best_s:.4e}" +
            y_str("\n\t\tCoeff: ") + str(coeff_str))
                
        return best_bias, best_coeff, best_coeff_idx
    
    @torch.inference_mode()
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
        global_coeff_idx = torch.empty((data_num), dtype=int, 
                                     device=self.coeff.device)
        
        for i in tqdm(range(num_coeff_groups), ncols=80, 
                      desc="Encoding groups", unit="group"):
            # logger.info(f"Encoding group {i + 1}/{num_coeff_groups}")
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, data_num)
            group_data = \
                data.flatten()[group_start:group_end].to(self.compute_dtype)
            coeff_bias, coeff, coeff_idx = self.__encode_data(group_data)
            self.coeff_bias[i] = coeff_bias.item()
            self.coeff[i] = coeff
            global_coeff_idx[group_start:group_end] = coeff_idx
            
        return global_coeff_idx
            
    @torch.inference_mode()
    def __dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @torch.inference_mode()
    def __bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)
            
    @torch.inference_mode()
    def __change_coeff_idx_to_bvr(self, coeff_idx):
        self.coeff_idx_len = coeff_idx.shape[0]
        bin_vec = self.__dec2bin(coeff_idx, self.num_sums).to(self.bvr_dtype)
        num_bits = bin_vec.element_size() * 8
        bin_vec = bin_vec.to(torch.int64).transpose(0, 1)
        padded_len = (bin_vec.shape[1] + num_bits - 1) // num_bits * num_bits
        bin_vec_padded = torch.zeros((self.num_sums, padded_len),
                          dtype=torch.int64, device=self.coeff.device)
        bin_vec_padded[:, :bin_vec.shape[1]] = bin_vec
        bin_vec_padded = bin_vec_padded.view(-1, num_bits)
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int64, device=self.coeff.device)
        bvr = torch.sum(bin_vec_padded * powers.unsqueeze(0), dim=1)
        self.bvr = bvr.view(self.num_sums, -1).to(self.bvr_dtype)
     
    @torch.inference_mode()
    def __change_bvr_to_coeff_idx(self):
        num_bits = self.bvr.element_size() * 8
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int32, device=self.coeff.device)
        bin_vec_padded = ((self.bvr.unsqueeze(-1) & powers) != 0)
        bin_vec_padded = bin_vec_padded.view(self.num_sums, -1).to(torch.int32)
        bin_vec = bin_vec_padded[:, :self.coeff_idx_len]
        bin_vec = bin_vec.transpose(0, 1)
        coeff_idx = self.__bin2dec(bin_vec, self.num_sums)
        coeff_idx = coeff_idx.view(-1)
        return coeff_idx
            
    @torch.inference_mode()
    def get_decoded_tensor(self):
        decoded_tensor = torch.empty(self.original_data_shape,
                                      dtype=self.original_dtype,
                                      device=self.coeff.device)
        num_coeff_groups = self.coeff_bias.shape[0]
        coeff_idx = self.__change_bvr_to_coeff_idx()
        for i in range(num_coeff_groups):
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, decoded_tensor.numel())
            group_coeff_bias = self.coeff_bias[i]
            group_coeff = self.coeff[i]
            group_coeff_idx = coeff_idx[group_start:group_end]
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
        y_str("\n\tBinary Vector Data Type: ") + str(self.bvr_dtype) + \
        y_str("\n\tCoefficient Tensor Size: ") + str(self.coeff.shape)
        return info_str


def randn_test(mat_len=512, sbvr_max_sums=6):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_size = (mat_len, mat_len)

    mat_a = torch.randn(mat_size, dtype=torch.float64, device=device)*0.3
    mat_b = torch.randn(mat_size, dtype=torch.float64, device=device)*0.3
    # print_tensor(mat_a, "mat_a")
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
    for i in range (sbvr_max_sums, 1, -1):
        time_start = time.time()
        sbvr_matmul = f64_matmul(sbvr(mat_a, num_sums=i).get_decoded_tensor(), 
                                sbvr(mat_b, num_sums=i).get_decoded_tensor())
        sbvr_dict[i] = sbvr_matmul
        time_dict[i] = time.time() - time_start

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
    
if __name__ == "__main__":
    mat_len = sys.argv[1]
    sbvr_max_sums = sys.argv[2]
    time_start = time.time()
    randn_test(int(mat_len), int(sbvr_max_sums))
    print (f"Total time taken: {time.time() - time_start:.4f} seconds")