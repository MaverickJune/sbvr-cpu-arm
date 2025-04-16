import torch
import itertools
import math
import os
import sys
import datetime
import time
import pickle
from tqdm import tqdm
from sbvr.sbvr_cuda import sbvr_mat_vec_mult

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

class sbvr(): 
    def __init__(self, 
                 data: torch.Tensor = None, 
                 num_sums: int = 4,
                 verbose_level: int = 1,
                 coeff_group_size: int = 128,
                 r_search_num = 80,
                 s_search_num = 48,
                 b_search_num = 48,
                 cache_warmup_num = 8,
                 mse_window_size: int = 20,
                 search_extend_ratio: float = 1.6,
                 bvr_dtype: torch.dtype = torch.int32,
                 cache_idx_dtype: torch.dtype = torch.uint8 ,
                 compute_dtype: torch.dtype = torch.float16):
        if data is None:
            raise ValueError(r_str("Data cannot be None"))
        
        self.use_bias = True
        self.num_sums = num_sums
        self.verbose_level = verbose_level
        self.coeff_group_size = coeff_group_size
        self.bvr_dtype = bvr_dtype
        self.compute_dtype = compute_dtype
        if num_sums > 11 and compute_dtype == torch.float16:
            raise UserWarning(
                r_str("Warning: compute_dtype float16 does not have sufficient "
                      "precision for num_sums > 11."))
        
        self.b_search_num = b_search_num
        if not self.use_bias:
            self.r_search_num = r_search_num * 2
            self.s_search_num = s_search_num * 2
        else:
            self.r_search_num = r_search_num
            self.s_search_num = s_search_num
        
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
        
        # Memory settings
        self.extend_ratio = search_extend_ratio
        elem_size = torch.tensor(0, dtype=self.compute_dtype).element_size()
        diff_mat_size = 3 * b_search_num * self.extend_ratio * \
            (2**self.num_sums) * coeff_group_size * elem_size
        total_mem = torch.cuda.mem_get_info()[0]
        self.search_batch_size = int(total_mem * 0.9 / diff_mat_size)
        
        # Cache settings
        self.cache_idx_dtype = cache_idx_dtype
        self.cache_warmup_num = cache_warmup_num
        elem_size = torch.tensor(0, dtype=cache_idx_dtype).element_size() * 8
        self.coeff_cache = torch.zeros((2**elem_size, num_sums),
            dtype=self.compute_dtype, device=data.device)
        self.bias_cache = torch.zeros((2**elem_size),
            dtype=self.compute_dtype, device=data.device)
        self.mse_window_size = mse_window_size
        self.acceptable_mse = 10**-12
        self.mse_history = []
        self.num_coeff_cache_lines = 0
        self.num_bias_cache_lines = 0
        self.cache_hits = 0
        self.group_idx = 0
        
        self.bin_combs = torch.tensor(
            list(itertools.product([0, 1], repeat=num_sums)),
            dtype=self.compute_dtype, device=data.device
        )
        
        self.bvr = None
        self._change_coeff_sel_to_bvr(self._encode_to_sbvr(data))
        
    @torch.inference_mode()
    def check_coeff_cache_full(self):
        if self.num_coeff_cache_lines >= self.coeff_cache.shape[0]:
            print(r_str("Warning: Coefficient cache is full - Cache size: ") +
                  f"{self.coeff_cache.shape[0]}")
            return True
        return False

    @torch.inference_mode()
    def check_bias_cache_full(self):
        if self.num_bias_cache_lines >= self.bias_cache.shape[0]:
            print(r_str("Warning: Bias cache is full. - Cache size: ") +
                  f"{self.bias_cache.shape[0]}")
            return True
        return False
        
    @torch.inference_mode()
    def _get_all_points(self, coeff_bias: torch.tensor, coeff: torch.tensor):
        return self.bin_combs @ coeff + coeff_bias
    
    @torch.inference_mode()
    def _get_coeff_search_space_from_lists(self, r_list, b_list, s_list):
        exponents = torch.arange(self.num_sums, device=r_list.device) 
        search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
        search_space = \
            search_space / torch.sum(search_space, dim=1).unsqueeze(1)
        search_space = s_list.view(-1, 1, 1) * search_space.unsqueeze(0)
        search_space = search_space.view(-1, self.num_sums)

        return search_space, r_list, b_list, s_list
    
    @torch.inference_mode()
    def _get_coeff_search_space(self, data, extended=False, extend_ratio=1.4):
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
            if self.verbose_level > 0:
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
        if self.verbose_level > 1:
            print(b_str("\tNum_sums: ") + f"{self.num_sums}",
                    ", " + y_str("Data range: ") + 
                    f"{data_min:.4e} to {data_max:.4e}" +
                    ", " + y_str("avg: ") + f"{data_avg:.4e}" +
                    ", " + y_str("97%: ") + f"{data_97:.4e}")
            print(y_str("\t\tR search range: ") + 
                  f"{r_min:.4e} to {r_max:.4e}, " +
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
        if not self.use_bias:
            r_list = -r_list
            b_list = torch.tensor([0], device=data.device, dtype=data.dtype)
        
        return self._get_coeff_search_space_from_lists(r_list, b_list, s_list)
    
    @torch.inference_mode()
    def _get_min_mse_coeff(self, biased_data, search_matrix):
        candidiate_matrix = search_matrix @ self.bin_combs.T
        
        data_size = biased_data.shape[1]
        bias_list_size = biased_data.shape[0]
        n_ss_row = candidiate_matrix.shape[0]
        n_ss_col = candidiate_matrix.shape[1]
        
        biased_data = biased_data.view(bias_list_size, 1, data_size, 1)
        candidiate_matrix = candidiate_matrix.view(1, n_ss_row, 1, n_ss_col) 
        
        diff = (biased_data - candidiate_matrix)**2

        # (bias_list_size, n_ss_row, data_size)
        diff_selected, coeff_comb_indices = diff.min(dim=-1) 
        mse = diff_selected.to(torch.float32).mean(dim=-1)
        
        flat_min_idx = mse.view(-1).argmin()
        min_idx = torch.unravel_index(flat_min_idx, mse.shape)
        bias_idx, coeff_set_idx = min_idx
        coeff_comb_sel = coeff_comb_indices[bias_idx, coeff_set_idx]
        min_mse = mse[bias_idx, coeff_set_idx].item()

        return min_mse, coeff_set_idx, bias_idx, coeff_comb_sel
    
    @torch.inference_mode()
    def _search_coeff_bias_space(self, coeff_search_space, 
                                  biased_data, min_mse):
        len_search_space = coeff_search_space.shape[0]
        best_bias_idx = -1
        best_coeff_idx = -1
        best_coeff_sel = -1
        # Loop over the bias values
        for search_start in range(0, len_search_space, self.search_batch_size):
            torch.cuda.empty_cache()
            search_end = \
                min(search_start + self.search_batch_size, len_search_space)
            coeff_list = coeff_search_space[search_start:search_end]

            # Call a method to get the index and MSE among these coefficients
            mse, coeff_set_idx, bias_idx, coeff_comb_sel = \
                self._get_min_mse_coeff(biased_data, coeff_list)
            
            search_space_idx = search_start + coeff_set_idx
            if mse < min_mse:
                min_mse = mse
                best_bias_idx = bias_idx
                best_coeff_idx = search_space_idx
                best_coeff_sel = coeff_comb_sel
                if mse < self.acceptable_mse:
                    break
        return min_mse, best_coeff_idx, best_bias_idx, best_coeff_sel
    
    @torch.inference_mode()
    def _encode_data(self, data):
        min_mse = float("inf")
        self.group_idx += 1
        # Check cached search space
        if (self.num_coeff_cache_lines >= self.cache_warmup_num):
            # Setup the search space
            coeff_search_space = self.coeff_cache[:self.num_coeff_cache_lines]
            b_list = self.bias_cache[:self.num_bias_cache_lines]
            biased_data = data.unsqueeze(0) - b_list.view(-1, 1)
            
            # Search the cache for the best coeff and bias
            min_mse, best_coeff_idx, best_bias_idx, best_coeff_sel = \
                self._search_coeff_bias_space(coeff_search_space, biased_data,
                                               min_mse)
            best_coeff_str = ['%.4f' % elem for elem in 
                              coeff_search_space[best_coeff_idx].tolist()]
            best_bias = b_list[best_bias_idx].item()
                
            # Check if the best coeff and bias satisfy the cutoff
            window_size = min(len(self.mse_history), self.mse_window_size)
            mse_window = self.mse_history[-window_size:]
            cutoff_mse = (sum(mse_window) / len(mse_window))
            if cutoff_mse < self.acceptable_mse:
                cutoff_mse = self.acceptable_mse
            if min_mse < cutoff_mse:
                self.cache_hits += 1
                return best_coeff_idx, best_bias_idx, best_coeff_sel
            else:
                if self.verbose_level > 0:
                    print (b_str("\n\tGroup ") + f"{self.group_idx}: " 
                        + r_str("Cache Miss ") +
                        f"(Hitrate: {self.cache_hits/self.group_idx:.2f}) - " +
                        y_str("Coeff cache: ") +
                        f"{self.num_coeff_cache_lines}/" +
                        f"{self.coeff_cache.shape[0]}" +
                        ", " + y_str(", Bias cache: ") +
                        f"{self.num_bias_cache_lines}/" +
                        f"{self.bias_cache.shape[0]}" +
                        y_str("\n\t\tCutoff MSE: ") + f"{cutoff_mse:.4e}" +
                        ", " + y_str("Best MSE: ") + f"{min_mse:.4e}" +
                        ", " + y_str("Bias: ") + f"{best_bias:.4e} " +
                        y_str("\n\t\tCoeff: ") + str(best_coeff_str))
        else:
            if self.verbose_level > 0:
                print(b_str("\n\tRun ") + f"{self.group_idx}: " +
                    r_str("Warming up cache... "))

        # Setup the search space
        coeff_search_space, r_list, b_list, s_list = \
            self._get_coeff_search_space(data, 
                                         (self.cache_hits/self.group_idx) > 0.4)
        biased_data = data.unsqueeze(0) - b_list.view(-1, 1)
        
        # Search the cache for the best coeff and bias
        old_min_mse = min_mse  
        min_mse, new_coeff_idx, new_bias_idx, new_coeff_sel = \
                self._search_coeff_bias_space(coeff_search_space, biased_data,
                                               min_mse)
        if min_mse >= old_min_mse:
            # If the new search space is NOT better than the cached one:
            min_mse = old_min_mse
            best_r = -1
            best_s = -1
        else:
            # If the new search space is better than the cached one:
            # Cache the results
            save_fail = False
            if not self.check_coeff_cache_full():
                self.coeff_cache[self.num_coeff_cache_lines] =\
                    coeff_search_space[new_coeff_idx]
                self.num_coeff_cache_lines += 1
            else:
                save_fail = True
            if not self.check_bias_cache_full():
                self.bias_cache[self.num_bias_cache_lines] = \
                    b_list[new_bias_idx]
                self.num_bias_cache_lines += 1
            else:
                save_fail = True
            if not save_fail:
                # If caching was successful, update the output
                best_bias_idx = self.num_bias_cache_lines - 1
                best_coeff_idx = self.num_coeff_cache_lines - 1
                best_coeff_sel = new_coeff_sel
                best_coeff_str = ['%.4f' % elem for elem in \
                    self.coeff_cache[best_coeff_idx].tolist()]
                best_bias = self.bias_cache[best_bias_idx].item()
                best_r = r_list[new_coeff_idx % len(r_list)]
                best_s = s_list[new_coeff_idx // len(r_list)]   
            self.mse_history.append(min_mse)
            
        if self.verbose_level > 0:
            print(g_str("\tBest MSE: ") + f"{min_mse:.4e}" +
                ", " + y_str("(bias, r, s): ") +
                f"{best_bias:.4e}, {best_r:.4e}, {best_s:.4e}" +
                y_str("\n\t\tCoeff: ") + str(best_coeff_str))
                
        return best_coeff_idx, best_bias_idx, best_coeff_sel
    
    @torch.inference_mode()
    def _encode_to_sbvr(self, data):
        print(b_str("Encoding to SBVR..."))
        data_num = data.numel()
        num_coeff_groups = \
            (data_num + self.coeff_group_size - 1) // self.coeff_group_size
        self.coeff_idx = torch.empty((num_coeff_groups), 
                                    dtype=self.cache_idx_dtype, 
                                    device=data.device)
        self.bias_idx = torch.empty((num_coeff_groups), 
                                    dtype=self.cache_idx_dtype, 
                                    device=data.device)
        out_coeff_sel = torch.empty((data_num), dtype=torch.int32,
                                     device=data.device)
        
        for i in tqdm(range(num_coeff_groups), ncols=80, 
                      desc="Encoding groups", unit="group"):
            torch.cuda.empty_cache()
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, data_num)
            group_data = \
                data.flatten()[group_start:group_end].to(self.compute_dtype)
            g_coeff_idx, g_bias_idx, coeff_sel = self._encode_data(group_data)
            self.coeff_idx[i] = g_coeff_idx
            self.bias_idx[i] = g_bias_idx
            out_coeff_sel[group_start:group_end] = coeff_sel

        return out_coeff_sel
            
    @torch.inference_mode()
    def _dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @torch.inference_mode()
    def _bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)
            
    @torch.inference_mode()
    def _change_coeff_sel_to_bvr(self, coeff_sel):
        self.coeff_sel_len = coeff_sel.shape[0]
        num_bits = torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        padded_len = (self.coeff_sel_len + num_bits - 1) // num_bits * num_bits
        self.bvr = torch.zeros((self.num_sums, (padded_len // num_bits)),
                          dtype=self.bvr_dtype, device=self.coeff_cache.device)
        powers = 2 ** torch.arange(num_bits, dtype=torch.int64, 
                                   device=self.coeff_cache.device)
        coeff_sel = torch.cat((coeff_sel, 
                               torch.zeros(padded_len - self.coeff_sel_len,
                                           dtype=coeff_sel.dtype,
                                           device=coeff_sel.device)))
        iter_size = 65536
        for i in range(0, padded_len, iter_size):
            max_i = min(i + iter_size, padded_len)
            coeff_sel_i = coeff_sel[i:max_i]
            bin_vec = self._dec2bin(coeff_sel_i, self.num_sums).to(torch.int64)
            bin_vec = \
                bin_vec.transpose(0, 1).reshape(self.num_sums, -1, num_bits)
            bvr_i = torch.sum(bin_vec * powers.unsqueeze(0), dim=2)
            self.bvr[:, i//32:max_i//32] = bvr_i
     
    @torch.inference_mode()
    def _change_bvr_to_coeff_sel(self):
        num_bits = self.bvr.element_size() * 8
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int32, 
                                   device=self.coeff_cache.device)
        coeff_sel = torch.empty((self.coeff_sel_len),
                               dtype=self.bvr.dtype,
                               device=self.coeff_cache.device)
        iter_size = 2048
        for i in range(0, self.bvr.shape[1], iter_size):
            max_i = min(i + iter_size, self.coeff_sel_len)
            bvr_i = self.bvr[:, i:max_i]
            bin_vec = ((bvr_i.unsqueeze(-1) & powers) != 0).to(torch.int32)
            bin_vec = bin_vec.view(self.num_sums, -1)
            max_coeff_i = min(max_i*num_bits, self.coeff_sel_len)
            bin_vec_trunc = bin_vec[:, :max_coeff_i].transpose(0, 1)
            coeff_sel_i = self._bin2dec(bin_vec_trunc, self.num_sums)
            coeff_sel[i*num_bits:max_coeff_i] = coeff_sel_i.view(-1)

        return coeff_sel
            
    @torch.inference_mode()
    def get_decoded_tensor(self):
        decoded_tensor = torch.empty(self.original_data_shape,
                                      dtype=self.original_dtype,
                                      device=self.coeff_cache.device)
        num_coeff_groups = self.coeff_idx.shape[0]
        coeff_sel = self._change_bvr_to_coeff_sel()
        for i in range(num_coeff_groups):
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, decoded_tensor.numel())
            group_coeff_bias = self.bias_cache[self.bias_idx[i].item()]
            group_coeff = self.coeff_cache[self.coeff_idx[i].item()]
            group_coeff_sel = coeff_sel[group_start:group_end]
            group_all_points = \
                self._get_all_points(group_coeff_bias, group_coeff)
            group_data = group_all_points[group_coeff_sel]
            decoded_tensor.flatten()[group_start:group_end] = group_data
        
        return decoded_tensor
    
    @torch.inference_mode()
    def cuda_matrix_vec_mul(self, vec1, vec2):
        return sbvr_mat_vec_mult(vec1, vec2)

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
    
def save_sbvr(sbvr_obj, filename):
    if not isinstance(sbvr_obj, sbvr):
        raise ValueError("The object is not a valid SBVR object.")
    torch.save(sbvr_obj, filename)
        
def load_sbvr(filename) -> sbvr:
    sbvr_obj = torch.load(filename)
    if not isinstance(sbvr_obj, sbvr):
        raise ValueError("The loaded object is not a valid SBVR object.")
    return sbvr_obj