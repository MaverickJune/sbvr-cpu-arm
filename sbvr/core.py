import torch
import itertools
import math
import copy
from tqdm import tqdm
from sbvr.sbvr_cuda import sbvr_mm

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
                 cgroup_len: int = 128,
                 r_search_num = 64,
                 b_search_num = 40,
                 s_search_num = 40,
                 cache_warmup_num = 8,
                 mse_window_size: int = 20,
                 search_extend_ratio: float = 1.2,
                 compute_dtype: torch.dtype = torch.float16,
                 load_only: bool = False,
                 **kwargs):
        if load_only:
            def load_from_kwargs(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            load_from_kwargs(self, **kwargs)
            return
        
        if data is None:
            raise ValueError(r_str("Data cannot be None"))
        
        self.num_sums = num_sums
        self.verbose_level = verbose_level
        self.cgroup_len = cgroup_len # Coefficient group length
        self.bvr_dtype = torch.uint32
        self.bvr_num_bits = \
            torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        if self.cgroup_len % self.bvr_num_bits != 0:
            raise ValueError(
                r_str("Coefficient group length must be a multiple of ") +
                f"{self.bvr_num_bits}")
        self.compute_dtype = compute_dtype
        if num_sums > 11 and compute_dtype == torch.float16:
            raise UserWarning(
                r_str("Warning: compute_dtype float16 does not have sufficient "
                      "precision for num_sums > 11."))
        
        self.r_search_num = r_search_num
        self.b_search_num = b_search_num
        self.s_search_num = s_search_num
        
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
        
        # Memory settings
        self.extend_ratio = search_extend_ratio
        elem_size = torch.tensor(0, dtype=self.compute_dtype).element_size()
        diff_mat_size = 3 * self.extend_ratio * (2**self.num_sums) \
            * cgroup_len * elem_size
        total_mem = torch.cuda.mem_get_info(data.device)[0]
        self.search_batch_size = int(total_mem * 0.8 / diff_mat_size)
        
        # Cache settings
        self.cache_idx_dtype = torch.uint16
        self.cache_warmup_num = cache_warmup_num
        elem_size = torch.tensor(0, dtype=self.cache_idx_dtype).element_size() 
        self.coeff_cache = torch.zeros((2**(8*elem_size), num_sums),
            dtype=self.compute_dtype, device=data.device)
        self.mse_window_size = mse_window_size
        self.acceptable_mse = 10**-12
        self.mse_history = []
        self.num_coeff_cache_lines = 0
        self.cache_hits = 0
        self.group_idx = 0
        
        # Pad the data to the nearest multiple of cgroup_len
        pad_length = (data.shape[-1] + self.cgroup_len - 1) // \
                        self.cgroup_len * self.cgroup_len
        if pad_length != data.shape[-1]:
            new_shape = list(data.shape)
            new_shape[-1] = pad_length
            data_padded = torch.zeros(new_shape, 
                                      dtype=data.dtype, device=data.device)
            slices = tuple(slice(0, s) for s in data.shape)
            data_padded[slices] = data
        else:
            data_padded = data
            
        self.padded_data_shape = data_padded.shape
        
        self.bvr = None
        self._change_coeff_sel_to_bvr(self._encode_to_sbvr(data_padded))
        
    @torch.inference_mode()
    def _get_bin_combs(self):
        if not hasattr(self, 'bin_combs'):
            self.bin_combs = torch.tensor(
                list(itertools.product([0, 1], repeat=self.num_sums)),
                dtype=self.compute_dtype, device=self.coeff_cache.device
            )
        return self.bin_combs
        
    @torch.inference_mode()
    def _check_coeff_cache_full(self):
        if self.num_coeff_cache_lines >= self.coeff_cache.shape[0]:
            return True
        return False
        
    @torch.inference_mode()
    def _get_all_points(self, coeff: torch.tensor):
        return self._get_bin_combs() @ coeff
    
    @torch.inference_mode()
    def _get_coeff_search_space_from_lists(self, r_list, b_list, s_list):
        exponents = torch.arange(self.num_sums, device=r_list.device) 
        search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
        search_space = \
            search_space / torch.sum(search_space, dim=1).unsqueeze(1)
        search_space = s_list.view(-1, 1, 1) * search_space.unsqueeze(0)
        search_space = b_list.view(-1, 1, 1, 1) + search_space.unsqueeze(0)
        search_space = search_space.view(-1, self.num_sums)

        return search_space, r_list, b_list, s_list
    
    @torch.inference_mode()
    def _get_coeff_search_space(self, data, extended=False):
        data_max = torch.max(data)
        data_avg = torch.mean(data)
        data_min = torch.min(data)

        r_max = (math.pi*2/3 + 0.1)
        r_min = 1.0
        r_gran = (r_max - r_min) / self.r_search_num 
        b_max = abs(data_avg) * 14.0 / self.num_sums 
        b_min = -b_max
        b_gran = (b_max - b_min) / self.b_search_num 
        s_max = ((data_max - data_min) / 2.0) * 1.2 
        s_min = 0.0
        s_gran = (s_max - s_min) / self.s_search_num 
        
        if extended:
            if self.verbose_level > 0:
                print (r_str("\tUsing extended search space..."))
            r_gran /= self.extend_ratio
            b_gran /= self.extend_ratio
            s_gran /= self.extend_ratio
            
        if self.verbose_level > 1:
            print(b_str("\tNum_sums: ") + f"{self.num_sums}",
                    ", " + y_str("Data range: ") + 
                    f"{data_min:.4e} to {data_max:.4e}" +
                    ", " + y_str("avg: ") + f"{data_avg:.4e}")
            print(y_str("\t\tR search range: ") + 
                  f"{r_min:.4e} to {r_max:.4e}, " +
                y_str("search granularity: ") + f"{r_gran:.4e}")
            print(y_str("\t\tBias search range: ") + 
                f"{b_min:.4e} to {b_max:.4e}, " +
                y_str("search granularity: ") + f"{b_gran:.4e}")
            print(y_str("\t\tScale search range: ") + 
                f"{s_min:.4e} to {s_max:.4e}, " +
                y_str("search granularity: ") + f"{s_gran:.4e}")
        
        r_list = -torch.arange(r_min + r_gran, r_max + r_gran, r_gran, 
                              device=data.device, dtype=data.dtype)
        if s_gran != 0:
            s_list = torch.arange(s_min + s_gran, s_max + s_gran, s_gran, 
                                device=data.device, dtype=data.dtype)
        else:
            s_list = torch.tensor([s_min], device=data.device, dtype=data.dtype)
        if b_gran != 0:
            b_list = torch.arange(b_min, b_max, b_gran, 
                                  device=data.device, dtype=data.dtype)
        else:
            b_list = torch.tensor([b_min], device=data.device, dtype=data.dtype)

        return self._get_coeff_search_space_from_lists(r_list, b_list, s_list)
    
    @torch.inference_mode()
    def _get_min_mse_coeff(self, data, search_matrix):
        candidiate_matrix = search_matrix @ self._get_bin_combs().T
        
        n_ss_row = candidiate_matrix.shape[0]
        n_ss_col = candidiate_matrix.shape[1]
        
        data = data.view(1, -1, 1)
        candidiate_matrix = candidiate_matrix.view(n_ss_row, 1, n_ss_col) 
        
        diff = (data - candidiate_matrix)**2

        diff_selected, coeff_comb_indices = diff.min(dim=-1) 
        mse = diff_selected.to(torch.float32).mean(dim=-1)
        
        min_idx = mse.argmin()
        coeff_comb_sel = coeff_comb_indices[min_idx]
        min_mse = mse[min_idx].item()

        return min_mse, min_idx, coeff_comb_sel
    
    @torch.inference_mode()
    def _search_coeff_bias_space(self, coeff_search_space, data, cutoff_mse):
        min_mse = float("inf")
        len_search_space = coeff_search_space.shape[0]
        best_coeff_idx = -1
        best_coeff_sel = -1
        # Loop over the bias values
        for search_start in range(0, len_search_space, self.search_batch_size):
            torch.cuda.empty_cache()
            search_end = \
                min(search_start + self.search_batch_size, len_search_space)
            coeff_list = coeff_search_space[search_start:search_end]
            # Call a method to get the index and MSE among these coefficients
            mse, min_idx, coeff_comb_sel = \
                self._get_min_mse_coeff(data, coeff_list)
            search_space_idx = search_start + min_idx
            if mse < min_mse:
                min_mse = mse
                best_coeff_idx = search_space_idx
                best_coeff_sel = coeff_comb_sel
                if min_mse < cutoff_mse:
                    break
        return min_mse, best_coeff_idx, best_coeff_sel
    
    @torch.inference_mode()
    def _encode_data(self, data):
        min_mse = float("inf")
        best_r = -1
        best_b = -1
        best_s = -1
        self.group_idx += 1
        # Check cached search space
        if (self.num_coeff_cache_lines >= self.cache_warmup_num):
            # Setup the search space
            coeff_search_space = self.coeff_cache[:self.num_coeff_cache_lines]
            # Setup the cutoff MSE 
            window_size = min(len(self.mse_history), self.mse_window_size)
            mse_window = self.mse_history[-window_size:]
            cutoff_mse = (sum(mse_window) / len(mse_window))
            if cutoff_mse < self.acceptable_mse:
                cutoff_mse = self.acceptable_mse

            # Search the cache for the best coeff and bias
            min_mse, best_coeff_idx, best_coeff_sel = \
                self._search_coeff_bias_space(coeff_search_space, 
                                              data, cutoff_mse)
            best_coeff_str = ['%.4f' % elem for elem in 
                              coeff_search_space[best_coeff_idx].tolist()]
            
            if min_mse < cutoff_mse:
                self.cache_hits += 1
                return best_coeff_idx, best_coeff_sel
            else:
                if self.verbose_level > 0:
                    print (b_str("\n\tGroup ") + f"{self.group_idx}: " 
                        + r_str("Cache Miss ") +
                        f"(Hitrate: {self.cache_hits/self.group_idx:.2f}) - " +
                        y_str("Coeff cache: ") +
                        f"{self.num_coeff_cache_lines}/" +
                        f"{self.coeff_cache.shape[0]}" +
                        y_str("\n\t\tCutoff MSE: ") + f"{cutoff_mse:.4e}" +
                        ", " + y_str("Best MSE: ") + f"{min_mse:.4e}" +
                        y_str("\n\t\tCoeff: ") + str(best_coeff_str))
        else:
            if self.verbose_level > 0:
                print(b_str("\n\tRun ") + f"{self.group_idx}: " +
                    r_str("Warming up cache... "))

        if not self._check_coeff_cache_full():
            coeff_search_space, r_list, b_list, s_list = \
                self._get_coeff_search_space(data, 
                                            (self.cache_hits/self.group_idx) 
                                                > 0.7)
            
            # Search the cache for the best coeff and bias  
            new_mse, new_coeff_idx, new_coeff_sel = \
                    self._search_coeff_bias_space(coeff_search_space, data, 
                                                  self.acceptable_mse)
            new_coeff_str = ['%.4f' % elem for elem in 
                             coeff_search_space[new_coeff_idx].tolist()]
            new_b = b_list[new_coeff_idx // (len(s_list) * len(r_list))]
            new_s = s_list[new_coeff_idx // len(r_list) % len(s_list)]
            new_r = r_list[new_coeff_idx % len(r_list)]
                    
            if self.verbose_level > 0:
                print(g_str("\tNew MSE: ") + f"{new_mse:.4e}" +
                    ", " + y_str("(r, b, s): ") +
                    f"{new_r:.4e}, {new_b:.4e}, {new_s:.4e}" +
                    y_str("\n\t\tCoeff: ") + str(new_coeff_str))
            if new_mse >= min_mse:
                # If the new search space is NOT better than the cached one:
                print(r_str("\t\tNo better coeff found: ") +
                      f"{new_mse:.4e} >= {min_mse:.4e}")
            else:
                # If the new search space is better than the cached one:
                # Cache the results
                coeff_diff = self.coeff_cache - \
                    coeff_search_space[new_coeff_idx].unsqueeze(0)
                avg_abs_coeff = coeff_search_space[new_coeff_idx].abs().sum(-1)
                mask = coeff_diff.abs().sum(-1) < avg_abs_coeff*0.0001
                if mask.any():
                    # If the coeff is already in the cache, use it
                    nonzero_idx = mask.nonzero(as_tuple=True)[0]
                    best_coeff_idx = nonzero_idx[0]
                else:
                    self.coeff_cache[self.num_coeff_cache_lines] =\
                        coeff_search_space[new_coeff_idx]
                    best_coeff_idx = self.num_coeff_cache_lines
                    self.num_coeff_cache_lines += 1

                # If caching was successful, update the output
                best_coeff_sel = new_coeff_sel
                self.mse_history.append(new_mse)
 
        return best_coeff_idx, best_coeff_sel
    
    @torch.inference_mode()
    def _encode_to_sbvr(self, data):
        data_num = data.numel()
        num_cgroups = \
            (data_num + self.cgroup_len - 1) // self.cgroup_len
        self.coeff_idx = torch.empty((num_cgroups), 
                                    dtype=self.cache_idx_dtype, 
                                    device=data.device)
        out_coeff_sel = torch.empty((data_num), dtype=torch.int32,
                                     device=data.device)
        
        for i in tqdm(range(num_cgroups), ncols=80, 
                      desc=b_str("Encoding SBVR groups"), unit="g"):
            torch.cuda.empty_cache()
            group_start = i * self.cgroup_len
            group_end = \
                min(group_start + self.cgroup_len, data_num)
            group_data = \
                data.flatten()[group_start:group_end].to(self.compute_dtype)
            g_coeff_idx, coeff_sel = self._encode_data(group_data)
            self.coeff_idx[i] = g_coeff_idx
            out_coeff_sel[group_start:group_end] = coeff_sel
        self.coeff_cache = \
            self.coeff_cache[:self.num_coeff_cache_lines].contiguous()

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
        coeff_sel_len = self.original_data_shape.numel()
        num_bits = self.bvr_num_bits
        padded_len = (coeff_sel_len + num_bits - 1) // num_bits * num_bits
        self.bvr = torch.zeros((self.num_sums, (padded_len // num_bits)),
                          dtype=self.bvr_dtype, device=self.coeff_cache.device)
        powers = 2 ** torch.arange(num_bits, dtype=torch.int64, 
                                   device=self.coeff_cache.device)
        coeff_sel = torch.cat((coeff_sel, 
                               torch.zeros(padded_len - coeff_sel_len,
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
            
        bvr_per_cgroup = self.cgroup_len // self.bvr_num_bits
        cgroup_per_inner_vec = self.padded_data_shape[-1] // self.cgroup_len
        self.bvr = self.bvr.view(self.num_sums, -1, cgroup_per_inner_vec,
                                 bvr_per_cgroup)
        self.bvr = self.bvr.transpose(0, 1).contiguous()
     
    @torch.inference_mode()
    def _change_bvr_to_coeff_sel(self):
        coeff_sel_len = self.original_data_shape.numel()
        bvr = self.bvr.transpose(0, 1).contiguous().view(self.num_sums, -1)
        num_bits = self.bvr_num_bits
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int64, 
                                   device=self.coeff_cache.device)
        coeff_sel = torch.empty((coeff_sel_len),
                               dtype=torch.int32, 
                               device=self.coeff_cache.device)
        iter_size = 2048
        for i in range(0, bvr.shape[1], iter_size):
            max_i = min(i + iter_size, coeff_sel_len)
            bvr_i = bvr[:, i:max_i].to(torch.int64)
            bin_vec = ((bvr_i.unsqueeze(-1) & powers) != 0).to(torch.int32)
            bin_vec = bin_vec.view(self.num_sums, -1)
            max_coeff_i = min(max_i*num_bits, coeff_sel_len)
            bin_vec_trunc = bin_vec[:, :max_coeff_i].transpose(0, 1)
            coeff_sel_i = self._bin2dec(bin_vec_trunc, self.num_sums)
            coeff_sel[i*num_bits:max_coeff_i] = coeff_sel_i.view(-1)

        return coeff_sel
            
    @torch.inference_mode()
    def decode(self):
        decoded_tensor = torch.empty(self.padded_data_shape,
                                      dtype=self.original_dtype,
                                      device=self.coeff_cache.device)
        num_cgroups = self.coeff_idx.shape[0]
        coeff_sel = self._change_bvr_to_coeff_sel()
        for i in range(num_cgroups):
            group_start = i * self.cgroup_len
            group_end = \
                min(group_start + self.cgroup_len, decoded_tensor.numel())
            group_coeff = self.coeff_cache[self.coeff_idx[i].item()]
            group_coeff_sel = coeff_sel[group_start:group_end]
            group_all_points = self._get_all_points(group_coeff)
            group_data = group_all_points[group_coeff_sel]
            decoded_tensor.flatten()[group_start:group_end] = group_data
            
        # Truncate the tensor to the original shape
        if self.original_data_shape != self.padded_data_shape:
            slices = tuple(slice(0, s) for s in self.original_data_shape)
            decoded_tensor = decoded_tensor[slices]
        
        return decoded_tensor
    
    @torch.inference_mode()
    def cuda_mat_mat_t_mul(self, rhs) -> torch.Tensor:
        if not isinstance(rhs, sbvr):
            raise ValueError(r_str("The RHS SBVR object is not valid."))
        if len(self.original_data_shape) != 2:
            raise ValueError(r_str("The LHS SBVR object is not a matrix."))
        if len(rhs.original_data_shape) != 2:
            raise ValueError(r_str("The RHS SBVR object is not a matrix."))
        if self.cgroup_len != rhs.cgroup_len:
            raise ValueError(r_str("Incompatible SBVR coeff group len: ") +
                             f"LHS SBVR group length: {self.cgroup_len}, "
                             f"RHS SBVR group length: {rhs.cgroup_len}")
        if self.padded_data_shape[1] != rhs.padded_data_shape[1]:
            raise ValueError(r_str("Incompatible matrix and vector shapes: ") +
                             f"LHS SBVR shape: {self.original_data_shape}, "
                             f"RHS SBVR shape: {rhs.original_data_shape}")
        if self.compute_dtype != torch.float16 or \
              rhs.compute_dtype != torch.float16:
            raise ValueError(r_str("Incompatible SBVR compute data types: ") +
                             f"LHS SBVR dtype: {self.compute_dtype}, "
                             f"RHS SBVR dtype: {rhs.compute_dtype}")
        if self.bvr_dtype != torch.uint32 or rhs.bvr_dtype != torch.uint32:
            raise ValueError(r_str("Incompatible SBVR vector data types: ") +
                             f"LHS BVR dtype: {self.bvr_dtype}, "
                             f"RHS BVR dtype: {rhs.bvr_dtype}")

        l_bvr = self.bvr
        l_coeff_idx = self.coeff_idx # [num_cgroups]
        l_coeff_cache = self.coeff_cache # [cache_lines, num_sums]
        
        r_bvr = rhs.bvr
        r_coeff_idx = rhs.coeff_idx # [num_cgroups]
        r_coeff_cache = rhs.coeff_cache # [cache_lines, num_sums]
        
        return sbvr_mm(l_bvr,
                       l_coeff_idx,
                       l_bias_idx,
                       l_coeff_cache,
                       l_bias_cache,
                       r_bvr,
                       r_coeff_idx,
                       r_bias_idx,
                       r_coeff_cache,
                       r_bias_cache)

    def get_sbvr_info(self):
        info_str = b_str("SBVR Info:") + \
        y_str("\n\tOriginal Data Type: ") + str(self.original_dtype) + \
        y_str("\n\tOriginal Data Shape: ") + str(self.original_data_shape) + \
        y_str("\n\tNumber of Summations: ") + str(self.num_sums) + \
        y_str("\n\tCoefficient Group Size: ") + str(self.cgroup_len) + \
        y_str("\n\tCoefficient Data Type: ") + str(self.coeff_dtype) + \
        y_str("\n\tBinary Vector Data Type: ") + str(self.bvr_dtype) + \
        y_str("\n\tCoefficient Tensor Size: ") + str(self.coeff.shape)
        return info_str
    
def save_sbvr(sbvr_obj, filename):
    if not isinstance(sbvr_obj, sbvr):
        raise ValueError("The object is not a valid SBVR object.")
    save_dict = {
        "num_sums": sbvr_obj.num_sums,
        "bvr_num_bits": sbvr_obj.bvr_num_bits,
        "cgroup_len": sbvr_obj.cgroup_len,
        
        "original_data_shape": sbvr_obj.original_data_shape,
        "padded_data_shape": sbvr_obj.padded_data_shape,
        "original_dtype": sbvr_obj.original_dtype,
        "bvr_dtype": sbvr_obj.bvr_dtype,
        "compute_dtype": sbvr_obj.compute_dtype,
        
        "coeff_cache": sbvr_obj.coeff_cache,
        "coeff_idx": sbvr_obj.coeff_idx,
        "bvr": sbvr_obj.bvr,
    }
    torch.save(save_dict, filename)
        
def load_sbvr(filename) -> sbvr:
    save_dict = torch.load(filename)
    sbvr_obj = sbvr(load_only=True, **save_dict)
    return sbvr_obj