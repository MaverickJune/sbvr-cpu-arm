import torch
import itertools
import math

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
    
def print_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    errors = tensor1 - tensor2
    mse = torch.mean(errors ** 2).item()
    max_error = torch.max(errors).item()
    min_error = torch.min(errors).item()
    std_dev = torch.std(errors).item()
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
                 coeff_group_size: int = 256,
                 coeff_dtype: torch.dtype = None,
                 bin_vec_dtype: torch.dtype = None):
        if data is None:
            raise ValueError(r_str("Data cannot be None"))
            
        self.num_sums = num_sums
        self.coeff_group_size = coeff_group_size
        self.coeff_dtype = data.dtype if coeff_dtype is None else coeff_dtype
        self.bin_vec_dtype = \
            torch.uint32 if bin_vec_dtype is None else bin_vec_dtype
        
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
        self.__encode_to_sbvr(data)
        
    def __get_all_points(self, coeff_bias: torch.tensor, coeff: torch.tensor):
        num_coeff = coeff.shape[0]
        bin_combs = torch.tensor(
            list(itertools.product([0, 1], repeat=num_coeff)),
            dtype=coeff.dtype, device=coeff.device
        )
        return bin_combs @ coeff + coeff_bias
    
    def __map_data_to_coeff(self, 
                            data: torch.tensor, 
                            coeff_bias: torch.tensor, 
                            coeff: torch.tensor):
        all_points = self.__get_all_points(coeff_bias, coeff)
        diff = torch.abs(data.unsqueeze(1) - all_points)
        coeff_idx = torch.argmin(diff, dim=1)
        return all_points[coeff_idx], coeff_idx, all_points
        
    def __return_coeff_mse(self, data, coeff_bias, coeff):
        predicted, coeff_idx, all_points = \
            self.__map_data_to_coeff(data, coeff_bias, coeff)
        errors = data - predicted
        mse = torch.mean(errors ** 2).item()
        return mse, predicted, coeff_idx, all_points
        
    def __encode_data(self, data):
        data_max = torch.max(data)
        data_avg = torch.mean(data)
        data_min = torch.min(data)
        data_97 = torch.quantile(data, 0.97)

        min_mse = 1e10
        r_max = math.pi*2/3 + 0.1
        r_min = 1.0
        r_gran = (r_max - r_min) / 64
        b_max = data_avg
        b_min = data_min - (abs(data_min)*0.1)
        b_gran = (b_max - b_min) / 32
        s_max = (data_max - data_min) * 1.1
        s_min = (data_97 - data_avg) * 2
        s_gran = (s_max - s_min) / 32
        print(g_str("\tData max: ") + f"{data_max:.4e}" +
              ", " + g_str("avg: ") + f"{data_avg:.4e}" +
              ", " + g_str("min: ") + f"{data_min:.4e}")
        print(r_str("\tR search range: ") + f"{r_min:.4e} to {r_max:.4e}, " +
              r_str("search granularity: ") + f"{r_gran:.4e}")
        print(r_str("\tBias search range: ") + f"{b_min:.4e} to {b_max:.4e}, " +
              r_str("search granularity: ") + f"{b_gran:.4e}")
        print(r_str("\tScale search range: ") + f"{s_min:.4e} to {s_max:.4e}, "
              + r_str("search granularity: ") + f"{s_gran:.4e}")
        
        for r in torch.arange(r_min + r_gran, r_max, r_gran):
            coeff = torch.tensor([r**i for i in range(self.num_sums)],
                dtype=data.dtype, device=data.device)
            coeff_sum = torch.sum(coeff)
            coeff_norm = coeff / coeff_sum
            for scale in torch.arange(s_min, s_max + s_gran, s_gran):
                coeff = coeff_norm * scale
                for coeff_bias in torch.arange(b_min, b_max + b_gran, b_gran):
                    coeff_bias = coeff_bias.to(data.dtype)
                    mse, predicted, coeff_idx, all_points = \
                        self.__return_coeff_mse(data, coeff_bias, coeff)
                    if mse < min_mse:
                        min_mse = mse
                        best_bias = coeff_bias
                        best_coeff = coeff
                        best_predicted = predicted
                        best_coeff_idx = coeff_idx
                        best_all_points = all_points
                        best_r = r
                        best_s = scale
        print (g_str("Best MSE: ") + f"{min_mse:.4e}" +
               ", " + g_str("Coeff: ") + str(best_coeff) +
                ", " + g_str("(r, b, s): ") + f"{best_r:.4e}, " +
                f"{best_bias.item():.4e}, {best_s:.4e}" +
                ", " + g_str("Num points: ") + 
                f"{torch.unique(all_points).numel()}")
        print(y_str("all_points: ") + str(torch.sort(best_all_points)[0]) +
              y_str("\nOriginal Data: ") + str(data) +
              y_str("\nMapped Data: ") + str(best_predicted))

        return best_bias, best_coeff, best_coeff_idx
        
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
            group_start = i * self.coeff_group_size
            group_end = \
                min(group_start + self.coeff_group_size, data_num)
            group_data = data.flatten()[group_start:group_end].to(torch.float64)
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


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mat_size = (16, 16)

mat_a = torch.randn(mat_size, dtype=torch.float64).to(device)
mat_b = torch.randn(mat_size, dtype=torch.float64).to(device)
print_tensor(mat_a, "mat_a")
print_tensor(mat_b, "mat_b")

mat_c_64 = f64_matmul(mat_a, mat_b)
mat_c_32 = f64_matmul(mat_a.to(torch.float32), mat_b.to(torch.float32))
mat_c_16 = f64_matmul(mat_a.to(torch.float16), mat_b.to(torch.float16))
mat_c_bf16 = f64_matmul(mat_a.to(torch.bfloat16), mat_b.to(torch.bfloat16))
mat_c_e4m3fn = f64_matmul(mat_a.to(torch.float8_e4m3fn), mat_b.to(torch.float8_e4m3fn))
mat_c_e5m2 = f64_matmul(mat_a.to(torch.float8_e5m2), mat_b.to(torch.float8_e5m2))
mat_c_sbvr_10 = f64_matmul(sbvr(mat_a, num_sums=10).get_decoded_tensor(), 
                        sbvr(mat_b, num_sums=10).get_decoded_tensor())
mat_c_sbvr_8 = f64_matmul(sbvr(mat_a, num_sums=8).get_decoded_tensor(), 
                        sbvr(mat_b, num_sums=8).get_decoded_tensor())
mat_c_sbvr_6 = f64_matmul(sbvr(mat_a, num_sums=6).get_decoded_tensor(), 
                        sbvr(mat_b, num_sums=6).get_decoded_tensor())
mat_c_sbvr_4 = f64_matmul(sbvr(mat_a, num_sums=4).get_decoded_tensor(), 
                        sbvr(mat_b, num_sums=4).get_decoded_tensor())
mat_c_sbvr_2 = f64_matmul(sbvr(mat_a, num_sums=2).get_decoded_tensor(), 
                        sbvr(mat_b, num_sums=2).get_decoded_tensor())

print_tensor(mat_c_64, "mat_c")
print_tensor(mat_c_16, "mat_c_16")
print_tensor(mat_c_bf16, "mat_c_bf16")
print_tensor(mat_c_e4m3fn, "mat_c_e4m3fn")
print_tensor(mat_c_e5m2, "mat_c_e5m2")
print_tensor(mat_c_sbvr_10, "mat_c_sbvr_10")
print_tensor(mat_c_sbvr_8, "mat_c_sbvr_8")
print_tensor(mat_c_sbvr_6, "mat_c_sbvr_6")
print_tensor(mat_c_sbvr_4, "mat_c_sbvr_4")
print_tensor(mat_c_sbvr_2, "mat_c_sbvr_2")

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
print(b_str("Case 6: Conversion to sbvr 10 bit"))
print_errors(mat_c_64, mat_c_sbvr_10)
print(b_str("Case 7: Conversion to sbvr 8 bit"))
print_errors(mat_c_64, mat_c_sbvr_8)
print(b_str("Case 8: Conversion to sbvr 6 bit"))
print_errors(mat_c_64, mat_c_sbvr_6)
print(b_str("Case 9: Conversion to sbvr 4 bit"))
print_errors(mat_c_64, mat_c_sbvr_4)
print(b_str("Case 10: Conversion to sbvr 2 bit"))
print_errors(mat_c_64, mat_c_sbvr_2)
