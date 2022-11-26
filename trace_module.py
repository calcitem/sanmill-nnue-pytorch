from model import Module
import torch

if __name__ == "__main__":
    mynet = Module()
    best_modules_path = "work_dir/20221114_231816/best_modules/best-20221114_231816.pt"
    mynet.load_state_dict(torch.load(best_modules_path))
    mynet.trace_module()