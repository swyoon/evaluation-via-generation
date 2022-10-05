import glob
import os
import sys

import torch


class AutoGPUAllocation:
    def __init__(self, lock_root="."):
        n_gpu = torch.cuda.device_count()

        set_used = set(
            [
                int(f.split(".")[-1])
                for f in glob.glob(os.path.join(lock_root, ".gpu.lock.*"))
            ]
        )
        set_all = set(range(n_gpu))
        set_avail = set_all.difference(set_used)
        if len(set_avail) == 0:
            print("no available GPU. exiting...")
            sys.exit()
        self.device_no = list(set_avail)[0]
        self.device = f"cuda:{self.device_no}"

        self.lockfile = os.path.join(lock_root, f".gpu.lock.{self.device_no}")
        open(self.lockfile, "a").close()  # create empty file

        print("used", set_used)
        print("avail", set_avail)
        print("lockfile", self.lockfile)

    def __del__(self):
        if not hasattr(self, "lockfile"):
            pass
        else:
            os.remove(self.lockfile)
            print(f"delete {self.lockfile}")
