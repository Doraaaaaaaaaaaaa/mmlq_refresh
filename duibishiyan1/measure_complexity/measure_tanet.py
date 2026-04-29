from __future__ import annotations

import torch
import sys
import types

from common import BASELINES, count_params, maybe_profile_macs, prepend_sys_path, print_result, pushd


TANET_DIR = BASELINES / "TANet-image-aesthetics-and-quality-assessment-main" / "code" / "AVA"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with prepend_sys_path(TANET_DIR), pushd(TANET_DIR):
        # TANet imports NNI at module import time, but model construction does
        # not need NNI. Provide a tiny shim so complexity measurement does not
        # require installing the full NNI package.
        if "nni" not in sys.modules:
            nni = types.ModuleType("nni")
            nni.get_next_parameter = lambda: {}
            nni.report_intermediate_result = lambda *args, **kwargs: None
            nni.report_final_result = lambda *args, **kwargs: None
            nni_utils = types.ModuleType("nni.utils")
            nni_utils.merge_parameter = lambda opt, tuner_params: opt
            sys.modules["nni"] = nni
            sys.modules["nni.utils"] = nni_utils
        if "tensorboardX" not in sys.modules:
            tbx = types.ModuleType("tensorboardX")
            tbx.SummaryWriter = lambda *args, **kwargs: None
            sys.modules["tensorboardX"] = tbx

        import torch.nn.functional as F

        import train_nni
        from train_nni import TANet

        def stable_targetnet_forward(self, x, paras):
            q = self.fc1(x)
            q = self.bn1(q)
            q = self.relu1(q)
            q = self.drop1(q)
            q = F.linear(q, paras["res_last_out_w"], paras["res_last_out_b"])
            # The released code creates a fresh BatchNorm1d inside forward,
            # which is not part of the registered model and breaks batch=1
            # profiling. Skipping it keeps registered params unchanged and
            # lets thop trace the actual model modules.
            q = self.relu7(q)
            return q

        train_nni.TargetNet.forward = stable_targetnet_forward

        model = TANet().to(device).eval()
        dummy = torch.randn(1, 3, 224, 224, device=device)

        total, trainable = count_params(model)
        gmacs, note = maybe_profile_macs(model, (dummy,))
        print_result("TANet", total, trainable, gmacs, note)


if __name__ == "__main__":
    main()
