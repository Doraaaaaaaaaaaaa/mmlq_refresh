import torch
ckpt = torch.load(r'C:\Users\admin\.cache\torch\hub\checkpoints\clip_vit_L.pth', map_location='cpu')
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print('Top-level keys:', keys[:10])
    # 看第一个值的类型
    first_val = ckpt[keys[0]]
    print('First value type:', type(first_val))
    if isinstance(first_val, dict):
        print('Sub-keys:', list(first_val.keys())[:10])
    if isinstance(first_val, torch.Tensor):
        print('First tensor shape:', first_val.shape)
else:
    print('Type:', type(ckpt))
    if hasattr(ckpt, 'state_dict'):
        sd = ckpt.state_dict()
        print('State dict keys[:10]:', list(sd.keys())[:10])

# 直接看所有顶层 key
print('\nAll top-level keys:')
if isinstance(ckpt, dict):
    for k in list(ckpt.keys())[:30]:
        v = ckpt[k]
        shape = v.shape if isinstance(v, torch.Tensor) else type(v)
        print(f'  {k}: {shape}')
