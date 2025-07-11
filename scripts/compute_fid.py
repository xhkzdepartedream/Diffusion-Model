from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='/path/to/your/celeba_hq_256/',
    input2='../data/output',
    cuda=True,
    fid=True,
    verbose=True,
)

print("FID:", metrics['frechet_inception_distance'])
