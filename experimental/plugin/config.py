import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args, unparsed = parser.parse_known_args()
    return args

def parse_config():
    args = parse_args()
    with open(args.config_path) as f:
        config = eval(f.read())
    return config
    
    #comp config
#    "TorchTrainer": {
#        train_loop_config={},
#        scaling_config=ScalingConfig(
#            num_workers:args.num_workers,
#            resources_per_worker={"CPU": 56},
#            placement_strategy="SPREAD"
#        ),
#        torch_config=torch_config,
#        run_config = RunConfig(
#            local_dir=args.ray_results_path,
#            failure_config=FailureConfig(
#                max_failures=args.ray_fault_tolerance
#            )
#        )
#    }
#torch_config = TorchConfig(backend="ccl")