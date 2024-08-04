import argparse
import os, json
from datetime import datetime
from typing import Any
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import _ExperimentalConfig, ExecutionTraceObserver
from train.compute.python.lib.init_helper import init_logging
import logging

def trace_handler(prof)-> Any:  
    kineto_file = "./data/worker"+str(dist.get_rank())+"_step_"+str(prof.step_num)
    prof.export_chrome_trace(kineto_file)

def run(local_rank, tensor_size=(256, 256)):
    tensor_a = torch.randn(tensor_size, device=torch.device('cuda', local_rank))
    tensor_b = torch.randn(tensor_size, device=torch.device('cuda', local_rank))
    result = torch.empty(tensor_size, device=torch.device('cuda', local_rank))
    dist.all_reduce(result)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Matrix Addition and AllReduce")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--output-prefix", type=str, default="./", help="Outout file path")
    args = parser.parse_args()

    pid = os.getpid()
    start_time = datetime.now()
    timestamp = int(datetime.timestamp(start_time))
    out_file_prefix = f"{args.output_prefix}_{pid}_{timestamp}"
    out_file_name = f"{out_file_prefix}.json"
    write_option = "a"

    logger = init_logging(getattr(logging, "DEBUG", logging.INFO))

    with open(out_file_name, write_option) as out_file:
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

        et_file = f"{out_file_prefix}_et.json"
        et = ExecutionTraceObserver()
        et.register_callback(et_file)
        logger.info(f"Exeution trace: {et_file}")

        cupti_profiler_config = (
            _ExperimentalConfig(
                profiler_metrics=["kineto__cuda_core_flops"],
                profiler_measure_per_kernel=False,
            )
        )

        # Profiler configuration
        with torch.autograd.profiler.profile(
            True,
            use_device="cuda",
            use_kineto=True,
            record_shapes=False,
            experimental_config=cupti_profiler_config,
            # use_cpu enables profiling and recodring of CPU pytorch operators.
            # This is useful in CUPTI profiler mode if we are measuring per GPU kernel metrics.
            use_cpu=True,
        ) as p:
            print("Running dummy profiler warmup for CUPTI.")

        with torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2
            ),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
            on_trace_ready=lambda prof: trace_handler(prof),
            execution_trace_observer=et,
        ) as prof:
            for epoch in range(20):
                run(local_rank)
                if epoch == 11:
                    et.stop()
                if epoch == 10:
                    et.start()
                prof.step()
        et.unregister_callback()
        trace_file = f"{out_file_prefix}_trace.json"
        logger.info(f"trace: {trace_file}")
        p.export_chrome_trace(trace_file)
        print(json.dumps({"trace_file": trace_file}), file=out_file)
        print(
            json.dumps({"finish_time": datetime.now().isoformat(timespec="seconds")}),
            file=out_file,
        )
        if local_rank == 0:
            print(p.key_averages().table(sort_by="cuda_time_total"))
            
    logger.info(f"Generated trace info: {out_file_name}")

if __name__ == "__main__":
    main()

