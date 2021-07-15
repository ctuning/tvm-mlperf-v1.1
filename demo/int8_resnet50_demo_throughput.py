import os
import time
import multiprocessing
import numpy as np
from PIL import Image

import tvm
from tvm.contrib.graph_executor import GraphModule


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def get_img(img_path, batch=1):
    img = Image.open(img_path)
    in_data = np.asarray(img)
    in_data = in_data.transpose((2, 0, 1))
    shape = in_data.shape
    in_data = preprocess(in_data)
    in_data = np.broadcast_to(in_data.astype("float32"), shape=(batch, *shape))
    return in_data


def load_module(module_path, img_path):
    dev = tvm.cpu()
    lib = tvm.runtime.load_module(module_path)
    g_module = GraphModule(lib["default"](dev))
    in_data = get_img(img_path)
    g_module.set_input(0, in_data)
    return g_module


def run_module(g_module):
    g_module.run()


def arena_wrp(init_f, action_f, args, arena_size, arena_idx, still_running, init_barrier, res_count, res_time):
    idx_start = arena_size * arena_idx
    arena_size = min(multiprocessing.cpu_count() - idx_start, arena_size)
    idx_end = idx_start + arena_size

    # OMP_PLACES="{N},{N+1},{N+2},...,{N+SZ}"
    arena_places_str = "{" + "},{".join(str(i) for i in range(idx_start, idx_end)) + "}"

    os.environ['OMP_NUM_THREADS'] = str(arena_size)
    os.environ['OMP_PLACES'] = arena_places_str  # "cores"
    os.environ['OMP_PROC_BIND'] = "true"

    workload_ctx = init_f(*args)
    init_barrier.wait()

    tot_count = 0
    start_timestamp = time.time()
    while still_running.value == 1:
        tot_count += 1
        action_f(workload_ctx)

    tot_duration = time.time() - start_timestamp

    # If you are interested in distribution of efficiency between arenas uncomment this line
    # print(tot_count, tot_duration)

    res_count[arena_idx] = tot_count
    res_time[arena_idx] = tot_duration


def run_within_arenas(init_f, action_f, args, *, arena_size=1, arena_num=None, benchmark_time_sec=10):
    if arena_num is None:
        arena_num = (multiprocessing.cpu_count() - 1) // arena_size + 1

    still_running = multiprocessing.Value("i", 1)
    res_count = multiprocessing.Array("i", [0] * arena_num)
    res_time = multiprocessing.Array("d", [0.0] * arena_num)
    init_barrier = multiprocessing.Barrier(arena_num + 1)

    processes = []
    for idx in range(arena_num):
        p = multiprocessing.Process(target=arena_wrp,
                                    args=[init_f, action_f, args, arena_size, idx, still_running, init_barrier,
                                          res_count, res_time])
        p.start()
        processes.append(p)

    init_barrier.wait()
    time.sleep(benchmark_time_sec)
    still_running.value = 0

    for p in processes:
        p.join()
        assert p.exitcode == 0

    total_count = 0
    total_duration = 0.0
    for count, duration in zip(res_count, res_time):
        total_count += count
        total_duration += duration
    total_duration /= total_count

    latency = total_duration * 1000
    throughput = benchmark_time_sec * 1000 / total_count
    print(f"Arena SZ:{arena_size}, NUM:{arena_num}, L:{latency:.2f}, THRP:{throughput:.2f}")
    return latency, throughput


def print_to_csv(file_path, res):
    file = open(file_path, 'w')
    for el in res:
        print(f"{el[0]}; {el[1][0]:.2f}; {el[1][1]:.2f}", file=file)


def main():
    module_path = "__prebuilt/dnnl_int8_resnet50.so"
    img_path = "__data/cat3.png"

    # One trial time
    benchmark_time_sec = 3
    chill_out_time_sec = 1

    num_cores = multiprocessing.cpu_count()

    latency_mod_res = []
    throughput_mod_res = []

    # Latency mode evaluation. Single subprocess.
    print("=== Latency mode ===")
    for arena_size in range(1, num_cores + 1):
        res = run_within_arenas(init_f=load_module, action_f=run_module, args=(module_path, img_path),
                                arena_size=arena_size, arena_num=1, benchmark_time_sec=benchmark_time_sec)
        latency_mod_res.append((arena_size, res))
        time.sleep(chill_out_time_sec)
    print_to_csv("latency.csv", latency_mod_res)

    # Throughput mode evaluation. Several subprocess with equal num of internal threads(arena_size).
    # Lets limit it by 30 because single Resnet50 cannot utilize more cores.
    print("=== Throughput mode ===")
    for arena_size in range(1, min(31, num_cores)):
        res = run_within_arenas(init_f=load_module, action_f=run_module, args=(module_path, img_path),
                                arena_size=arena_size, arena_num=None, benchmark_time_sec=benchmark_time_sec)
        throughput_mod_res.append((arena_size, res))
        time.sleep(chill_out_time_sec)
    print_to_csv("throughput.csv", throughput_mod_res)


if __name__ == "__main__":
    main()
