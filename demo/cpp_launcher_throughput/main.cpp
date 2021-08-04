#include <iostream>
#include <thread>
#include <chrono>

#include "tvm/runtime/module.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::microseconds ms;

void action(tvm::runtime::PackedFunc run, bool *still_running,
            size_t *time, size_t *count) {
  size_t loc_count = 0;
  auto start = Time::now();
  while (*still_running) {
    run();
    loc_count++;
  }
  *time = std::chrono::duration_cast<ms>(Time::now() - start).count();
  *count = loc_count;
}

std::thread create_worker(tvm::runtime::Module &mod_factory, DLDevice &dev, bool *still_running,
                          size_t *time, size_t *count) {
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");

  std::thread thr (action, run, still_running, time, count);

  return thr;
}

int main_check (tvm::runtime::Module mod_factory, int NUM, int TIME) {
//  std::cout << "Hello (" << NUM << ")" << std::endl;

  DLDevice dev {kDLCPU, 0};
  // create workers
  bool still_running = true;

  std::vector<std::thread> workers(NUM);
  std::vector<size_t> times(NUM);
  std::vector<size_t> counts(NUM);

  for (int i = 0; i < NUM; i++) {
    workers[i] = create_worker(mod_factory, dev, &still_running, &times[i], &counts[i]);
  }


  std::this_thread::sleep_for(std::chrono::seconds(TIME));
  still_running = false;
  for (auto &worker : workers)
    worker.join();

  size_t avg_TPF = 0;
  for (int i = 0; i < NUM; i++) {
    size_t time_per_frame = times[i] / counts[i];
//    std::cout << "  #"<< i << " Time per frame: " << time_per_frame << " us"<< std::endl;
    avg_TPF += time_per_frame;
  }
  avg_TPF /= NUM;
  std::cout << NUM << ";  " << avg_TPF << std::endl;

  return 0;
}


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "expected path to model as argument" << std::endl;
    return 1;
  }

  std::string model_path(argv[1]);
  int NUM = std::atoi(argv[2]);
  auto mod_factory = tvm::runtime::Module::LoadFromFile(model_path);

  for (int n = 1; n < NUM; n++) {
    main_check(mod_factory, n, 1);
  }
}
