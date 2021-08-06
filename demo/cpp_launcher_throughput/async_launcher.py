import threading
import tvm
import numpy


class AsyncGraphExecutorImpl:
    """
    TBD
    """
    def __init__(self, mod, idx):
        self._mod = mod
        self._infer = mod["infer"]
        self._init = mod["init"]
        self._init(idx)

    def infer(self, input_data):
        return self._infer(input_data)


class AsyncGraphExecutor:
    """
    TVM Graph Executor wrapper with additional capabilities like:
      - Multi threading inference. Method infer may be called form any
        num of threads.
      - Implicit batch slicing. In case originally compiled model has
        batch different from provided as input it will be automatically
        sliced by chunk with supported batch size (remaining part of chunk
        tensor is zeroed).
      - Allow to apply affinity rules to workers.
    """

    # Lazy initialization
    _async_launcher_module = None
    _async_launcher_factory = None

    @classmethod
    def async_launcher_factory(cls):
        if cls._async_launcher_module is None:
            search_path = "/Users/apeskov/git/tvm/cmake-build-debug-no-omp/demo/cpp_launcher_throughput/"
            cls._async_launcher_module = tvm.runtime.load_module(search_path + "libasync_launcher.dylib")
            cls._async_launcher_factory = cls._async_launcher_module["CreateAsyncLauncher"]

        return cls._async_launcher_factory

    def __init__(self, path, affinity_policy="default"):
        """
        Constructor
        :param path:
        """
        self._lib_path = path
        self._lib_mod = tvm.runtime.load_module(path)
        self._impl_holder = threading.local()

        self._policy = affinity_policy

        # Worker thread safe counting
        self._workers_count = 0
        self._workers_count_lock = threading.Lock()

        self._numa_node_size = 1
        self._numa_nodes = [None] * 16

    def get_next_worker_idx(self):
        with self._workers_count_lock:
            res = self._workers_count
            self._workers_count += 1
        return res

    def get_lib_mod(self, worker_id):
        if self._policy == "default":
            return self._lib_mod
        if self._policy == "numa":
            numa_node_idx = worker_id // self._numa_node_size
            if self._numa_nodes[numa_node_idx] is None:
                self._numa_nodes[numa_node_idx] = tvm.runtime.load_module(self._lib_path)

            return self._numa_nodes[numa_node_idx]

    def initialize_for_thread(self):
        """
        Initialize async runner to use current thread

        Should be called before using this runner from that thread
        """
        if hasattr(self._impl_holder, "impl"):
            return

        executor_id = self.get_next_worker_idx()
        lib_mode = self.get_lib_mod(executor_id)
        launcher_mod = AsyncGraphExecutor.async_launcher_factory()(lib_mode)
        impl = AsyncGraphExecutorImpl(launcher_mod, executor_id)

        self._impl_holder.impl = impl

    def infer(self, input_data):
        """

        Perform inference of particular input data

        :param input_data: list(numpy.array)
        :return: list(numpy.array)
        """

        if not hasattr(self._impl_holder, "impl"):
            self.initialize_for_thread()

        assert len(input_data) == 1

        input_tensor = input_data[0]
        assert isinstance(input_tensor, (numpy.ndarray, tvm.nd.NDArray))

        if isinstance(input_tensor, numpy.ndarray):
            input_tensor = tvm.runtime.ndarray.array(input_tensor)

        return self._impl_holder.impl.infer(input_tensor).numpy()

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1