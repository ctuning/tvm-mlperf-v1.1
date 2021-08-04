import threading
import tvm
import numpy


class AsyncGraphExecutorImpl:
    """
    TBD
    """
    def __init__(self, mod):
        self._mod = mod
        self._infer = mod["infer"]
        self._init = mod["init"]
        self._init()

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

    def __init__(self, path):
        """
        Constructor
        :param path:
        """
        self._lib_path = path
        self._lib_mod = tvm.runtime.load_module(path)
        self._impl_holder = threading.local()

    def initialize_for_thread(self, affinity_policy="default"):
        """
        Initialize async runner to use current thread

        Should be called before using this runner from that thread
        """
        if hasattr(self._impl_holder, "impl"):
            return

        launcher_mod = AsyncGraphExecutor.async_launcher_factory()(self._lib_mod)
        impl = AsyncGraphExecutorImpl(launcher_mod)

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
        assert isinstance(input_data[0], numpy.ndarray)

        tvm_input = tvm.runtime.ndarray.array(input_data[0])

        return self._impl_holder.impl.infer(tvm_input).numpy()

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1