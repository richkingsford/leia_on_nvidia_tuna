"""Small TensorRT runtime wrapper for Leia's current brick YOLO model.

The repo's brick detector still owns pre/post-processing. This helper runs the
required Jetson-native TensorRT engine.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

import numpy as np


class TensorRTYoloError(RuntimeError):
    """Raised when TensorRT/CUDA runtime setup or inference fails."""


class _CudaRuntime:
    _CUDA_SUCCESS = 0
    _CUDA_MEMCPY_HOST_TO_DEVICE = 1
    _CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self):
        lib_name = ctypes.util.find_library("cudart") or "libcudart.so"
        try:
            self._lib = ctypes.CDLL(lib_name)
        except OSError as exc:
            raise TensorRTYoloError(f"Unable to load CUDA runtime: {exc}") from exc

        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int
        self._lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, status: int, op: str) -> None:
        if int(status) == self._CUDA_SUCCESS:
            return
        msg = self._lib.cudaGetErrorString(int(status))
        text = msg.decode("utf-8", errors="replace") if msg else f"code {status}"
        raise TensorRTYoloError(f"CUDA {op} failed: {text}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))), "malloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p | None) -> None:
        if ptr and ptr.value:
            self._check(self._lib.cudaFree(ptr), "free")

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(stream)), "stream create")
        return stream

    def stream_destroy(self, stream: ctypes.c_void_p | None) -> None:
        if stream and stream.value:
            self._check(self._lib.cudaStreamDestroy(stream), "stream destroy")

    def memcpy_host_to_device_async(self, dst: ctypes.c_void_p, src: np.ndarray, stream: ctypes.c_void_p) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(
                dst,
                src.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(src.nbytes),
                self._CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "memcpy host-to-device",
        )

    def memcpy_device_to_host_async(self, dst: np.ndarray, src: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(
                dst.ctypes.data_as(ctypes.c_void_p),
                src,
                ctypes.c_size_t(dst.nbytes),
                self._CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "memcpy device-to-host",
        )

    def stream_synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self._lib.cudaStreamSynchronize(stream), "stream synchronize")


def _trt_dtype_to_numpy(dtype):
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return np.float32
    if dtype == trt.DataType.HALF:
        return np.float16
    if dtype == trt.DataType.INT8:
        return np.int8
    if dtype == trt.DataType.INT32:
        return np.int32
    if dtype == trt.DataType.BOOL:
        return np.bool_
    raise TensorRTYoloError(f"Unsupported TensorRT tensor dtype: {dtype}")


class TensorRTYoloEngine:
    """Run a static-shape YOLO TensorRT engine from numpy input blobs."""

    def __init__(self, engine_path: str | Path):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise TensorRTYoloError(f"TensorRT engine not found: {self.engine_path}")

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise TensorRTYoloError(f"TensorRT Python module is unavailable: {exc}") from exc

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise TensorRTYoloError(f"Unable to deserialize TensorRT engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise TensorRTYoloError("Unable to create TensorRT execution context")

        self.input_name = None
        self.output_name = None
        for idx in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(idx)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if not self.input_name or not self.output_name:
            raise TensorRTYoloError("TensorRT engine must have one input and one output tensor")

        self.input_shape = tuple(int(v) for v in self._engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(int(v) for v in self._engine.get_tensor_shape(self.output_name))
        if any(v <= 0 for v in self.input_shape + self.output_shape):
            raise TensorRTYoloError(
                "Dynamic TensorRT shapes are not supported by this lightweight runtime"
            )

        self.input_dtype = _trt_dtype_to_numpy(self._engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _trt_dtype_to_numpy(self._engine.get_tensor_dtype(self.output_name))
        self._input_host = np.empty(self.input_shape, dtype=self.input_dtype)
        self._output_host = np.empty(self.output_shape, dtype=self.output_dtype)
        self._cuda = _CudaRuntime()
        self._stream = self._cuda.stream_create()
        self._input_device = self._cuda.malloc(self._input_host.nbytes)
        self._output_device = self._cuda.malloc(self._output_host.nbytes)
        self._context.set_tensor_address(self.input_name, int(self._input_device.value))
        self._context.set_tensor_address(self.output_name, int(self._output_device.value))

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run inference and return a copy of the engine output tensor."""
        arr = np.asarray(blob, dtype=self.input_dtype)
        if tuple(arr.shape) != self.input_shape:
            raise TensorRTYoloError(
                f"Expected input shape {self.input_shape}, got {tuple(arr.shape)}"
            )
        self._input_host[...] = np.ascontiguousarray(arr)
        self._cuda.memcpy_host_to_device_async(self._input_device, self._input_host, self._stream)
        ok = self._context.execute_async_v3(stream_handle=int(self._stream.value))
        if not ok:
            raise TensorRTYoloError("TensorRT execute_async_v3 returned false")
        self._cuda.memcpy_device_to_host_async(self._output_host, self._output_device, self._stream)
        self._cuda.stream_synchronize(self._stream)
        return self._output_host.copy()

    def close(self) -> None:
        cuda = getattr(self, "_cuda", None)
        if cuda is None:
            return
        self._cuda = None
        try:
            cuda.free(getattr(self, "_input_device", None))
        finally:
            self._input_device = None
        try:
            cuda.free(getattr(self, "_output_device", None))
        finally:
            self._output_device = None
        try:
            cuda.stream_destroy(getattr(self, "_stream", None))
        finally:
            self._stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
