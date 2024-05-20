import ctypes
from ctypes import (
    c_void_p,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_float,
    c_bool,
    POINTER,
)
import pathlib

_base_path = pathlib.Path(__file__).parent.parent.parent.resolve()
# if _base_path.name != "site-packages": # editable mode
#     _base_path = _base_path.parent.resolve()
_base_path = f"{str(_base_path)}/inference_engine"

def load_engine_lib(model_type: str):
    native_lib = None
    if model_type == "llama" or model_type == "llama2":
        native_lib = ctypes.CDLL(f"{_base_path}/libllama_vllm_cb_cpp.so")
    if native_lib:
        native_lib.create_new_model.argtypes = []
        native_lib.create_new_model.restype = c_void_p

        native_lib.destroy_model.argtypes = [c_void_p]
        native_lib.destroy_model.restype = None

        native_lib.init_model.argtypes = [c_void_p, c_char_p, c_int, c_int, c_int, c_int32, c_char_p, c_float, c_int, c_int]
        native_lib.init_model.restype = c_bool

        native_lib.quantize_model.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_int, c_char_p, c_char_p, c_bool, c_int]
        native_lib.quantize_model.restype = c_int

        native_lib.generate.argtypes = [c_void_p, c_void_p, c_void_p, c_bool, c_void_p, c_void_p, POINTER(c_int), c_int]
        native_lib.generate.restype = c_void_p

        native_lib.set_block_size.argtypes = [c_void_p, c_int64]
        native_lib.set_block_size.restype = None

        native_lib.free_slots.argtypes = [c_void_p, POINTER(c_int64), c_int]
        native_lib.free_slots.restype = c_bool

        native_lib.set_kv_caches_ptr.argtypes = [c_void_p, c_void_p]
        native_lib.set_kv_caches_ptr.restype = None

        native_lib.get_last_error.argtypes = [c_void_p]
        native_lib.get_last_error.restype = c_char_p
    return native_lib