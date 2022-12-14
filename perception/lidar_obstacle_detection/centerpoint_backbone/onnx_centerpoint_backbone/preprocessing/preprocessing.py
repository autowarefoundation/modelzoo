import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class PreprocessingModule:
    @T.prim_func
    def scatter(pillar_features: T.Buffer[(40000, 1, 32), "float32"],
                coords: T.Buffer[(40000, 3), "int32"],
                spatial_features: T.Buffer[(1, 32, 560, 560), "float32"]):
        # coord = T.alloc_buffer([3], "int32")
        T.func_attr({"global_symbol": "scatter", "tir.noalias": True})
        for i, j, k in T.grid(32, 560, 560):
            with T.block("spatial_features"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                spatial_features[0, vi, vj, vk] = T.float32(0)
        for i, j in T.grid(40000, 32):
            with T.block("spatial_features"):
                vi, vj = T.axis.remap("SS", [i, j])
                if(coords[vi,0] >= 0):
                    spatial_features[0, vj, coords[vi,1], coords[vi,2]] = pillar_features[vi, 0, vj]
