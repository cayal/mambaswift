//
// Parameter sizes are given in (C=1|H|W) order, and buffers are serialized row-major.
//
//

#include <metal_stdlib>
#include "Arguments.h"
using namespace metal;


kernel void negexp(device float* data [[buffer(0)]],
                    uint tig [[thread_position_in_grid]]
                   ) {
    data[tig] = -exp(data[tig]);
}

kernel void softplus(device float* data [[buffer(0)]],
                     constant uint& rowWidth [[buffer(1)]],
                     uint2 tig [[thread_position_in_grid]]) {
    uint cur = (tig.y * rowWidth) + tig.x;
    data[cur] = log(1 + exp(data[cur]));
}

kernel void rowSquaredSums(constant OGScratchContextAB& context [[buffer(0)]],
                 constant OGHParams* hParams    [[buffer(1)]],
                 constant OGStateAB& state      [[buffer(2)]],
                 device   atomic_float* sums    [[buffer(3)]],
                 uint2 tig [[thread_position_in_grid]],
                 uint tisg [[thread_index_in_simdgroup]],
                 uint tpsg [[threads_per_simdgroup]]
                 ) {
    threadgroup float my_sum;
    float x = (float)context.ld1[tig.x + (hParams->dModel * tig.y)];
    my_sum = simd_prefix_inclusive_sum(x*x);
    if (tisg == (tpsg - 1)) {
        atomic_fetch_add_explicit(sums + tig.y, my_sum, memory_order_relaxed);
    }
}

kernel void rmsNormAndCopy(constant OGScratchContextAB& context [[buffer(0)]],
                           constant OGHParams* hParams    [[buffer(1)]],
                           constant OGStateAB& state      [[buffer(2)]],
                           constant float*     rowSums    [[buffer(3)]],
                           uint2 tig [[thread_position_in_grid]]
                           ) {
    int cur = tig.x + (hParams->dModel * tig.y);
    context.ld2[cur] = context.ld1[cur];
    float rowMean = rowSums[tig.y] / (float)hParams->dModel;
    float factor = rsqrt(rowMean + FLT_EPSILON);
    
    float product = factor
    * context.ld1[cur]
    * state.layers[0].norm[tig.x];
    
    context.ld1[cur] = product;
}

kernel void inProj(device   OGScratchContextAB& context [[buffer(0)]],
                   constant OGHParams* hParams    [[buffer(1)]],
                   constant OGStateAB& state      [[buffer(2)]],
                   device   atomic_float* sums    [[buffer(3)]],
                   constant uint& layer_number    [[buffer(4)]],
                   constant uint& L               [[buffer(5)]],
                   uint2 tig  [[thread_position_in_grid]],
                   uint2 tgpg [[threadgroups_per_grid]],
                   uint2 tptg [[threads_per_threadgroup]],
                   uint  tisg [[thread_index_in_simdgroup]],
                   uint  tpsg [[threads_per_simdgroup]]
                   ) {
    // dimensions out (x): L
    // dimensions out (y): dInProj
    uint tg_per_L = tgpg.x / L;
    uint threads_per_L = tptg.x * tg_per_L;
    
    // 0 <= x <= L
    // 0 <= y <= dInProj
    uint2 out_cursor;
    out_cursor.x = tig.x / threads_per_L;
    out_cursor.y = (L * tig.y);
    
    // 0 <= x <= dModel
    uint in_cursor_z = tig.x % threads_per_L;
    uint in_cursor_i = (hParams->dModel * tig.y);
    uint in_cursor_l = (hParams->dModel * out_cursor.x);
    
    uint matmul_cursor_ld1 = in_cursor_l + in_cursor_z;
    uint matmul_cursor_ipw = in_cursor_i + in_cursor_z;
    
    float ld1_val = (float)context.ld1[matmul_cursor_ld1];
    float ipw_val = (float)state.layers[layer_number].in_proj_weight[matmul_cursor_ipw];
    float x = ld1_val * ipw_val;
    
    float local_sum = simd_prefix_inclusive_sum(x);
    
    if (tisg == (tpsg - 1)) {
        atomic_fetch_add_explicit(sums + out_cursor.y + out_cursor.x, local_sum, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if(tig.x % threads_per_L == 0) {
        float wack = atomic_load_explicit(sums + out_cursor.y + out_cursor.x, memory_order_relaxed);
        
        context.dbld_in[out_cursor.x + out_cursor.y] = wack;
    }
}

kernel void dtProj(device   OGScratchContextAB& context [[buffer(0)]],
                   constant OGHParams* hParams    [[buffer(1)]],
                   constant OGStateAB& state      [[buffer(2)]],
                   constant uint& layer_number    [[buffer(3)]],
                   uint3 tig  [[thread_position_in_grid]],
                   uint3 tgpg [[threadgroups_per_grid]],
                   uint3 tptg [[threads_per_threadgroup]],
                   uint  titg [[thread_index_in_threadgroup]],
                   uint  tisg [[thread_index_in_simdgroup]],
                   uint  tpsg [[threads_per_simdgroup]],
                   uint  sitg [[simdgroup_index_in_threadgroup]],
                   uint  sgptg [[simdgroups_per_threadgroup]]
                   ) {
    // tig.z <-- dtRank --> ! buffer is shared, full stride is dXProj, don't dispatch more than dtRank threads deep
    // tig.y   <-- L -->
    // tig.x <-- dInner -->
    threadgroup float simd_sums[32];
    float w = state.layers[layer_number].dt_proj_weight[(tig.x * hParams->dtRank) + tig.z];
    float b = state.layers[layer_number].dt_proj_bias[tig.x];
    float x = context.debici[(tig.y * hParams->dXProj) + tig.z];
    simd_sums[sitg] = simd_sum(x * w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(titg == 0) {
        float total = 0.0;
        for(uint i=0; i < sgptg; i++) {
            total += simd_sums[i];
        }
        total = total + b;
        context.deli[(tig.y * hParams->dInner) + tig.x] = total;
    }
}

// todo clean up the chaos of the consequences of my confusion
kernel void outProj(device   OGScratchContextAB& context [[buffer(0)]],
                    constant OGHParams* hParams    [[buffer(1)]],
                    constant OGStateAB& state      [[buffer(2)]],
                    constant uint& layer_number    [[buffer(3)]],
                    constant uint& L               [[buffer(4)]],
                    constant uint& zOffset               [[buffer(5)]],
                    device atomic_float* simd_sums [[buffer(6)]],
                   uint3 tig  [[thread_position_in_grid]],
                   uint3 tgpg [[threadgroups_per_grid]],
                   uint3 tptg [[threads_per_threadgroup]],
                   uint  titg [[thread_index_in_threadgroup]],
                   uint  tisg [[thread_index_in_simdgroup]],
                   uint  tpsg [[threads_per_simdgroup]],
                   uint  sitg [[simdgroup_index_in_threadgroup]],
                   uint  sgptg [[simdgroups_per_threadgroup]]
                   ) {
    uint out_cur = (tig.y * hParams->dModel) + tig.x;
    uint zIndex = tig.z + zOffset;
    float w = state.layers[layer_number].out_proj_weight[(tig.x * hParams->dInner) + zIndex];
    float x = context.tricky[(tig.y * hParams->dInner) + zIndex];
    
        atomic_fetch_add_explicit(simd_sums + out_cur, x * w, memory_order_relaxed);
    //    simd_sums[sitg] = simd_sum(x * w);
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_barrier(mem_flags::mem_device);
    float total = atomic_load_explicit(simd_sums + out_cur, memory_order_relaxed);
    if(titg == 0 && tisg == 0 && tig.z == 0) {
//        float total = 0.0;
//        for(uint i=0; i < sgptg; i++) {
//              total += atomic_load_explicit(simd_sums + (out_cur * 32) + i, memory_order_relaxed);
////            total += simd_sums[i];
//        }
        float ld2 = context.ld2[(tig.y * hParams->dModel) + tig.x];
        context.ld1[out_cur] = total + ld2;
    }
}

kernel void ySiluRes(device   OGScratchContextAB& context [[buffer(0)]],
                    constant OGHParams* hParams    [[buffer(1)]],
                    constant OGStateAB& state      [[buffer(2)]],
                    constant uint& L               [[buffer(3)]],
                    uint2 tig  [[thread_position_in_grid]]
                    ) {
    uint resOffset = (hParams->dInner * L);
    float res = context.dbld_in[resOffset + (tig.x * L) + tig.y];
    float siluRes = res * (1/(1+exp(-res)));
    float y = context.tricky[tig.y * hParams->dInner + tig.x];
    context.tricky[tig.y * hParams->dInner + tig.x] = y * siluRes;
}

kernel void conv1d4dconv4(device   OGScratchContextAB& context [[buffer(0)]],
                          constant OGHParams* hParams    [[buffer(1)]],
                          constant OGStateAB& state      [[buffer(2)]],
                          constant uint& layer_number    [[buffer(3)]],
                          constant uint& L               [[buffer(4)]],
                          uint2 tig  [[thread_position_in_grid]],
                          uint tiqu  [[thread_index_in_quadgroup]]
                          ) {
    assert(hParams->dConv == 4); // This kernel only does its thing with dConv of 4
    uint out_x = tig.x / 4;
    uint out_cur = (tig.y * L) + out_x;
    uint conv_weight_cur = (tig.y * 4) + tiqu;
    float edge_mask = ((int)tiqu + ((int)out_x - 2)) > 0;
    
    float weight = state.layers[layer_number].conv1d_weight[conv_weight_cur];
                
    float val = context.deli[out_cur - (3 - tiqu)];
    
    float sum = quad_prefix_inclusive_sum(edge_mask * val * weight);
    
    if(tiqu == 3) {
        float c1d = sum + state.layers[layer_number].conv1d_bias[tig.y];
        float silu = c1d * (1/(1+exp(-c1d)));
        context.dbld_in[out_cur] = silu;
    }
}

kernel void xProjToDBC(device   OGScratchContextAB& context [[buffer(0)]],
                       constant OGHParams* hParams    [[buffer(1)]],
                       constant OGStateAB& state      [[buffer(2)]],
                       constant uint& layer_number    [[buffer(3)]],
                       constant uint& L               [[buffer(4)]],
                       constant uint& zIndexOffset    [[buffer(5)]],
                       device atomic_float* simd_sums        [[buffer(6)]],
                       device atomic_float* pls_debug_me [[buffer(7)]],
                       uint3 tig  [[thread_position_in_grid]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       uint3 tgpg [[threadgroups_per_grid]],
                       uint titg [[thread_index_in_threadgroup]],
                       uint  tisg [[thread_index_in_simdgroup]],
                       uint  tpsg [[threads_per_simdgroup]],
                       uint3  tptg [[threads_per_threadgroup]],
                       uint  sitg [[simdgroup_index_in_threadgroup]],
                       uint  sgptg [[simdgroups_per_threadgroup]]
                       ) {
    uint cur_out = (tig.y * hParams->dXProj) + tig.x;
    uint cur_dIn    = (tig.z + zIndexOffset);
    uint cur_dbld   = (cur_dIn * L) + (tig.y);
    ulong cur_xProjW = (tig.x * hParams->dInner) + cur_dIn;
    float x = context.dbld_in[cur_dbld];
    float weight = state.layers[layer_number].x_proj_weight[cur_xProjW];
    float val = x * weight;
    float local_sum = simd_sum(val) * (float)(tisg == (tpsg - 1));
    
    atomic_fetch_add_explicit(simd_sums + (cur_out * sgptg) + sitg, local_sum, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    
    //    float q = atomic_load_explicit(simd_sums + sitg, memory_order_relaxed);
    if(titg == 0) {
        float wtf = 0.0;
        for (uint i = 0; i < sgptg; i++) {
            wtf += atomic_load_explicit(simd_sums + (cur_out * sgptg) + i, memory_order_relaxed);
        }
        context.debici[cur_out] += wtf;
    }
//    }
//    if(sitg == 0) {
//        float q = 0;
//        if (tisg < sgptg) {
//            q = simd_sums[tisg];
//        }
//        simdgroup_barrier(mem_flags::mem_threadgroup);
//        float total = simd_sum(q);
}

kernel void smokeTest(device   OGScratchContextAB& context [[buffer(0)]],
                        constant OGHParams* hParams    [[buffer(1)]],
                        constant OGStateAB& state      [[buffer(2)]],
                        uint tig [[thread_position_in_grid]]
                        ) {
    context.n1[tig] = tig;
}

// Gather embeddings for a list of tokens based on a list of token IDs.
//    • tokenIds  : (1,      seqLen)       Index into embeddings' slowest dimension
//    • output    : (seqLen, dModel)  Where the row at index i is embeddings' row tokenIds[i]
//    • embeddings: (nVocab, dModel)  Embeddings
kernel void getEmbeddings(device int  *tokenIds [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          device float *embeddings [[buffer(2)]],
                          device uint& seqLen [[buffer(3)]],
                          device uint& nVocab [[buffer(4)]],
                          device uint& dModel [[buffer(5)]],
                          uint2 tig [[thread_position_in_grid]]
                          ) {
        uint embRowId     = tokenIds[tig.y];
    output[(tig.y * dModel) + tig.x] = embeddings[(embRowId * dModel) + tig.x];
}

kernel void computeDeltaACube(constant float *spDelta [[buffer(0)]],
                              constant float *A       [[buffer(1)]],
                              constant    uint& seqLen   [[buffer(2)]],
                              constant    uint& dInner   [[buffer(3)]],
                              constant    uint& dState   [[buffer(4)]],
                              device float *deltaAOut    [[buffer(5)]],
                              uint2 tig [[thread_position_in_grid]]
                              ) {
    //deltaAOut[(tig.y * )]
}

// Perform the selective scan
kernel void selectiveScan(constant float *u       [[buffer(0)]],
                          constant float *spDelta [[buffer(1)]],
                          constant float *A       [[buffer(2)]],
                          constant float *B       [[buffer(3)]],
                          constant float *C       [[buffer(4)]],
                          constant float *D       [[buffer(5)]],
                          constant    uint& li    [[buffer(6)]],
                          constant    uint& dInner   [[buffer(7)]],
                          constant    uint& dState   [[buffer(8)]],
                          device float *x       [[buffer(9)]],
                          device float *yOut    [[buffer(10)]],
                          uint2 tig [[thread_position_in_grid]],
                          uint2 titg [[thread_position_in_threadgroup]],
                          uint tisg [[thread_index_in_simdgroup]]
                          ) {
    uint dini = tig.x;     // Starting index along dInner axis (0 <= dini < dInner)
    uint ni   = tig.y; // Starting index along dState axis (0, 4, 8, 12) for dState=16 row ni
    
    float ACubeInNi = exp(A[(dini * dState) + ni] * spDelta[dini]);
    float BuCubeInNi = spDelta[dini] * B[ni] * u[dini];
    
    // x = Ax + deltaBu
    float xup = ACubeInNi * x[dini * dState + ni] + BuCubeInNi;
    x[dini * dState + ni] = xup;
    
    float xCs = simd_sum(x[dini * dState + ni] * C[ni]);
    yOut[dini + (li * dInner)] = xCs + u[dini] * D[dini];
}

// Perform the selective scan
kernel void discretizeAndScanSSM(device   OGScratchContextAB& context [[buffer(0)]],
                    constant OGHParams* hParams    [[buffer(1)]],
                    constant OGStateAB& state      [[buffer(2)]],
                    constant uint& layer_number    [[buffer(3)]],
                    constant uint& L               [[buffer(4)]],
                    constant uint& dispatchWidth   [[buffer(5)]],
                    device   float* debugme        [[buffer(6)]],
                    threadgroup float* u           [[threadgroup(0)]],
                    threadgroup float* dlt         [[threadgroup(1)]],
                    threadgroup float* a           [[threadgroup(2)]],
                    threadgroup float* b           [[threadgroup(3)]],
                    threadgroup float* c           [[threadgroup(4)]],
                    threadgroup float* x           [[threadgroup(8)]],
                    uint3 tig   [[thread_position_in_grid]],
                    uint3 tgpig [[threadgroup_position_in_grid]],
                    uint3 tgpg  [[threadgroups_per_grid]],
                    uint  titg  [[thread_index_in_threadgroup]],
                    uint  tisg  [[thread_index_in_simdgroup]],
                    uint  tpsg  [[threads_per_simdgroup]],
                    uint3 tptg  [[threads_per_threadgroup]],
                    uint  sitg  [[simdgroup_index_in_threadgroup]],
                    uint  sgptg [[simdgroups_per_threadgroup]]) {
    uint dIn_cur = tig.x;
    uint n_cur = tig.z;
    
    float d = state.layers[layer_number].D[dIn_cur];
    uint dn_cur   = (dIn_cur * hParams->dState) + n_cur;
    a[n_cur] = state.layers[layer_number].A[dn_cur];
    float x_prev = 0.0;
    x[n_cur] = 0.0;
    
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint l_cur = 0; n_cur < hParams->dState && l_cur < L; l_cur++) {
        uint ld_cur     =  l_cur * hParams->dInner  + dIn_cur;
        uint dsp_ld_cur = (l_cur * dispatchWidth)   + dIn_cur;
        uint l2d_cur    = (dIn_cur * L) + l_cur;
        
        float uf = context.dbld_in[l2d_cur];
        
        float dluf = context.deli[ld_cur];
        
        // (0 <= n_cur < dState) (hope dState == 16)
        uint B_cur   = (l_cur * hParams->dXProj) + hParams->dtRank + n_cur;
        uint C_cur   = (l_cur * hParams->dXProj) + hParams->dtRank + hParams->dState + n_cur;
        
        uint dsp_ln_cur = (l_cur * hParams->dState) + n_cur;
        
        float buuf = context.debici[B_cur];
        b[dsp_ln_cur] = buuf;
        
        float cuuf = context.debici[C_cur];
        c[dsp_ln_cur] = cuuf;
        
        float aloof = a[n_cur];
        float a__ = exp(aloof * dluf);
        float b__ = dluf * buuf * uf;
        
        float wooof = x_prev;
        x[n_cur] = wooof * a__ + b__;
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float xoof = x[n_cur];
        float y = xoof * cuuf;
        float ys = simd_sum(y);
        if(tig.x == 3) {
            debugme[n_cur * 3 + 0] = cuuf;
            debugme[n_cur * 3 + 1] = xoof;
            debugme[n_cur * 3 + 2] = y;
        }
        
        context.tricky[ld_cur] = ys + uf * d;
        x_prev = xoof;
    }
    
    
//    threadgroup_barrier(mem_flags::mem_device);
//    for(uint l_cur = 0; l_cur < L; l_cur++) {
//        uint ld_cur     =  l_cur * hParams->dInner  + dIn_cur;
//        context.tricky[ld_cur] = context.tricky[ld_cur] + u[ld_cur] * d[dIn_cur];
//    }
    
    
//    for(uint n_cur = 0; n_cur < hParams->dState; n_cur++) {
//        context.dA_l[dIn_cur * hParams->dState + n_cur]  = exp(A * spDelta);
//        context.dBu_l[dIn_cur * hParams->dState + n_cur] = spDelta * B * u;
//    }
    
    //    uint dini = tig.x;     // Starting index along dInner axis (0 <= dini < dInner)
//    uint ni   = tig.y; // Starting index along dState axis (0, 4, 8, 12) for dState=16 row ni
    
//    float ACubeInNi = exp(A[(dini * dState) + ni] * spDelta[dini]);
//    float BuCubeInNi = spDelta[dini] * B[ni] * u[dini];
//    
//    // x = Ax + deltaBu
//    float xup = ACubeInNi * x[dini * dState + ni] + BuCubeInNi;
//    x[dini * dState + ni] = xup;
//    
//    float xCs = simd_sum(x[dini * dState + ni] * C[ni]);
//    yOut[dini + (li * dInner)] = xCs + u[dini] * D[dini];
}

// Perform the selective scan
kernel void discretize(device   OGScratchContextAB& context [[buffer(0)]],
                       constant OGHParams* hParams    [[buffer(1)]],
                       constant OGStateAB& state      [[buffer(2)]],
                       constant uint& layer_number    [[buffer(3)]],
                       constant uint& L               [[buffer(4)]],
                       constant uint& lCur            [[buffer(5)]],
                       uint3 tig  [[thread_position_in_grid]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       uint3 tgpg [[threadgroups_per_grid]],
                       uint titg [[thread_index_in_threadgroup]],
                       uint  tisg [[thread_index_in_simdgroup]],
                       uint  tpsg [[threads_per_simdgroup]],
                       uint3  tptg [[threads_per_threadgroup]],
                       uint  sitg [[simdgroup_index_in_threadgroup]],
                       uint  sgptg [[simdgroups_per_threadgroup]]
                          ) {
    // (0 <= dIn_cur < dInner) (offsetting 0 <= tig.z <= maxThreadsPerThreadgroup at a time)
    uint dIn_cur = tig.x;
    uint deli_cur = lCur * hParams->dInner + dIn_cur;
    
    // (0 <= n_cur < dState) (hope dState == 16)
    uint n_cur   = tig.y;
    uint B_cur    = (lCur * hParams->dXProj) + hParams->dtRank + n_cur;
    
    float spDelta = context.deli[deli_cur];
    float A = state.layers[layer_number].A[(dIn_cur * hParams->dState) + n_cur];
    float B = context.debici[B_cur];
    float u = context.dbld_in[(dIn_cur * L) + lCur];
    
    context.dA_l[dIn_cur * hParams->dState + n_cur]  = exp(A * spDelta);
    context.dBu_l[dIn_cur * hParams->dState + n_cur] = spDelta * B * u;
    
    // x = Ax + deltaBu
//    float xup = ACubeInNi * x[dIn_cur * dState + ni] + BuCubeInNi;
//    context.ssx[dIn_cur * hParams->dState + n_cur] = xup;
//    
//    float xCs = simd_sum(x[dini * dState + ni] * C[ni]);
//    yOut[dini + (li * dInner)] = xCs + u[dini] * D[dini];
}
