#include <simd/simd.h>
#ifndef Arguments_h
#define Arguments_h
#if __METAL_VERSION__
#define CONSTANT_PTR(x) constant x*
#define DEVICE_PTR(x)   device x*
#else
#define CONSTANT_PTR(x) uint64_t
#define DEVICE_PTR(x)   uint64_t
#endif

struct OGHParams {
    /// Number of Mamba Layers in the model
    uint32_t nLayers;
    
    /// SSM state expansion factor (aka n)
    uint32_t dState;
    
    /// Vocab size
    uint32_t nVocab;
    
    /// Model dimension (aka d, d_model)
    uint32_t dModel;
    
    /// Block expansion factor (aka e)
    uint32_t expand;
    
    /// Local convolution width
    uint32_t dConv;
    
    /// Eh, it's 1, no domain model is perfect
    uint32_t dConvHeight;
    
    /// Rank of ∆
    uint32_t dtRank;
    
    /// Hidden state dimension (aka d_in)
    uint32_t dInner;
    
    /// Set to dInner * expand
    uint32_t dInProj;
    
    /// Set to dtRank + 2*dState
    uint32_t dXProj;
    
};

// Argument buffers
struct OGLayerAB {
    CONSTANT_PTR(float) norm;
    CONSTANT_PTR(float) conv1d_bias;
    CONSTANT_PTR(float) conv1d_weight;
    CONSTANT_PTR(float) A;
    CONSTANT_PTR(float) D;
    CONSTANT_PTR(float) dt_proj_bias;
    CONSTANT_PTR(float) dt_proj_weight;
    
    CONSTANT_PTR(float)  in_proj_weight;
    CONSTANT_PTR(float)  out_proj_weight;
    CONSTANT_PTR(float)  x_proj_weight;
};

struct OGStateAB {
    CONSTANT_PTR(float)      embedding;
    CONSTANT_PTR(float)      lm_head;
    CONSTANT_PTR(float)      norm_f;
    CONSTANT_PTR(OGLayerAB)  layers;
};

struct OGScratchContextAB {
    /// original input/projection in, post-conv1d->silu, and out projection/last residual addition
    DEVICE_PTR(float)    ld1;
    
    /// residual copy, hangs around to add to ld1
    DEVICE_PTR(float)    ld2;
    
    /// gentle giant, only used for x_and_res, chill type personality
    DEVICE_PTR(float)    dbld_in;
    
    /// small buffer for ∆BC in the SSM after xProj reads ld1
    DEVICE_PTR(float)    debici;
    
    /// lazy type evolution of debici after using dtProj item
    DEVICE_PTR(float)    deli;
    
    /// ∆A but only for one l, making it (dIn, n) instead of (l, dIn, n)
    DEVICE_PTR(float)    dA_l;
    
    /// Same but for ∆Bu
    DEVICE_PTR(float)    dBu_l;
    
    /// state space X buffer, small but powerful, gets written and zeroed a lot
    DEVICE_PTR(float)    ssx;
    
    /// trick buffer for y, we repeatedly do small x-grabs to fill up the uber meter, as opposed to one huge A/B grab (l, dIn, n is too slow for our threadgroup airtime), once this is full we can combo with deli
    DEVICE_PTR(float)    tricky;
    
    /// nVocab/1 spot for out logits
    DEVICE_PTR(float)    n1;
};

#endif /* Arguments_h */
