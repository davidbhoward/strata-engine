// src/kernels/topk_split_parity.cpp - qsa_block_topk_split (each query over several CTAs, no clusters) against
// qsa_block_topk_ref and qsa_block_topk's dispatch, bit for bit: every id of every row of the `cap`-wide buffer,
// including the entries past the width that none may write (GPU, synthetic, no model).
//
// Inputs: decode_cluster_parity's fixtures - random indexer-like scores and adversarial ones (all equal, a few values,
// NaN, +-inf, signed zeros, denormals, monotone rows, all NaN, quantized) - at contexts from below the selection width
// to 262,143 cells, every tail weight, 1-8 queries; and one CUDA graph that captures the split launches, replayed with
// new inputs.
//
//   topk_split_parity --selftest     parity (exit 0 = identical everywhere)
//   topk_split_parity --bench        per-call times at 32K / 128K / 240K, 1-8 queries
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, count * sizeof(T) + 64), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T>& h) {
        ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
        ck(cudaDeviceSynchronize(), "up landed");
    }
    std::vector<T> get() const {
        std::vector<T> h(n);
        ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

const float kNaN = std::numeric_limits<float>::quiet_NaN(), kInf = std::numeric_limits<float>::infinity();
enum Pattern { kRandom, kAllEqual, kFewValues, kNaNs, kInfs, kSignedZero, kDenormal, kAscending, kDescending, kAllNaN,
               kQuantized, kPatterns };
const char* pattern_name(int p) {
    static const char* n[] = {"random", "all-equal", "few-values", "nan", "inf", "signed-zero", "denormal", "ascending",
                              "descending", "all-nan", "quantized"};
    return n[p];
}

void fill_row(float* row, int64_t n_kv, int64_t n_bid, int pattern, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pick(0, 99);
    for (int64_t b = 0; b <= n_bid; ++b) {
        float v = 0.f;
        const float g = nd(rng);
        switch (pattern) {
            case kRandom: v = std::fmax(0.f, g) * 3.f + std::fmax(0.f, nd(rng)); break;
            case kAllEqual: v = 1.f; break;
            case kFewValues: v = (float) (pick(rng) % 4) * 0.5f; break;
            case kNaNs: v = pick(rng) < 15 ? kNaN : std::fabs(g); break;
            case kInfs: {
                const int r = pick(rng);
                v = r < 5 ? kInf : r < 15 ? -kInf : r < 30 ? -std::fabs(g) : g;
                break;
            }
            case kSignedZero: v = pick(rng) < 50 ? -0.0f : (pick(rng) < 50 ? 0.0f : -std::fabs(g) * 1e-3f); break;
            case kDenormal: v = (float) (pick(rng) % 7) * 1e-41f; break;
            case kAscending: v = (float) b; break;
            case kDescending: v = (float) (n_bid - b); break;
            case kAllNaN: v = kNaN; break;
            case kQuantized: v = std::round(std::fabs(g) * 8.f) / 8.f; break;
        }
        row[b] = v;
    }
    if (n_kv % 4 != 0 && pattern != kAllNaN && pattern != kAllEqual) row[n_bid] += 1e9f;
}

struct Inputs {
    int64_t max_blocks = 0, cap = 0;
    std::vector<float> sc;
    std::vector<int32_t> steps;
};

Inputs make_inputs(int64_t ctx, int64_t capacity, int nq, int pattern, std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    Inputs in;
    in.max_blocks = capacity / 4 + 2;
    in.cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    in.sc.assign((size_t) (nq * in.max_blocks), 0.f);
    in.steps.assign((size_t) nq * k::kStepCount, 0);
    for (int i = 0; i < nq; ++i) {
        const int64_t n_kv = ctx - (nq - 1 - i);
        int32_t* st = in.steps.data() + (size_t) i * k::kStepCount;
        st[k::kStepPos] = (int32_t) (n_kv - 1);
        st[k::kStepNKv] = (int32_t) n_kv;
        st[k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
        fill_row(in.sc.data() + (size_t) i * in.max_blocks, n_kv, n_kv / s.idx_block, pattern, rng);
    }
    return in;
}

int compare(const char* what, const std::vector<int32_t>& a, const std::vector<int32_t>& b, int64_t cap, int64_t ctx,
            int nq, int pattern) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            std::printf("FAIL %s ctx %lld nq %d %s: row %lld entry %lld: %d vs %d\n", what, (long long) ctx, nq,
                        pattern_name(pattern), (long long) (i / (size_t) cap), (long long) (i % (size_t) cap), a[i], b[i]);
            return 1;
        }
    return 0;
}

int run_case(int64_t ctx, int64_t capacity, int nq, int pattern, std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const Inputs in = make_inputs(ctx, capacity, nq, pattern, rng);
    Dev<float> d_sc(in.sc.size());
    const size_t n_ids = (size_t) (nq * in.cap);
    Dev<int32_t> d_st(in.steps.size()), d_new(n_ids), d_ref(n_ids), d_disp(n_ids),
        d_scr((size_t) k::qsa_block_topk_split_scratch(nq));
    d_sc.put(in.sc);
    d_st.put(in.steps);
    ck(cudaMemset(d_new.p, 0xA5, d_new.n * 4), "memset");
    ck(cudaMemset(d_ref.p, 0xA5, d_ref.n * 4), "memset");
    ck(cudaMemset(d_disp.p, 0xA5, d_disp.n * 4), "memset");
    if (!k::qsa_block_topk_split(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_new.p, d_scr.p, nullptr)) {
        std::printf("FAIL: the split kernel did not run\n");
        return 1;
    }
    k::qsa_block_topk_ref(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_ref.p, nullptr);
    k::qsa_block_topk(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_disp.p, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const auto a = d_new.get(), b = d_ref.get(), c = d_disp.get();
    return compare("split vs ref", a, b, in.cap, ctx, nq, pattern) + compare("split vs dispatch", a, c, in.cap, ctx, nq, pattern);
}

// the split launches captured once in a graph, replayed with new inputs written into the same buffers
int run_graph_case(std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int nq = 4;
    const int64_t capacity = 262144;
    Inputs in = make_inputs(200001, capacity, nq, kRandom, rng);
    Dev<float> d_sc(in.sc.size());
    const size_t n_ids = (size_t) (nq * in.cap);
    Dev<int32_t> d_st(in.steps.size()), d_new(n_ids), d_ref(n_ids), d_scr((size_t) k::qsa_block_topk_split_scratch(nq));
    d_sc.put(in.sc);
    d_st.put(in.steps);
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    cudaGraph_t g;
    ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture");
    k::qsa_block_topk_split(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_new.p, d_scr.p, cs);
    ck(cudaStreamEndCapture(cs, &g), "end capture");
    cudaGraphExec_t ge;
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    int fails = 0;
    const int64_t ctxs[] = {200001, 1000, 262143, 35000, 131075};
    const int pats[] = {kRandom, kFewValues, kQuantized, kNaNs, kAllEqual};
    for (int r = 0; r < 5; ++r) {
        in = make_inputs(ctxs[r], capacity, nq, pats[r], rng);
        d_sc.put(in.sc);
        d_st.put(in.steps);
        ck(cudaMemset(d_new.p, 0xA5, d_new.n * 4), "memset");
        ck(cudaMemset(d_ref.p, 0xA5, d_ref.n * 4), "memset");
        ck(cudaGraphLaunch(ge, cs), "launch");
        ck(cudaStreamSynchronize(cs), "replay");
        k::qsa_block_topk_ref(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_ref.p, nullptr);
        ck(cudaDeviceSynchronize(), "sync");
        fails += compare("graph replay", d_new.get(), d_ref.get(), in.cap, ctxs[r], nq, pats[r]);
    }
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(cs);
    return fails;
}

template <typename F> float time_us(cudaStream_t cs, F&& f, int reps = 300) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) f();
    cudaEventRecord(a, cs);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b, cs);
    ck(cudaEventSynchronize(b), "time");
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms * 1000.f / (float) reps;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::mt19937 rng(1234);
    if (!bench) {
        int fails = 0, cases = 0;
        const int64_t ctxs[] = {5, 700, 2048, 2049, 4097, 33790, 33797, 90001, 135170, 200002, 240003, 262143};
        for (const int64_t ctx : ctxs)
            for (int nq : {1, 3, 4, 8})
                for (int p = 0; p < kPatterns; ++p) {
                    if (ctx < nq) continue;
                    fails += run_case(ctx, 262144, nq, p, rng);
                    ++cases;
                }
        for (const int64_t ctx : {1000LL, 30001LL, 65538LL})   // a smaller capacity too (max_blocks = capacity / 4 + 2)
            for (int p = 0; p < kPatterns; ++p) { fails += run_case(ctx, 65536, 2, p, rng); ++cases; }
        fails += run_graph_case(rng);
        std::printf("%s: %d cases + a graph replay, %d failures\n", fails ? "FAIL" : "PASS", cases, fails);
        return fails ? 1 : 0;
    }
    const k::QsaShapes s = k::qsa_real_shapes();
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    std::printf("QSA top-k per call (us), capacity 262144\n");
    for (const int64_t ctx : {32768LL, 131072LL, 240000LL})
        for (const int nq : {1, 4, 8}) {
            const Inputs in = make_inputs(ctx, 262144, nq, kRandom, rng);
            Dev<float> d_sc(in.sc.size());
            Dev<int32_t> d_st(in.steps.size()), d_ids((size_t) (nq * in.cap)),
                d_scr((size_t) k::qsa_block_topk_split_scratch(nq));
            d_sc.put(in.sc);
            d_st.put(in.steps);
            const float t_disp = time_us(cs, [&] { k::qsa_block_topk(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_ids.p, cs); });
            const float t_split = time_us(cs, [&] {
                k::qsa_block_topk_split(d_sc.p, d_st.p, nq, in.max_blocks, in.cap, s, d_ids.p, d_scr.p, cs);
            });
            std::printf("  ctx %6lld nq %d: dispatch %7.1f  split %7.1f  -> %.2fx\n", (long long) ctx, nq, t_disp, t_split,
                        t_disp / t_split);
        }
    return 0;
}
