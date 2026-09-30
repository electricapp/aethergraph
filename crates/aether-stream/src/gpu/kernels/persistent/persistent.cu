// K5.1 persistent loader with warp specialization (roofline attempt).
//
// One CTA, three worker threads (lane 0 of warps 0-2):
//   warp 0 FETCH     — claim host ring slots into a shared fetch→xform queue
//   warp 1 TRANSFORM — pop fetch queue, light touch / classify, push compute q
//   warp 2 COMPUTE   — pop compute queue, account completion
//
// The control block and ring live in mapped pinned host memory, so the host
// posts with plain stores and never issues a CUDA call against a stream the
// kernel occupies. Host → device words (`tail`, `stop`, ring entries) are
// read at system scope with acquire/relaxed loads; device → host words
// (`head`, `completed`) are published with system-scope release stores.
//
// Exit: COMPUTE is the only thread that knows how much work is finished, so
// it alone decides. Once `stop` is observed, the `tail` read after it is the
// final post count; when `completed` reaches it every posted item has been
// fetched and computed, both local queues are empty, and the kernel exits.
//
// Payload bodies are still placeholders (count work) — RDMA/gather fill
// lands on the same queues. TODO(HARDWARE): prove under MPS + RDMA producers.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "persistent needs sm_70 for __nanosleep and .sys-scoped ld/st; see NVRTC_ARCH_FLOOR"
#endif

struct PersistentWork {
    unsigned int kind;
    unsigned long long payload;
    unsigned int len;
};

// Mirrors `PersistentControl` in persistent/mod.rs.
struct PersistentControl {
    unsigned int tail;               // host → device: items posted
    unsigned int stop;               // host → device: nonzero once posting ends
    unsigned int head;               // device → host: items claimed
    unsigned int pad;
    unsigned long long completed;    // device → host: items finished
};

static const int LOCAL_CAP = 64; // power of two

__device__ __forceinline__ unsigned int ld_acquire_sys_u32(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned int ld_relaxed_sys_u32(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned long long ld_relaxed_sys_u64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys_u32(unsigned int* p, unsigned int v) {
    asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_sys_u64(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

extern "C" __global__ void persistent_work_drain(
    PersistentControl* ctl,
    const PersistentWork* ring,
    int capacity
) {
    __shared__ PersistentWork fetch_q[LOCAL_CAP];
    __shared__ PersistentWork compute_q[LOCAL_CAP];
    __shared__ volatile unsigned int fq_head, fq_tail, cq_head, cq_tail, done_flag;

    if (threadIdx.x == 0) {
        fq_head = fq_tail = cq_head = cq_tail = done_flag = 0;
    }
    __syncthreads();

    if (capacity <= 0 || blockIdx.x != 0) return;
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (lane != 0 || warp > 2) return;

    const unsigned int gmask = (unsigned int)(capacity - 1);
    const unsigned int lmask = (unsigned int)(LOCAL_CAP - 1);

    if (warp == 0) {
        unsigned int claimed = 0;
        while (!done_flag) {
            const unsigned int fqh = fq_head;
            const unsigned int fqt = fq_tail;
            if (fqt - fqh >= (unsigned int)LOCAL_CAP) {
                __nanosleep(100);
                continue;
            }
            // Acquire: the entry loads below see what the host wrote
            // before publishing this tail.
            const unsigned int posted = ld_acquire_sys_u32(&ctl->tail);
            if (posted == claimed) {
                __nanosleep(500);
                continue;
            }
            const PersistentWork* src = ring + (claimed & gmask);
            PersistentWork work;
            work.kind = ld_relaxed_sys_u32(&src->kind);
            work.payload = ld_relaxed_sys_u64(&src->payload);
            work.len = ld_relaxed_sys_u32(&src->len);
            // Release: the host may reuse the slot only after the copy.
            claimed += 1;
            st_release_sys_u32(&ctl->head, claimed);
            fetch_q[fqt & lmask] = work;
            __threadfence_block();
            fq_tail = fqt + 1;
        }
    } else if (warp == 1) {
        while (!done_flag) {
            const unsigned int fqh = fq_head;
            const unsigned int fqt = fq_tail;
            const unsigned int cqh = cq_head;
            const unsigned int cqt = cq_tail;
            if (fqh == fqt || cqt - cqh >= (unsigned int)LOCAL_CAP) {
                __nanosleep(100);
                continue;
            }
            // Order the entry read after observing its published index.
            __threadfence_block();
            const PersistentWork work = fetch_q[fqh & lmask];
            __threadfence_block();
            fq_head = fqh + 1;
            // Classify hook: the transform stage rewrites `work` here.
            compute_q[cqt & lmask] = work;
            __threadfence_block();
            cq_tail = cqt + 1;
        }
    } else {
        unsigned long long completed = 0;
        while (true) {
            const unsigned int cqh = cq_head;
            const unsigned int cqt = cq_tail;
            if (cqh != cqt) {
                __threadfence_block();
                const PersistentWork work = compute_q[cqh & lmask];
                __threadfence_block();
                cq_head = cqh + 1;
                (void)work;
                completed += 1;
                st_release_sys_u64(&ctl->completed, completed);
                continue;
            }
            if (ld_acquire_sys_u32(&ctl->stop) != 0) {
                // Read after `stop`, so this is the final post count.
                const unsigned int posted = ld_acquire_sys_u32(&ctl->tail);
                if ((unsigned int)completed == posted) {
                    done_flag = 1;
                    break;
                }
            }
            __nanosleep(500);
        }
    }
}
