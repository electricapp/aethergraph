// K5.1 persistent loader with warp specialization (roofline attempt).
//
// One CTA, three warps:
//   warp 0 FETCH     — claim SPSC ring slots into a shared fetch→xform queue
//   warp 1 TRANSFORM — pop fetch queue, light touch / classify, push compute q
//   warp 2 COMPUTE   — pop compute queue, account completion
//
// Forward progress: nanosleep only when the global ring AND both local
// queues are empty. Fetch never waits on compute; compute never holds a
// lock the fetch warp needs. Host is the sole global producer.
//
// On stop: drain local queues before exit so posted work is not lost.
//
// Payload bodies are still placeholders (count work) — RDMA/gather fill
// lands on the same queues. TODO(HARDWARE): prove under MPS + RDMA producers.

struct PersistentWork {
    unsigned int kind;
    unsigned long long payload;
    unsigned int len;
};

static const int LOCAL_CAP = 64; // power of two

extern "C" __global__ void persistent_work_drain(
    PersistentWork* ring,
    unsigned int* head,
    const unsigned int* tail,
    const int* stop,
    unsigned long long* completed,
    int capacity
) {
    __shared__ PersistentWork fetch_q[LOCAL_CAP];
    __shared__ PersistentWork compute_q[LOCAL_CAP];
    __shared__ volatile unsigned int fq_head, fq_tail, cq_head, cq_tail;

    if (threadIdx.x == 0) {
        fq_head = fq_tail = cq_head = cq_tail = 0;
    }
    __syncthreads();

    if (capacity <= 0 || blockIdx.x != 0) return;
    const unsigned int gmask = (unsigned int)(capacity - 1);
    const unsigned int lmask = (unsigned int)(LOCAL_CAP - 1);
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);

    while (true) {
        const int stopping = *(volatile int*)stop;
        const unsigned int h = *(volatile unsigned int*)head;
        const unsigned int t = *(volatile unsigned int*)tail;
        const unsigned int fqh = fq_head;
        const unsigned int fqt = fq_tail;
        const unsigned int cqh = cq_head;
        const unsigned int cqt = cq_tail;
        const bool global_empty = (h == t);
        const bool local_empty = (fqh == fqt) && (cqh == cqt);

        if (stopping && global_empty && local_empty) {
            break;
        }

        if (warp == 0 && lane == 0) {
            if (!global_empty && (fqt - fqh) < (unsigned int)LOCAL_CAP) {
                const PersistentWork work = ring[h & gmask];
                *(volatile unsigned int*)head = h + 1;
                fetch_q[fqt & lmask] = work;
                __threadfence_block();
                fq_tail = fqt + 1;
            } else if (global_empty && local_empty && !stopping) {
                __nanosleep(500);
            }
        } else if (warp == 1 && lane == 0) {
            if (fqh != fqt && (cqt - cqh) < (unsigned int)LOCAL_CAP) {
                const PersistentWork work = fetch_q[fqh & lmask];
                fq_head = fqh + 1;
                PersistentWork out = work;
                out.kind = work.kind; // classify hook
                compute_q[cqt & lmask] = out;
                __threadfence_block();
                cq_tail = cqt + 1;
            } else if (global_empty && local_empty && !stopping) {
                __nanosleep(500);
            }
        } else if (warp == 2 && lane == 0) {
            if (cqh != cqt) {
                const PersistentWork work = compute_q[cqh & lmask];
                cq_head = cqh + 1;
                (void)work;
                atomicAdd(completed, 1ULL);
            } else if (global_empty && local_empty && !stopping) {
                __nanosleep(500);
            }
        } else if (!stopping) {
            __nanosleep(1000);
        }
    }
}
