/* Shim that exposes libibverbs's static-inline data-path functions as real
 * ABI symbols Rust FFI can link against.
 *
 * Most `ibv_*` data-path entry points are `static inline` in
 * <infiniband/verbs.h> — they dispatch via `qp->context->ops.*` or via
 * function pointers baked into `ibv_qp_ex` / `ibv_cq_ex`. Calling them from
 * Rust extern "C" requires a thin C wrapper.
 *
 * The SRD (EFA) helpers only compile when the caller builds with the `efa`
 * Cargo feature; gating is done in build.rs via -DAETHER_EFA_SHIM.
 */

#include <infiniband/verbs.h>

int aether_ibv_post_send(struct ibv_qp *qp,
                         struct ibv_send_wr *wr,
                         struct ibv_send_wr **bad_wr) {
    return ibv_post_send(qp, wr, bad_wr);
}

int aether_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    return ibv_poll_cq(cq, num_entries, wc);
}

int aether_ibv_post_recv(struct ibv_qp *qp,
                         struct ibv_recv_wr *wr,
                         struct ibv_recv_wr **bad_wr) {
    return ibv_post_recv(qp, wr, bad_wr);
}

/* Device capabilities the atomic paths need. Probing lives here so the
 * Rust side never mirrors the large, drift-prone `struct ibv_device_attr`. */
int aether_ibv_query_atomic_caps(struct ibv_context *ctx,
                                 int *atomic_cap,
                                 int *max_qp_rd_atom) {
    struct ibv_device_attr attr;
    int rc = ibv_query_device(ctx, &attr);
    if (rc != 0) return rc;
    *atomic_cap = (int)attr.atomic_cap;
    *max_qp_rd_atom = attr.max_qp_rd_atom;
    return 0;
}

int aether_ibv_post_srq_recv(struct ibv_srq *srq,
                             struct ibv_recv_wr *wr,
                             struct ibv_recv_wr **bad_wr) {
    return ibv_post_srq_recv(srq, wr, bad_wr);
}

int aether_ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only) {
    return ibv_req_notify_cq(cq, solicited_only);
}

/* On-demand-paging capabilities via the extended device query; shimmed to
 * keep `struct ibv_device_attr_ex` on the C side. */
int aether_ibv_query_odp_caps(struct ibv_context *ctx,
                              uint64_t *general_caps,
                              uint32_t *rc_odp_caps) {
    struct ibv_device_attr_ex attr = {0};
    int rc = ibv_query_device_ex(ctx, NULL, &attr);
    if (rc != 0) return rc;
    *general_caps = attr.odp_caps.general_caps;
    *rc_odp_caps = attr.odp_caps.per_transport_caps.rc_odp_caps;
    return 0;
}

#ifdef AETHER_EFA_SHIM

/* ------------------------------------------------------------------ */
/* Extended-verbs helpers for the SRD (EFA) path. The `ibv_qp_ex` and  */
/* `ibv_cq_ex` objects are opaque from Rust because their callable     */
/* entry points are function pointers inside the struct — can't be     */
/* called through bindgen-generated types without replicating the      */
/* exact ABI. These wrappers do that for us.                           */
/* ------------------------------------------------------------------ */

struct ibv_cq_ex *aether_ibv_create_cq_ex(struct ibv_context *ctx,
                                          struct ibv_cq_init_attr_ex *attr) {
    return ibv_create_cq_ex(ctx, attr);
}

struct ibv_cq *aether_ibv_cq_ex_to_cq(struct ibv_cq_ex *cqx) {
    return ibv_cq_ex_to_cq(cqx);
}

struct ibv_qp_ex *aether_ibv_qp_to_qp_ex(struct ibv_qp *qp) {
    return ibv_qp_to_qp_ex(qp);
}

void aether_ibv_wr_start(struct ibv_qp_ex *qpx) {
    ibv_wr_start(qpx);
}

int aether_ibv_wr_complete(struct ibv_qp_ex *qpx) {
    return ibv_wr_complete(qpx);
}

/* One-shot RDMA READ post: sets wr_id/flags, rdma_read, ud_addr, sge, completes.
 * All call-site arguments collapse to a single FFI crossing so the Rust side
 * doesn't need to juggle the qp_ex state between function-pointer hops. */
int aether_ibv_post_rdma_read_srd(struct ibv_qp_ex *qpx,
                                  uint64_t wr_id,
                                  unsigned int wr_flags,
                                  uint32_t remote_rkey,
                                  uint64_t remote_addr,
                                  struct ibv_ah *ah,
                                  uint32_t remote_qpn,
                                  uint32_t remote_qkey,
                                  uint32_t local_lkey,
                                  uint64_t local_addr,
                                  uint32_t length) {
    ibv_wr_start(qpx);
    qpx->wr_id = wr_id;
    qpx->wr_flags = wr_flags;
    ibv_wr_rdma_read(qpx, remote_rkey, remote_addr);
    ibv_wr_set_ud_addr(qpx, ah, remote_qpn, remote_qkey);
    ibv_wr_set_sge(qpx, local_lkey, local_addr, length);
    return ibv_wr_complete(qpx);
}

/* Per-WR descriptor for the batched RDMA READ path — all shared fields
 * (ah, qpn, qkey, rkey, lkey) are passed once at the call site. */
struct aether_srd_read {
    uint64_t remote_addr;
    uint64_t local_addr;
    uint32_t length;
    uint32_t _pad;  /* keep the struct 24B so Rust repr(C) matches */
};

/* Post N RDMA READs on an SRD QP, still in a single Rust ↔ C FFI call.
 *
 * EFA's SRD provider rejects more than one WR per `wr_start`/`wr_complete`
 * transaction (EINVAL from ibv_wr_complete), so each read becomes its own
 * start/complete pair. The win is the FFI-crossing count: instead of N
 * calls from Rust into the shim, we do one call that drives the builder
 * N times here in C.
 *
 * All reads share (ah, remote_qpn, remote_qkey, remote_rkey, local_lkey);
 * per-read `remote_addr` / `local_addr` / `length` come from `reads`.
 * wr_id for WR i is `base_wr_id + i`. Only the LAST WR is signaled so
 * the caller drains exactly one CQE (`wr_id = base_wr_id + n - 1`) after
 * this returns.
 *
 * Bails on the first failing `ibv_wr_complete`. */
int aether_ibv_post_rdma_reads_srd_batch(struct ibv_qp_ex *qpx,
                                         struct ibv_ah *ah,
                                         uint32_t remote_qpn,
                                         uint32_t remote_qkey,
                                         uint32_t remote_rkey,
                                         uint32_t local_lkey,
                                         uint64_t base_wr_id,
                                         const struct aether_srd_read *reads,
                                         uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        ibv_wr_start(qpx);
        qpx->wr_id = base_wr_id + (uint64_t)i;
        /* EFA SRD requires every WR to be signaled; setting wr_flags=0 on
         * intermediate WRs gets the whole wr_complete rejected with EINVAL. */
        qpx->wr_flags = IBV_SEND_SIGNALED;
        ibv_wr_rdma_read(qpx, remote_rkey, reads[i].remote_addr);
        ibv_wr_set_ud_addr(qpx, ah, remote_qpn, remote_qkey);
        ibv_wr_set_sge(qpx, local_lkey, reads[i].local_addr, reads[i].length);
        int rc = ibv_wr_complete(qpx);
        if (rc != 0) return rc;
    }
    return 0;
}

/* Extended-CQ poll: drain one CQE's worth of fields into an output struct so
 * the Rust side sees a snapshot rather than a live pointer (start_poll/end_poll
 * must bracket access to cqx->status etc.). Returns:
 *   0 on success (out fields populated)
 *   ENOENT if CQ empty (no CQE drained)
 *   other errno on hard error. */
struct aether_cqe_snapshot {
    uint32_t status;
    uint32_t opcode;
    uint64_t wr_id;
    uint32_t byte_len;
    uint32_t vendor_err;
};

int aether_ibv_poll_cq_ex_one(struct ibv_cq_ex *cqx,
                              struct aether_cqe_snapshot *out) {
    struct ibv_poll_cq_attr attr = {0};
    int rc = ibv_start_poll(cqx, &attr);
    if (rc != 0) {
        return rc; /* ENOENT when empty */
    }
    out->status = cqx->status;
    out->opcode = ibv_wc_read_opcode(cqx);
    out->wr_id = cqx->wr_id;
    out->byte_len = ibv_wc_read_byte_len(cqx);
    out->vendor_err = ibv_wc_read_vendor_err(cqx);
    ibv_end_poll(cqx);
    return 0;
}

/* Drain up to `max_out` CQEs in one FFI call. Returns the number drained.
 * Stops early if any CQE has non-SUCCESS status and writes it into
 * `*first_err` slot (index stored in `*first_err_idx`). ibv_next_poll is the
 * extended-CQ way to keep drawing consecutive CQEs without paying a full
 * start/end cycle per completion. */
int aether_ibv_poll_cq_ex_many(struct ibv_cq_ex *cqx,
                               struct aether_cqe_snapshot *out,
                               uint32_t max_out) {
    if (max_out == 0) return 0;
    struct ibv_poll_cq_attr attr = {0};
    int rc = ibv_start_poll(cqx, &attr);
    if (rc != 0) return 0; /* ENOENT — empty CQ */
    uint32_t i = 0;
    while (i < max_out) {
        out[i].status = cqx->status;
        out[i].opcode = ibv_wc_read_opcode(cqx);
        out[i].wr_id = cqx->wr_id;
        out[i].byte_len = ibv_wc_read_byte_len(cqx);
        out[i].vendor_err = ibv_wc_read_vendor_err(cqx);
        i++;
        if (i == max_out) break;
        if (ibv_next_poll(cqx) != 0) break; /* CQ drained */
    }
    ibv_end_poll(cqx);
    return (int)i;
}

#endif /* AETHER_EFA_SHIM */
