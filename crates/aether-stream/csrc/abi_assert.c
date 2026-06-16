/* Cross-check the hand-written #[repr(C)] mirrors in src/rdma/ffi.rs against the
 * real libibverbs ABI. Compiled under the `rdma` feature on Linux against the
 * installed <infiniband/verbs.h>; any drift is a build error, turning silent
 * memory corruption into a failed build. The matching size_of/offset_of asserts
 * on the Rust side pin our structs; these pin the real headers to the same
 * values we depend on.
 *
 * Direction of the size checks: for structs the kernel reads from (ibv_send_wr)
 * or writes into (ibv_wc, ibv_port_attr), our mirror must be at least as large
 * as the real struct so the kernel never reads or writes past our allocation —
 * hence `sizeof(real) <= sizeof(ours)`. Field offsets we read or walk must match
 * exactly. */

#include <infiniband/verbs.h>
#include <stddef.h>

/* Work-request fields ibv_post_send reads and walks. */
_Static_assert(offsetof(struct ibv_send_wr, next) == 8, "ibv_send_wr.next offset");
_Static_assert(offsetof(struct ibv_send_wr, sg_list) == 16, "ibv_send_wr.sg_list offset");
_Static_assert(offsetof(struct ibv_send_wr, num_sge) == 24, "ibv_send_wr.num_sge offset");
_Static_assert(offsetof(struct ibv_send_wr, opcode) == 28, "ibv_send_wr.opcode offset");
_Static_assert(offsetof(struct ibv_send_wr, send_flags) == 32, "ibv_send_wr.send_flags offset");
_Static_assert(sizeof(struct ibv_send_wr) <= 128, "IbvSendWr smaller than ibv_send_wr");

/* Completion fields the poll loop reads. */
_Static_assert(offsetof(struct ibv_wc, wr_id) == 0, "ibv_wc.wr_id offset");
_Static_assert(offsetof(struct ibv_wc, status) == 8, "ibv_wc.status offset");
_Static_assert(sizeof(struct ibv_wc) <= 48, "IbvWc smaller than ibv_wc");

/* Fields read directly off the QP / MR handles. */
_Static_assert(offsetof(struct ibv_qp, qp_num) == 52, "ibv_qp.qp_num offset");
_Static_assert(offsetof(struct ibv_mr, lkey) == 36, "ibv_mr.lkey offset");
_Static_assert(offsetof(struct ibv_mr, rkey) == 40, "ibv_mr.rkey offset");

/* The kernel writes ibv_query_port into our IbvPortAttr; ours must be at least
 * as large so the write cannot overrun. */
_Static_assert(sizeof(struct ibv_port_attr) <= 56, "IbvPortAttr smaller than ibv_port_attr");
