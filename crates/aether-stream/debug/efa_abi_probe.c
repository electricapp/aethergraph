/*
 * EFA / extended-verbs ABI probe.
 *
 * Prints struct sizes, field offsets, and enum values we depend on in
 * `src/rdma/efa_ffi.rs`. Run after every kernel / rdma-core / libefa bump
 * on an EFA-capable instance (c7g.16xlarge, c7gn.16xlarge, p4d.24xlarge,
 * etc.) and reconcile any drift with the compile-time `const _` assertions
 * in `efa_ffi.rs`.
 *
 * Build:
 *   gcc -o /tmp/efa_abi_probe efa_abi_probe.c -libverbs -lefa
 * Run:
 *   /tmp/efa_abi_probe
 *
 * The mechanism-of-record for ABI guarantees is the `const _: [(); N] = ...`
 * assertions in efa_ffi.rs; this binary exists so a human can print the
 * ground truth from a current kernel alongside those expectations.
 */

#include <stdio.h>
#include <stddef.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

int main(void) {
    printf("=== struct sizes ===\n");
    printf("ibv_qp_init_attr_ex  = %zu\n", sizeof(struct ibv_qp_init_attr_ex));
    printf("ibv_cq_init_attr_ex  = %zu\n", sizeof(struct ibv_cq_init_attr_ex));
    printf("efadv_qp_init_attr   = %zu\n", sizeof(struct efadv_qp_init_attr));
    printf("efadv_device_attr    = %zu\n", sizeof(struct efadv_device_attr));
    printf("efadv_ah_attr        = %zu\n", sizeof(struct efadv_ah_attr));
    printf("ibv_ah_attr          = %zu\n", sizeof(struct ibv_ah_attr));
    printf("ibv_qp_attr          = %zu\n", sizeof(struct ibv_qp_attr));
    printf("ibv_global_route     = %zu\n", sizeof(struct ibv_global_route));
    printf("ibv_qp_cap           = %zu\n", sizeof(struct ibv_qp_cap));

    printf("\n=== ibv_qp_attr field offsets ===\n");
    printf("  qp_state=%zu cur_qp_state=%zu path_mtu=%zu path_mig_state=%zu\n",
        offsetof(struct ibv_qp_attr, qp_state),
        offsetof(struct ibv_qp_attr, cur_qp_state),
        offsetof(struct ibv_qp_attr, path_mtu),
        offsetof(struct ibv_qp_attr, path_mig_state));
    printf("  qkey=%zu rq_psn=%zu sq_psn=%zu dest_qp_num=%zu qp_access_flags=%zu\n",
        offsetof(struct ibv_qp_attr, qkey),
        offsetof(struct ibv_qp_attr, rq_psn),
        offsetof(struct ibv_qp_attr, sq_psn),
        offsetof(struct ibv_qp_attr, dest_qp_num),
        offsetof(struct ibv_qp_attr, qp_access_flags));
    printf("  cap=%zu ah_attr=%zu alt_ah_attr=%zu pkey_index=%zu port_num=%zu\n",
        offsetof(struct ibv_qp_attr, cap),
        offsetof(struct ibv_qp_attr, ah_attr),
        offsetof(struct ibv_qp_attr, alt_ah_attr),
        offsetof(struct ibv_qp_attr, pkey_index),
        offsetof(struct ibv_qp_attr, port_num));

    printf("\n=== ibv_ah_attr field offsets ===\n");
    printf("  grh=%zu dlid=%zu sl=%zu src_path_bits=%zu static_rate=%zu is_global=%zu port_num=%zu\n",
        offsetof(struct ibv_ah_attr, grh),
        offsetof(struct ibv_ah_attr, dlid),
        offsetof(struct ibv_ah_attr, sl),
        offsetof(struct ibv_ah_attr, src_path_bits),
        offsetof(struct ibv_ah_attr, static_rate),
        offsetof(struct ibv_ah_attr, is_global),
        offsetof(struct ibv_ah_attr, port_num));
    printf("\n=== ibv_global_route field offsets ===\n");
    printf("  dgid=%zu flow_label=%zu sgid_index=%zu hop_limit=%zu traffic_class=%zu\n",
        offsetof(struct ibv_global_route, dgid),
        offsetof(struct ibv_global_route, flow_label),
        offsetof(struct ibv_global_route, sgid_index),
        offsetof(struct ibv_global_route, hop_limit),
        offsetof(struct ibv_global_route, traffic_class));

    printf("\n=== ibv_qp_init_attr_ex field offsets ===\n");
    printf("  qp_context=%zu send_cq=%zu recv_cq=%zu srq=%zu cap=%zu\n",
        offsetof(struct ibv_qp_init_attr_ex, qp_context),
        offsetof(struct ibv_qp_init_attr_ex, send_cq),
        offsetof(struct ibv_qp_init_attr_ex, recv_cq),
        offsetof(struct ibv_qp_init_attr_ex, srq),
        offsetof(struct ibv_qp_init_attr_ex, cap));
    printf("  qp_type=%zu sq_sig_all=%zu comp_mask=%zu pd=%zu xrcd=%zu\n",
        offsetof(struct ibv_qp_init_attr_ex, qp_type),
        offsetof(struct ibv_qp_init_attr_ex, sq_sig_all),
        offsetof(struct ibv_qp_init_attr_ex, comp_mask),
        offsetof(struct ibv_qp_init_attr_ex, pd),
        offsetof(struct ibv_qp_init_attr_ex, xrcd));
    printf("  create_flags=%zu max_tso_header=%zu rwq_ind_tbl=%zu rx_hash_conf=%zu source_qpn=%zu send_ops_flags=%zu\n",
        offsetof(struct ibv_qp_init_attr_ex, create_flags),
        offsetof(struct ibv_qp_init_attr_ex, max_tso_header),
        offsetof(struct ibv_qp_init_attr_ex, rwq_ind_tbl),
        offsetof(struct ibv_qp_init_attr_ex, rx_hash_conf),
        offsetof(struct ibv_qp_init_attr_ex, source_qpn),
        offsetof(struct ibv_qp_init_attr_ex, send_ops_flags));

    printf("\n=== ibv_cq_init_attr_ex field offsets ===\n");
    printf("  cqe=%zu cq_context=%zu channel=%zu comp_vector=%zu\n",
        offsetof(struct ibv_cq_init_attr_ex, cqe),
        offsetof(struct ibv_cq_init_attr_ex, cq_context),
        offsetof(struct ibv_cq_init_attr_ex, channel),
        offsetof(struct ibv_cq_init_attr_ex, comp_vector));
    printf("  wc_flags=%zu comp_mask=%zu flags=%zu parent_domain=%zu\n",
        offsetof(struct ibv_cq_init_attr_ex, wc_flags),
        offsetof(struct ibv_cq_init_attr_ex, comp_mask),
        offsetof(struct ibv_cq_init_attr_ex, flags),
        offsetof(struct ibv_cq_init_attr_ex, parent_domain));

    printf("\n=== constants we depend on ===\n");
    printf("IBV_QPT_DRIVER                     = %d\n", IBV_QPT_DRIVER);
    printf("IBV_QP_INIT_ATTR_PD                = 0x%x\n", IBV_QP_INIT_ATTR_PD);
    printf("IBV_QP_INIT_ATTR_SEND_OPS_FLAGS    = 0x%x\n", IBV_QP_INIT_ATTR_SEND_OPS_FLAGS);
    printf("IBV_QP_EX_WITH_RDMA_READ           = 0x%lx\n", (unsigned long)IBV_QP_EX_WITH_RDMA_READ);
    printf("IBV_WC_EX_WITH_BYTE_LEN            = 0x%x\n", IBV_WC_EX_WITH_BYTE_LEN);
    printf("EFADV_QP_DRIVER_TYPE_SRD           = %d\n", EFADV_QP_DRIVER_TYPE_SRD);
    printf("EFADV_DEVICE_ATTR_CAPS_RDMA_READ   = 0x%x\n", EFADV_DEVICE_ATTR_CAPS_RDMA_READ);
    return 0;
}
