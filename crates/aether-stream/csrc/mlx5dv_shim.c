/* mlx5 direct-verbs capability probe, compiled only under the `mlx5dv`
 * Cargo feature (build.rs links libmlx5 alongside).
 *
 * Exposes what the fast-post paths need to know — the provider flag word
 * and how many dynamic BlueFlame doorbell registers the device offers —
 * without the Rust side touching the mlx5dv structs. Per-QP BlueFlame
 * mapping (`mlx5dv_init_obj`) builds on these caps on real ConnectX
 * hardware.
 */

#include <errno.h>
#include <stdint.h>
#include <infiniband/mlx5dv.h>

int aether_mlx5dv_query(struct ibv_context *ctx,
                        uint32_t *max_dynamic_bfregs,
                        uint64_t *flags) {
    if (!mlx5dv_is_supported(ctx->device)) {
        return ENOTSUP;
    }
    struct mlx5dv_context dv = {0};
    dv.comp_mask = MLX5DV_CONTEXT_MASK_DYN_BFREGS;
    int rc = mlx5dv_query_device(ctx, &dv);
    if (rc != 0) return rc;
    *max_dynamic_bfregs =
        (dv.comp_mask & MLX5DV_CONTEXT_MASK_DYN_BFREGS) ? dv.max_dynamic_bfregs : 0;
    *flags = dv.flags;
    return 0;
}
