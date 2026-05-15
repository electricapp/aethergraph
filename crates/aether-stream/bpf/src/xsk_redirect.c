// SPDX-License-Identifier: GPL-2.0
//
// XDP program that hands every incoming frame on the bound interface to the
// AF_XDP socket registered in `xsks_map` for that RX queue.
//
// Userspace inserts the socket fd at `xsks_map[queue_id]` before calling
// `bind(AF_XDP, …)`. Without this redirect step, `bind` returns `EINVAL`
// on veth and the socket never sees packets.
//
// Build (Linux + clang with BPF target):
//
//   clang --target=bpf -O2 -g -c bpf/src/xsk_redirect.c -o xsk_redirect.bpf.o
//
// `build.rs` runs the same command behind `feature = "xdp_bpf"`.

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, 64);
} xsks_map SEC(".maps");

SEC("xdp")
int xsk_redirect_prog(struct xdp_md *ctx)
{
    // Send frames into the AF_XDP socket bound to this RX queue, falling
    // back to the kernel stack (XDP_PASS) when no socket is registered.
    return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
}

char _license[] SEC("license") = "GPL";
