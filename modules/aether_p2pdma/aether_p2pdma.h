/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef _UAPI_AETHER_P2PDMA_H
#define _UAPI_AETHER_P2PDMA_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define AETHER_P2PDMA_BDF_LEN 32

/* Userspace → kernel request. */
struct aether_p2pdma_req {
	char producer_bdf[AETHER_P2PDMA_BDF_LEN];
	char consumer_bdf[AETHER_P2PDMA_BDF_LEN];
	__s32 dmabuf_fd;
	__u32 maximum_distance;
	__u8 require_iommu;
	__u8 _pad[3];
};

/* Kernel → userspace response. Status mirrors P2pdmaPath in Rust. */
#define AETHER_P2PDMA_OK 0
#define AETHER_P2PDMA_TOO_FAR 1
#define AETHER_P2PDMA_NO_IOMMU 2
#define AETHER_P2PDMA_ACS_REDIRECTED 3
#define AETHER_P2PDMA_UNSUPPORTED 4

struct aether_p2pdma_resp {
	__u32 status;
	__u32 distance;
	__u64 peer_bus_addr;
};

struct aether_p2pdma_ioctl {
	struct aether_p2pdma_req req;
	struct aether_p2pdma_resp resp;
};

#define AETHER_P2PDMA_IOCTL_MAGIC 0xAE
#define AETHER_P2PDMA_VALIDATE \
	_IOWR(AETHER_P2PDMA_IOCTL_MAGIC, 1, struct aether_p2pdma_ioctl)

#endif /* _UAPI_AETHER_P2PDMA_H */
