/* SPDX-License-Identifier: Apache-2.0 */
/*
 * K2.3 BlueField-3 DPA edge-parse stub (KERNELS.md).
 *
 * This is the device-side companion to
 * aethergraph_core::internal::device::rdma::flexio::FlexIoHost.
 * On a BF3 box it is compiled with the DOCA FlexIO / DPA toolchain and
 * attached by FlexIoHost::attach. Without that SDK it remains a readable
 * reference of the control-block and delta-header ABI.
 */

#include <stdint.h>

#define DPA_DELTA_MAGIC 0x41454450u /* ADEP */
#define DPA_FLAG_DEDUP 1u

struct dpa_control {
	uint64_t staging_va;
	uint64_t staging_bytes;
	uint64_t meta; /* field_mask | dedup<<32 | csr_delta_bytes<<33 */
	uint64_t magic;
};

struct dpa_delta_header {
	uint32_t magic;
	uint32_t n_edges;
	uint32_t flags;
	uint32_t reserved;
};

/* Entry called by FlexIO runtime once per batch of NIC packets. */
void aether_dpa_parse_edges(const struct dpa_control *ctl, const uint8_t *pkt,
			    uint32_t pkt_len)
{
	struct dpa_delta_header *hdr;
	uint32_t field_mask;
	int dedup;

	if (!ctl || ctl->magic != DPA_DELTA_MAGIC || !ctl->staging_va)
		return;
	if (pkt_len < 16)
		return;

	field_mask = (uint32_t)(ctl->meta & 0xffffffffu);
	dedup = (int)((ctl->meta >> 32) & 1u);
	(void)field_mask;
	(void)pkt;

	hdr = (struct dpa_delta_header *)(uintptr_t)ctl->staging_va;
	hdr->magic = DPA_DELTA_MAGIC;
	hdr->n_edges = 0; /* filled as edges are parsed */
	hdr->flags = dedup ? DPA_FLAG_DEDUP : 0;
	hdr->reserved = 0;
}
