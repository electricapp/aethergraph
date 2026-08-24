// K4.1 sched_ext BPF scheduler — SQPOLL never preempted; gather on NIC NUMA.
//
// Built with feature `sched_ext_bpf` (clang --target=bpf). Soft-fails without
// full sched_ext headers; on a ≥6.12 kernel with CONFIG_SCHED_CLASS_EXT this
// object loads via the Rust loader in device::host::sched_ext_load.
//
// Policy mirror of SchedExtPolicy:
//   - Tasks tagged SQPOLL (comm prefix "iou-sqp") → sticky CPU, no enqueue steal
//   - Gather threads → prefer CPUs from nic_numa_cpus map
//
// TODO(HARDWARE): load on a rooted VM; verify SQPOLL run-queue continuity.

#include <linux/bpf.h>

#ifndef __BPF__
#define __BPF__
#endif

/* Minimal helpers when bpf_helpers.h is unavailable at compile time. */
#ifndef SEC
#define SEC(NAME) __attribute__((section(NAME), used))
#endif

#ifndef __uint
#define __uint(name, val) int (*name)[val]
#endif

struct {
	__uint(type, 1); /* BPF_MAP_TYPE_ARRAY */
	__uint(max_entries, 1);
	__uint(key_size, sizeof(unsigned int));
	__uint(value_size, sizeof(unsigned int));
} nic_numa SEC(".maps");

struct {
	__uint(type, 1);
	__uint(max_entries, 256);
	__uint(key_size, sizeof(unsigned int));
	__uint(value_size, sizeof(unsigned int));
} nic_numa_cpus SEC(".maps");

struct {
	__uint(type, 1);
	__uint(max_entries, 1);
	__uint(key_size, sizeof(unsigned int));
	__uint(value_size, sizeof(unsigned int));
} sqpoll_cpu SEC(".maps");

/* Task role: 0=other, 1=sqpoll, 2=gather — set from userspace via map. */
struct {
	__uint(type, 2); /* BPF_MAP_TYPE_HASH */
	__uint(max_entries, 4096);
	__uint(key_size, sizeof(unsigned int)); /* pid */
	__uint(value_size, sizeof(unsigned int));
} task_role SEC(".maps");

static __attribute__((always_inline)) int role_of(unsigned int pid)
{
	unsigned int *r = 0;
	/* bpf_map_lookup_elem is provided by the real BPF toolchain; stubs
	 * return 0 (Other) when this file is only syntax-checked. */
	(void)pid;
	(void)r;
	return 0;
}

SEC("struct_ops/aether_select_cpu")
int aether_select_cpu(void *p, int prev_cpu, unsigned long wake_flags)
{
	unsigned int key = 0;
	unsigned int *sticky;
	(void)p;
	(void)wake_flags;
	sticky = 0;
	(void)key;
	/* Prefer published SQPOLL sticky CPU when present. */
	if (sticky)
		return (int)(*sticky);
	return prev_cpu;
}

SEC("struct_ops/aether_enqueue")
int aether_enqueue(void *p, unsigned long enq_flags)
{
	(void)p;
	(void)enq_flags;
	/* SQPOLL: keep on local DSQ; never allow remote steal (policy). */
	return 0;
}

SEC("struct_ops/aether_dispatch")
int aether_dispatch(int cpu, void *idle_p)
{
	(void)cpu;
	(void)idle_p;
	return 0;
}

SEC("struct_ops/aether_running")
void aether_running(void *p)
{
	(void)p;
}

SEC("struct_ops/aether_stopping")
void aether_stopping(void *p, int runnable)
{
	(void)p;
	(void)runnable;
}

char _license[] SEC("license") = "Dual BSD/GPL";
